import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCVRPTW(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    service_times: torch.Tensor
    demand: torch.Tensor
    time_window_start: torch.Tensor
    time_window_finish: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    total_service_times: torch.Tensor
    total_delay_times: torch.Tensor
    total_early_times: torch.Tensor
    cur_coord: torch.Tensor
    cur_time: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    VEHICLE_CAPACITY = 1.0  # Hardcoded
    DISTANCE_COST_COEF = 100
    SERVICE_TIME_COST_COEF = 0
    DELAY_COEF = 0.5
    EARLY_COEF = 0.1
    MAX_NUM_OF_VEHICLES = 1

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
            )
        return super(StateCVRPTW, self).__getitem__(key)

    @staticmethod
    def initialize(input, vehicle_count=MAX_NUM_OF_VEHICLES, visited_dtype=torch.uint8):

        depot = input['depot']
        loc = input['loc']
        demand = input['demand']
        time_window_start = input['timeWindowStart']
        time_window_finish = input['timeWindowFinish']
        depot_start_time = input['depotStartTime']
        depot_finish_time = input['depotFinishTime']
        service_time = input['serviceTime']

        batch_size, n_loc, _ = loc.size()
        return StateCVRPTW(
            coords=torch.cat((depot[:, None, :], loc), -2),
            demand=demand,
            time_window_start=torch.cat((depot_start_time, time_window_start), -1),
            time_window_finish=torch.cat((depot_finish_time, time_window_finish), -1),
            service_times=torch.cat((torch.zeros((batch_size, 1), device=loc.device), service_time), -1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, vehicle_count, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1, vehicle_count),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, vehicle_count, device=loc.device),
            total_service_times=torch.zeros(batch_size, 1, vehicle_count, device=loc.device),
            total_delay_times=torch.zeros(batch_size, 1, vehicle_count, device=loc.device),
            total_early_times=torch.zeros(batch_size, 1, vehicle_count, device=loc.device),
            cur_coord=(input['depot'][:, None, None, :]).repeat([1, 1, vehicle_count, 1]),  # Add step dimension
            cur_time=torch.zeros(batch_size, vehicle_count, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):
        assert self.all_finished()
        distance_cost = self.DISTANCE_COST_COEF*((self.lengths + (self.coords[self.ids, 0, :] -
                                                                  self.cur_coord).norm(p=2, dim=-1)).sum(dim=-1))
        time_cost = self.SERVICE_TIME_COST_COEF*(self.total_service_times.sum(dim=-1))
        delay_cost = self.DELAY_COEF*(self.total_delay_times.sum(dim=-1))
        early_cost = self.EARLY_COEF*(self.total_early_times.sum(dim=-1))

        return distance_cost + time_cost + delay_cost + early_cost

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        vehicle_index = selected // self.MAX_NUM_OF_VEHICLES
        selected_node = selected % self.MAX_NUM_OF_VEHICLES
        selected = selected_node[:, None]  # Add dimension for step
        prev_a = selected
        n_loc = self.demand.size(-1)  # Excludes depot

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        cur_time = self.cur_time[self.ids, vehicle_index] + \
                   (cur_coord - self.cur_coord[self.ids, 0, vehicle_index]).norm(p=2, dim=-1)
        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]
        lengths = self.lengths[self.ids, 0, vehicle_index] +\
                  (cur_coord - self.cur_coord[self.ids, 0, vehicle_index]).norm(p=2, dim=-1)  # (batch_dim, 1)
        total_service_times = self.total_service_times[self.ids, 0, vehicle_index] +\
                              self.service_times[self.ids, selected]
        delay = self.time_window_finish[self.ids, selected] - cur_time
        early = cur_time - self.time_window_start[self.ids, selected]
        total_delay_times = self.total_delay_times[self.ids, 0, vehicle_index] + delay*(delay > 0).float()
        total_early_times = self.total_early_times[self.ids, 0, vehicle_index] + early*(early > 0).float()

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        #selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]

        # Increase capacity if depot is not visited, otherwise set to 0
        #used_capacity = torch.where(selected == 0, 0, self.used_capacity + selected_demand)
        used_capacity = (self.used_capacity[self.ids, 0, vehicle_index] + selected_demand) * (prev_a != 0).float()

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        prev_a_tmp = self.prev_a
        prev_a_tmp[self.ids, 0, vehicle_index] = prev_a
        used_capacity_tmp = self.used_capacity
        used_capacity_tmp[self.ids, 0, vehicle_index] = used_capacity
        lengths_tmp = self.lengths
        lengths_tmp[self.ids, 0, vehicle_index] = lengths
        cur_coord_tmp = self.cur_coord
        cur_coord_tmp[self.ids, 0, vehicle_index] = cur_coord
        total_service_times_tmp = self.total_service_times
        total_service_times_tmp[self.ids, 0, vehicle_index] = total_service_times
        total_early_times_tmp = self.total_early_times
        total_early_times_tmp[self.ids, 0, vehicle_index] = total_early_times
        total_delay_times_tmp = self.total_delay_times
        total_delay_times_tmp[self.ids, 0, vehicle_index] = total_delay_times

        return self._replace(
            prev_a=prev_a_tmp, used_capacity=used_capacity_tmp, visited_=visited_,
            lengths=lengths_tmp, cur_coord=cur_coord_tmp, i=self.i + 1, total_service_times=total_service_times_tmp,
            total_delay_times=total_delay_times_tmp, total_early_times=total_early_times_tmp
        )

    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))


        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, 0] > self.VEHICLE_CAPACITY)
        # eta = (self.coords[:, 1:, :] - self.cur_coord).norm(p=2, dim=-1)
        # not_started_time = (self.time_window_start[self.ids, 1:] > self.cur_time[:, :, None] + eta[:, None, :])
        # closed_time = (self.time_window_finish[self.ids, 1:] < self.cur_time[:, :, None] + eta[:, None, :])
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap
        # | not_started_time | closed_time

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a[:, :, 0] == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        mask = torch.cat((mask_depot[:, :, None], mask_loc), -1)

        for i in range(1, self.MAX_NUM_OF_VEHICLES):
            exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, i] > self.VEHICLE_CAPACITY)
            mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap
            mask_depot = (self.prev_a[:, :, i] == 0) & ((mask_loc == 0).int().sum(-1) > 0)

            mask = torch.cat((mask, torch.cat((mask_depot[:, :, None], mask_loc), -1)), dim=-1)

        return mask

    def construct_solutions(self, actions):
        return actions
