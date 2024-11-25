import torch
import numpy as np
import random


from .core_switch import CoreSwitch
from .agg_switch import AggregateSwitch
from .edge_switch import EdgeSwitch
from .phys_machine import PhysicalMachine
from .vm_pair import VmPair


class FatTree:
    def __init__(self, k, vm_pair_count, vnf_capacity, vnf_count, pm_capacity):
        self.uuid = 0
        self.k = k
        self.vm_pair_count = vm_pair_count
        self.vnf_capacity = vnf_capacity
        self.vnf_count = vnf_count
        self.pm_capacity = pm_capacity
        self.migration_coefficient= 10
        
        self.first_pm = (k * k) // 4 + (k * k // 2) + (k * k // 2)
        self.last_pm = self.first_pm + (k * k * k) // 4 - 1
        self.pm_count = self.last_pm - self.first_pm + 1
        
        tree_size = (k * k // 4) + (k // 2 * k) + (k // 2 * k) + (k * k * k // 4)
        self.tree = np.empty(tree_size, dtype=object)  # Optimized using NumPy arrays
        self.vnfs = np.empty(vnf_count, dtype=int)
        self.vm_pairs = np.empty(vm_pair_count, dtype=object)
        
        self.traffic_low = 0
        self.traffic_high = 0
        
        self.build_tree()  # Initializing the tree
        self.place_vnfs()  # Placing VNFs
    
    def place_vnfs(self):
        # random placement using NumPy
        for i in range(self.vnf_count):
            flag = True
            while flag:
                random_node = np.random.randint(0, self.first_pm)
                if random_node not in self.vnfs:  # NumPy's fast lookup
                    self.vnfs[i] = random_node
                    flag = False

    def build_tree(self):
        # Add core switches
        for i in range(self.k * self.k // 4):
            self.add_to_tree("core", -1, -1)
        
        # Add aggregate switches
        pod_count = -1
        core_count = 0
        for i in range(self.k // 2 * self.k):
            if i % (self.k // 2) == 0:
                pod_count += 1
                core_count = 0
            self.add_to_tree("aggregate", pod_count, -1)
            
            # Create edges between aggregate and core switches
            for j in range(self.k // 2):
                temp = self.tree[self.uuid - 1]
                temp.core_edges[j] = core_count

                core = self.tree[core_count]
                core.add_aggr_edge(self.uuid - 1)
                core_count += 1
        
        # Add edge switches
        pod_count = -1
        edge_id = self.uuid - 1
        for i in range(self.k // 2 * self.k):
            if i % (self.k // 2) == 0:
                pod_count += 1
            self.add_to_tree("edge", pod_count, -1)
        
        # Add physical machines
        pod_count = -1
        for i in range(self.k * self.k * self.k // 4):
            if i % (self.k * self.k // 4) == 0:
                pod_count += 1
            if i % (self.k // 2) == 0:
                edge_id += 1
            self.add_to_tree("pm", pod_count, edge_id)

    def add_to_tree(self, node_type, pod, edge_id):
        # Using NumPy for efficient array management
        if node_type == "core":
            self.tree[self.uuid] = CoreSwitch(self.uuid, self.k)
        elif node_type == "aggregate":
            self.tree[self.uuid] = AggregateSwitch(self.uuid, pod, self.k)
        elif node_type == "edge":
            self.tree[self.uuid] = EdgeSwitch(self.uuid, pod)
        elif node_type == "pm":
            self.tree[self.uuid] = PhysicalMachine(self.uuid, pod, edge_id, self.pm_capacity)
        self.uuid += 1

    def get_pair_cost(self, pm1, pm2):
        cost=0
        cost+=self.distance(pm1, self.vnfs[0], True)
        cost+=self.distance(self.vnfs[self.vnf_count-1], pm2, True)
    @staticmethod
    def distance(one, two):
        one_id, two_id = one.id, two.id

        # If the two nodes are the same
        if one_id == two_id:
            return 0

        # Core to Core
        if isinstance(one, CoreSwitch) and isinstance(two, CoreSwitch):
            for aggr_id in one.aggr_edges:
                if aggr_id in two.aggr_edges:
                    return 2  # Both core switches are connected to the same aggregate switch
            return 4  # Different aggregate switches

        # Aggregate to Aggregate
        if isinstance(one, AggregateSwitch) and isinstance(two, AggregateSwitch):
            if one.pod == two.pod:
                return 2  # Both aggregate switches are in the same pod
            for core_id in one.core_edges:
                if core_id in two.core_edges:
                    return 2  # Connected to the same core switch
            return 4  # Different core switches

        # Edge to Edge
        if isinstance(one, EdgeSwitch) and isinstance(two, EdgeSwitch):
            if one.pod == two.pod:
                return 2  # Both edge switches are in the same pod
            return 4  # Different pods

        # Physical Machine to Physical Machine
        if isinstance(one, PhysicalMachine) and isinstance(two, PhysicalMachine):
            if one.edge_id == two.edge_id:
                return 2  # Both physical machines are under the same edge switch
            if one.pod == two.pod:
                return 4  # Both physical machines are in the same pod
            return 6  # Different pods

        # Core to Aggregate or Aggregate to Core
        if (isinstance(one, CoreSwitch) and isinstance(two, AggregateSwitch)) or (isinstance(one, AggregateSwitch) and isinstance(two, CoreSwitch)):
            if isinstance(one, CoreSwitch):
                return 1 if two.id in one.aggr_edges else 3
            return 1 if one.id in two.core_edges else 3

        # Core to Edge or Edge to Core
        if (isinstance(one, CoreSwitch) and isinstance(two, EdgeSwitch)) or (isinstance(one, EdgeSwitch) and isinstance(two, CoreSwitch)):
            return 2  # Distance between any core switch and any edge switch is always 2

        # Core to Physical Machine or Physical Machine to Core
        if (isinstance(one, CoreSwitch) and isinstance(two, PhysicalMachine)) or (isinstance(one, PhysicalMachine) and isinstance(two, CoreSwitch)):
            return 3  # Distance between any core switch and any physical machine is always 3

        # Aggregate to Edge or Edge to Aggregate
        if (isinstance(one, AggregateSwitch) and isinstance(two, EdgeSwitch)) or (isinstance(one, EdgeSwitch) and isinstance(two, AggregateSwitch)):
            if isinstance(one, AggregateSwitch):
                return 1 if one.pod == two.pod else 3
            return 1 if two.pod == one.pod else 3

        # Aggregate to Physical Machine or Physical Machine to Aggregate
        if (isinstance(one, AggregateSwitch) and isinstance(two, PhysicalMachine)) or (isinstance(one, PhysicalMachine) and isinstance(two, AggregateSwitch)):
            if isinstance(one, AggregateSwitch):
                return 2 if one.pod == two.pod else 4
            return 2 if two.pod == one.pod else 4

        # Edge to Physical Machine or Physical Machine to Edge
        if (isinstance(one, EdgeSwitch) and isinstance(two, PhysicalMachine)) or (isinstance(one, PhysicalMachine) and isinstance(two, EdgeSwitch)):
            if isinstance(one, EdgeSwitch):
                return 1 if one.id == two.edge_id else (3 if one.pod == two.pod else 5)
            return 1 if two.id == one.edge_id else (3 if two.pod == one.pod else 5)

        # Default case (should never reach this)
        return -1
    
    @staticmethod
    def distance(one, two, flag):
        one = tree[one]
        two = tree[two]
        one_id, two_id = one.id, two.id

        # If the two nodes are the same
        if one_id == two_id:
            return 0

        # Core to Core
        if isinstance(one, CoreSwitch) and isinstance(two, CoreSwitch):
            for aggr_id in one.aggr_edges:
                if aggr_id in two.aggr_edges:
                    return 2  # Both core switches are connected to the same aggregate switch
            return 4  # Different aggregate switches

        # Aggregate to Aggregate
        if isinstance(one, AggregateSwitch) and isinstance(two, AggregateSwitch):
            if one.pod == two.pod:
                return 2  # Both aggregate switches are in the same pod
            for core_id in one.core_edges:
                if core_id in two.core_edges:
                    return 2  # Connected to the same core switch
            return 4  # Different core switches

        # Edge to Edge
        if isinstance(one, EdgeSwitch) and isinstance(two, EdgeSwitch):
            if one.pod == two.pod:
                return 2  # Both edge switches are in the same pod
            return 4  # Different pods

        # Physical Machine to Physical Machine
        if isinstance(one, PhysicalMachine) and isinstance(two, PhysicalMachine):
            if one.edge_id == two.edge_id:
                return 2  # Both physical machines are under the same edge switch
            if one.pod == two.pod:
                return 4  # Both physical machines are in the same pod
            return 6  # Different pods

        # Core to Aggregate or Aggregate to Core
        if (isinstance(one, CoreSwitch) and isinstance(two, AggregateSwitch)) or (isinstance(one, AggregateSwitch) and isinstance(two, CoreSwitch)):
            if isinstance(one, CoreSwitch):
                return 1 if two.id in one.aggr_edges else 3
            return 1 if one.id in two.core_edges else 3

        # Core to Edge or Edge to Core
        if (isinstance(one, CoreSwitch) and isinstance(two, EdgeSwitch)) or (isinstance(one, EdgeSwitch) and isinstance(two, CoreSwitch)):
            return 2  # Distance between any core switch and any edge switch is always 2

        # Core to Physical Machine or Physical Machine to Core
        if (isinstance(one, CoreSwitch) and isinstance(two, PhysicalMachine)) or (isinstance(one, PhysicalMachine) and isinstance(two, CoreSwitch)):
            return 3  # Distance between any core switch and any physical machine is always 3

        # Aggregate to Edge or Edge to Aggregate
        if (isinstance(one, AggregateSwitch) and isinstance(two, EdgeSwitch)) or (isinstance(one, EdgeSwitch) and isinstance(two, AggregateSwitch)):
            if isinstance(one, AggregateSwitch):
                return 1 if one.pod == two.pod else 3
            return 1 if two.pod == one.pod else 3

        # Aggregate to Physical Machine or Physical Machine to Aggregate
        if (isinstance(one, AggregateSwitch) and isinstance(two, PhysicalMachine)) or (isinstance(one, PhysicalMachine) and isinstance(two, AggregateSwitch)):
            if isinstance(one, AggregateSwitch):
                return 2 if one.pod == two.pod else 4
            return 2 if two.pod == one.pod else 4

        # Edge to Physical Machine or Physical Machine to Edge
        if (isinstance(one, EdgeSwitch) and isinstance(two, PhysicalMachine)) or (isinstance(one, PhysicalMachine) and isinstance(two, EdgeSwitch)):
            if isinstance(one, EdgeSwitch):
                return 1 if one.id == two.edge_id else (3 if one.pod == two.pod else 5)
            return 1 if two.id == one.edge_id else (3 if two.pod == one.pod else 5)

        # Default case (should never reach this)
        return -1


    def set_traffic_range(self, traffic_low, traffic_high):
        self.traffic_low = traffic_low
        self.traffic_high = traffic_high

    def create_vm_pairs(self):
        # Using random placement for VMs on physical machines
        for i in range(self.vm_pair_count):
            flag = True
            while flag:
                first = random.randint(self.first_pm, self.last_pm)
                second = random.randint(self.first_pm, self.last_pm)
                if first == second:
                    continue
                first_pm = self.tree[first]
                second_pm = self.tree[second]
                if first_pm.capacity_left <= 0 or second_pm.capacity_left <= 0:
                    continue
                flag = False

            first_pm.add_vm()
            second_pm.add_vm()
            rand_rate = random.randint(self.traffic_low, self.traffic_high)
            self.vm_pairs[i] = VmPair(first, second, rand_rate)

    def randomize_traffic(self):
        # Using NumPy for efficient random traffic generation
        traffic_rates = np.random.randint(self.traffic_low, self.traffic_high + 1, len(self.vm_pairs))
        for i in range(len(self.vm_pairs)):
            self.vm_pairs[i].traffic_rate = traffic_rates[i]

    def cs2_migration(self):
        # Implement migration functionality (placeholder)
        pass


    def ac_migration(self):
        #use Megh state projection
        #there are d basis vectors
        #TODO caller function
        pass

    def get_state(self, curr_pair):
        #State 
        # State includes VM pair locations
        vm_locations = [(vm.first_vm_location, vm.second_vm_location) for vm in self.vm_pairs]
        flat_locations = [item for pair in vm_locations for item in pair]  # Flattened
        flat_locations.append(curr_pair)  # Append current VM pair
        return torch.tensor(flat_locations, dtype=torch.float32)
    
    def get_action_basis(self,curr_pair, pm1, pm2, num_pms):
        # Create a sparse basis vector for action (curr_pair, pm1, pm2)
        basis_vector = torch.zeros(num_pms * num_pms)
        index1 = curr_pair * num_pms + pm1  # Position for VM1
        index2 = curr_pair * num_pms + pm2  # Position for VM2
        basis_vector[index1] = 1  # Assign VM1 to pm1
        basis_vector[index2] = 1  # Assign VM2 to pm2
        return basis_vector

    
    def get_projected_state(self,tree, curr_pair, action_basis_vectors):
        # Get the current state
        state = tree.get_state(curr_pair)
    
         # Project the state into the subspace
        projected_state = torch.matmul(action_basis_vectors.T, state)
        return projected_state

    def get_valid_action_basis(self,tree, curr_pair, num_pms):
        valid_actions = tree.get_valid_actions(curr_pair)
        action_basis_vectors = []

        for action in valid_actions:
            _, pm1, pm2 = action
            basis_vector = self.get_action_basis(curr_pair, pm1, pm2, num_pms)
            action_basis_vectors.append(basis_vector)
    
        return torch.stack(action_basis_vectors)

    
    def get_valid_actions(self, curr_pair):
        actions = []
        #sorted_pairs = self.vm_pairs
        #for i, vmp in enumerate (sorted_pairs):
            #current_pm1, current_pm2 = vmp.first_vm_location, vmp.second_vm_location
            #if i is less than curr_pair, then it has already been assigned a pm
            #if (i<curr_pair):
            #    actions.append((i, current_pm1, current_pm2))
            #    continue
            #if i is curr_pair, then it is its turn to be assigned a pm
            #if (i==curr_pair):
        for pm1 in range(self.first_pm, self.last_pm+1):
            for pm2 in range(self.first_pm, self.last_pm+1):
                if (pm1!=pm2 and self.tree[pm1].capacity_left>0 and self.tree[pm2].capacity_left>0):
                    actions.append((curr_pair, pm1, pm2))
                            
            #elif(i>curr_pair):
            #    actions.append((i,-1,-1))#not assigned a pm, not its turn yet
    
    def do_next_state(self, action):
        curr_pair, pm1, pm2 = action
        self.vm_pairs[curr_pair].first_vm_location = pm1
        self.vm_pairs[curr_pair].second_vm_location = pm2
        self.tree[pm1].add_vm()
        self.tree[pm2].add_vm()

    def get_reward(self, action):
        # Reward is the negative of the difference in communication cost 
        # between the vm pairs old location and new location
        # (old location comm. cost - new location comm. cost)
        # plus the migration cost
        curr_pair, pm1, pm2 = action
        old_comm_cost = self.vm_pairs[curr_pair].get_communication_cost(self)
        new_comm_cost = self.get_pair_cost(pm1, pm2) * self.vm_pairs[curr_pair].traffic_rate
        migration_cost = self.get_pair_cost(self.vm_pairs[curr_pair].first_vm_location, pm1)
        migration_cost += self.get_pair_cost(self.vm_pairs[curr_pair].second_vm_location, pm2)
        migration_cost *= self.migration_coefficient

        return -(old_comm_cost - (new_comm_cost + migration_cost))

    

