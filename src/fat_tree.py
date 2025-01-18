import torch
import numpy as np
import random
import os

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

        self.discount_factor = 0.5
        self.episodes = 100
        self.temperature = 3
        self.epsilon = 0.01
        self.q_table = {}

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

    def get_sorted_pairs(self):
        return sorted(self.vm_pairs, key=lambda vm_pair: vm_pair.traffic_rate, reverse=True)

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
        return cost
    
    
    def distance(self,one, two, flag):
        if flag==True:
            one = self.tree[one]
            two = self.tree[two]
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
        else:
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
        self.calculate_initial_cost()
        self.vmp_mcf_file()
        self.read_mcf_pairs_output()

    def init_ac(self):
        self.d = self.vm_pair_count * 2 * self.pm_count
        self.policy = {vm: {pm: 1 / self.pm_count for pm in range(self.first_pm, self.last_pm + 1)} for vm in self.vm_pairs}
        self.T = np.eye(self.d)
        self.q_table = {}
        self.B = np.eye(self.d)
        #self.phi = np.zeros(self.d)
        self.theta = np.zeros(self.d)
        self.z = np.zeros(self.d)
        self.C = 0
        self.time=1
        #set all pm capacities to empty
        for i in range(self.first_pm, self.last_pm+1):
            self.tree[i].capacity_left = self.pm_capacity

    def select_action(self, curr_vm):
        pm_choices = list(self.policy[curr_vm].keys())
        pm_probs = list(self.policy[curr_vm].values())

        # Check PM capacities and set probabilities to 0 for PMs at capacity
        for i, pm in enumerate(pm_choices):
            if self.tree[pm].capacity_left <= 0:  
                pm_probs[i] = 0

        # Normalize the probabilities to sum to 1
        total_prob = sum(pm_probs)
        if total_prob > 0:
            pm_probs = [p / total_prob for p in pm_probs]
        else:
            raise ValueError("No valid PMs available for VM migration.")

        # Select an action based on the adjusted probabilities
        selected_action = np.random.choice(pm_choices, p=pm_probs)
        #decrement selected PM's capacity 
        self.tree[selected_action].capacity_left -= 1
        return selected_action

        
    def simulate_action(self,actions):
        total_cost = 0

        for i in range(self.actions):
            
            #migration cost
            if(i%2==0):
                total_cost+= self.distance(actions[i], self.vnfs[0], True)
                total_cost*= self.vm_pairs[i//2].traffic_rate
                total_cost+= self.distance(self.vm_pairs[i//2].first_vm_location, actions[i], True) * self.migration_coefficient
            else:
                total_cost+= self.distance(actions[i], self.vnfs[self.vnf_count-1], True)
                total_cost*= self.vm_pairs[i//2].traffic_rate
                total_cost+= self.distance(self.vm_pairs[i//2].second_vm_location, actions[i], True) * self.migration_coefficient

        #save original location, set new location, get next action, get its phi, reset location to original, finally also return the nect phi
        original_locations = []
        next_actions = {}
        current_state = self.get_state()
        for i in range(self.vm_pair_count):
            original_locations.append(self.vm_pairs[i].first_vm_location)
            original_locations.append(self.vm_pairs[i].second_vm_location)
            
            self.vm_pairs[i].first_vm_location = actions[i*2]
            self.vm_pairs[i].second_vm_location = actions[i*2+1]

            

            next_actions[self.vm_pairs_sorted_index[i]*2]= self.select_action(current_state[self.vm_pairs_sorted_index[i]*2])   
            next_actions[self.vm_pairs_sorted_index[i]*2+1]= self.select_action(current_state[self.vm_pairs_sorted_index[i]*2+1]) 
        phi = self.get_phi(next_actions)
        for i in range(self.vm_pair_count):
            self.vm_pairs[i].first_vm_location = original_locations[i*2]
            self.vm_pairs[i].second_vm_location = original_locations[i*2+1]


        return total_cost, phi
    
    def get_phi(self, actions):
        phi = np.zeros((self.vm_pair_count * 2, self.pm_count))
        for i, action in enumerate(actions):
            #vm_pair_index = i // 2
            phi[i,action] = 1
        return phi.flatten()

            

    def ac_migration(self):
        #use Megh state projection
        self.init_ac()
        #there are d basis vectors
        #d = self.vm_pair_count * 2 * self.pm_count
        #initialize policy to equally do all migrations
        #self.policy = {vm: {pm: 1 / self.pm_count for pm in range(self.first_pm, self.last_pm + 1)} for vm in self.vm_pairs}
        
        
        for i in range(self.episodes):
            current_state = self.get_state()
            actions={}
            for i in range(self.vm_pair_count*2):
                if i % 2 == 0:
                    actions[self.vm_pairs_sorted_index[i]*2]= self.select_action(current_state[self.vm_pairs_sorted_index[i]*2]) 
                else:
                    actions[self.vm_pairs_sorted_index[i]*2+1]= self.select_action(current_state[self.vm_pairs_sorted_index[i]*2+1]) 
                

            cost, next_phi = self.simulate_action(actions)
            phi = self.get_phi(actions)

            self.C+=cost
            self.z += cost * phi * cost
            self.B+=np.outer(self.phi, (self.phi - self.discount_factor * next_phi))

            self.theta = self.B + self.z
            self.policy = self.policy_calculator(actions, self.theta)
            #actions = self.select_action(current_state)
            #self.basis_vectors = np.zeros((self.d,len(current_state)))

            #for i in range(self.d):
                #self.basis_vectors[i] = np.random.rand(len(current_state))
            
            #initialize Q table
            

            #for vm_pair in self.vm_pairs:
            #    vm_actions = {}
             #   for pm in range(self.first_pm, self.last_pm + 1):
              #      vm_actions[pm] = 0  # Start with zero reward for all actions
               # self.q_table[vm_pair] = vm_actions
        #pass

    def policy_calculator(self, actions, theta):
        self.temperature *= np.exp(-self.epsilon)
        for i in range(self.d):
        #TODO POLICY CALCULATION

    def get_state(self):
        #State 
        # State includes VM pair locations
        self.sorted = self.get_sorted_pairs()
        self.vm_pairs_sorted_index=[] #index i of this list is the vm pair num of the ith element in sorted
        state = []
        for i in range(self.vm_pair_count):
            #state.append(sorted[i].first_vm_location)
            #state.append(sorted[i].second_vm_location)
            state.append(self.vm_pairs[i].first_vm_location)
            state.append(self.vm_pairs[i].second_vm_location)

            self.vm_pairs_sorted_index.append(self.vm_pairs.index(self.sorted[i]))
        return state
    
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
        old_comm_cost = self.calc_pair_cost(self.vm_pairs[curr_pair])
        #old_comm_cost = self.vm_pairs[curr_pair].get_communication_cost(self)
        new_comm_cost = self.get_pair_cost(pm1, pm2)
        migration_cost = self.get_pair_cost(self.vm_pairs[curr_pair].first_vm_location, pm1)
        migration_cost += self.get_pair_cost(self.vm_pairs[curr_pair].second_vm_location, pm2)
        migration_cost *= self.migration_coefficient

        return -(old_comm_cost - (new_comm_cost + migration_cost))
    
    def calc_pair_cost(self, vm_pair):
        first_vnf = self.vnfs[0]
        last_vnf = self.vnfs[self.vnf_count - 1]
        #ingress
        cost = FatTree.distance(vm_pair.first_vm_location, first_vnf, True)
        #egress 
        cost+= FatTree.distance(last_vnf, vm_pair.second_vm_location, True)
        cost *= vm_pair.traffic_rate
        return cost

    def calc_total_cost(self):
        total_cost = 0
        for i in range(self.vm_pair_count):
            #total_cost += self.vm_pairs[i].get_communication_cost(self)
            self.calc_pair_cost(self.vm_pairs[i])
        return total_cost

    def vmp_mcf_file(self):
        """
        Generates the MCF input file for VM migration and replication.
        """
        arccount = 2 * self.vm_pair_count * self.pm_count  # Edges between VMs and PMs
        arccount += self.vm_pair_count * 2  # Edges from supply node to VM pairs
        arccount += self.pm_count  # Edges between PMs and the demand node

        nodecount = (self.vm_pair_count * 2) + self.pm_count + 2

        firstline = f"p min {nodecount} {arccount}\n"
        secline = f"c min-cost flow problem with {nodecount} nodes and {arccount} arcs \n"
        thirdline = f"n 1 {self.vm_pair_count * 2}\n"
        fourthln = f"c supply of {self.vm_pair_count * 2} at node 1 \n"
        fifthln = f"n {nodecount} {-1 * self.vm_pair_count * 2}\n"
        sixthln = f"c demand of {-1 * self.vm_pair_count * 2} at node {nodecount}\n"
        sevln = "c arc list follows \n"
        eithln = "c arc has <tail> <head> <capacity l.b.> <capacity u.b> <cost> \n"

        firstlns = firstline + secline + thirdline + fourthln + fifthln + sixthln + sevln + eithln

        supplyarcs = []
        countnode = 2

        for i in range(self.vm_pair_count * 2):
            supplyarcs.append(f"a 1 {countnode} 0 1 0 \n")
            countnode += 1

        firstvm = countnode
        vmarcs = ["c arcs from VMs to PMs \n"]

        for i in range(self.vm_pair_count * 2):
            countnode = firstvm
            for j in range(self.pm_count):
                last_val = 0
                vmnum = i // 2

                if i % 2 == 0:  # Ingress VM
                    last_val += (self.migration_coefficient * self.distance(self.tree[self.vm_pairs[vmnum].first_vm_location], self.tree[self.first_pm + j], False))
                    last_val += (self.vm_pairs[vmnum].traffic_rate * self.distance(self.tree[self.first_pm + j], self.tree[self.vnfs[0]], False))
                else:  # Egress VM
                    last_val += (self.migration_coefficient * self.distance(self.tree[self.vm_pairs[vmnum].second_vm_location], self.tree[self.first_pm + j], False))
                    last_val += (self.vm_pairs[vmnum].traffic_rate * self.distance(self.tree[self.first_pm + j], self.tree[self.vnfs[-1]], False))

                vmarcs.append(f"a {i + 2} {countnode} 0 1 {last_val}\n")
                countnode += 1

        pmarcs = ["c arcs from PMs to destination \n"]
        for i in range(self.pm_count):
            pmarcs.append(f"a {i + firstvm} {countnode} 0 {self.pm_capacity} 0 \n")

        output = firstlns + ''.join(supplyarcs) + ''.join(vmarcs) + ''.join(pmarcs)

        try:
            with open("mcf_replication.inp", "w") as file:
                file.write(output)

            print("mcf_replication.inp has been written to in the project root file directory")
        except Exception as e:
            print(f"Failed to write MCF file: {e}")

        if self.pm_count * self.pm_capacity < self.vm_pair_count * 2:
            print("Replication of every VM not possible.")

        #next, exec cs2 with passing in the generated file, and saving the output to a file
        #we want to run cs2 < mcf_replication.inp > output.txt
        os.system("cs2 < mcf_replication.inp > output.txt")

    def calculate_initial_cost(self):
        """
        Calculate and print the total communication cost before migration.
        """
        initial_total_cost = 0

        for vm_pair in self.vm_pairs:
            # Cost of communication for ingress
            ingress_cost = (
                vm_pair.traffic_rate
                * self.distance(
                    self.tree[vm_pair.first_vm_location],
                    self.tree[self.vnfs[0]],
                    False,
                )
            )
            # Cost of communication for egress
            egress_cost = (
                vm_pair.traffic_rate
                * self.distance(
                    self.tree[self.vnfs[-1]],
                    self.tree[vm_pair.second_vm_location],
                    False,
                )
            )
            ordered_cost = 0
            for j in range(len(self.vnfs) - 1):
                ordered_cost += vm_pair.traffic_rate * self.distance(
                    self.tree[self.vnfs[j]], self.tree[self.vnfs[j + 1]], False
                )
            
            initial_total_cost += ingress_cost + egress_cost + ordered_cost

        print(f"The total communication cost before migration is: {initial_total_cost}")
        return initial_total_cost

    def read_mcf_pairs_output(self):
        output_file = "output.txt"

        if not os.path.exists(output_file):
            print(f"File not found: {output_file}")
            return

        placed = 0
        total_cost = 0
        migr_cost = 0

        try:
            with open(output_file, 'r') as file:
                for line in file:
                    line = line.strip()

                    # Ignore comment lines
                    if line.startswith('c') or line.startswith('s'):
                        continue

                    # Parse the line into tokens
                    tokens = line.split()
                    if len(tokens) < 4:
                        continue

                    first_num = int(tokens[1])
                    if first_num == 1:
                        continue  # Ignore supply arc lines

                    vmpair_num = (first_num - 2) // 2  # Calculate VM pair ID
                    second_num = int(tokens[2])
                    third_num = int(tokens[3])

                    # Skip sink node flow
                    if second_num == (2 + (self.vm_pair_count * 2) + self.pm_count):
                        continue

                    pm_num = second_num - ((self.vm_pair_count * 2) + 2)

                    if third_num > 0:
                        # Update placement for VM pairs
                        if first_num % 2 == 0:
                            self.vm_pairs[vmpair_num].mcfMigrVm1Pm = pm_num + self.first_pm
                        else:
                            self.vm_pairs[vmpair_num].mcfMigrVm2Pm = pm_num + self.first_pm
                        placed += 1

            # Calculate migration and total costs
            for i in range(len(self.vm_pairs)):
                migr_cost += self.migration_coefficient * self.distance(
                    self.tree[self.vm_pairs[i].first_vm_location], self.tree[self.vm_pairs[i].mcfMigrVm1Pm], False
                )
                migr_cost += self.migration_coefficient * self.distance(
                    self.tree[self.vm_pairs[i].second_vm_location], self.tree[self.vm_pairs[i].mcfMigrVm2Pm], False
                )

                total_cost += self.vm_pairs[i].traffic_rate * self.distance(
                    self.tree[self.vm_pairs[i].mcfMigrVm1Pm], self.tree[self.vnfs[0]], False
                )
                total_cost += self.vm_pairs[i].traffic_rate * self.distance(
                    self.tree[self.vm_pairs[i].mcfMigrVm2Pm], self.tree[self.vnfs[-1]], False
                )

                for j in range(len(self.vnfs) - 1):
                    total_cost += self.vm_pairs[i].traffic_rate * self.distance(
                        self.tree[self.vnfs[j]], self.tree[self.vnfs[j + 1]], False
                    )

            # Output results
            print(f"Number of VMs placed: {placed}")
            print(f"The MCF total migration cost is: {migr_cost}")
            print(f"The MCF total cost is: {migr_cost + total_cost}")



        except Exception as e:
            print(f"An error occurred: {e}")

