from src.fat_tree import FatTree
class VmPair:
    uuid = 0  # Static class variable to track unique ids

    def __init__(self, first_vm_location, second_vm_location, traffic_rate):
        # Assign a unique ID and increment the static variable
        self.id = VmPair.uuid
        VmPair.uuid += 1
        
        self.first_vm_location = first_vm_location
        self.second_vm_location = second_vm_location
        self.traffic_rate = traffic_rate
    
    def get_communication_cost(self, fat_tree):
        first_vnf = fat_tree.vnfs[0]
        last_vnf = fat_tree.vnfs[fat_tree.vnf_count - 1]
        #ingress
        cost = FatTree.distance(self.first_vm_location, first_vnf, True)*self.traffic_rate
        #egress 
        cost+= FatTree.distance(last_vnf, self.second_vm_location, True) * self.traffic_rate
    