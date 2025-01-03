
class VmPair:
    uuid = 0  # Static class variable to track unique ids

    def __init__(self, first_vm_location, second_vm_location, traffic_rate):
        # Assign a unique ID and increment the static variable
        self.id = VmPair.uuid
        VmPair.uuid += 1
        
        self.first_vm_location = first_vm_location
        self.second_vm_location = second_vm_location
        self.traffic_rate = traffic_rate
    
    
    