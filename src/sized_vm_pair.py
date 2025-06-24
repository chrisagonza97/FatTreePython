from .vm_pair import VmPair

class SizedVmPair(VmPair):
    def __init__(self, first_vm_location, second_vm_location, traffic_rate, vm_size):
        super().__init__(first_vm_location, second_vm_location, traffic_rate)
        self.vm_size = vm_size  
        
