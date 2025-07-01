from src.fat_tree import FatTree


class App:
    def __init__(self):
        pass

    @staticmethod
    def placement_compare_plot():
        #first generate a plot where  the y axis is total communication cost of configuration
        #x axis is number of VM pairs
        #vary the number of VM pairs with 500, 1000, 1500, 2000
        #3 VNFs, k=16, PM capacity 
        pass

    @staticmethod
    def main():
        # Creating an instance of FatTree
        tree = FatTree(8, 100, 3, 3, 40)
        tree.set_traffic_range(0, 1000)
        #tree.create_pairs_pal_place()
        tree.create_sized_pairs_ff_place(lower_bound=1, upper_bound=10)
        tree.create_pairs_sized_pal_place(lower_bound=1, upper_bound=10)
        App.placement_compare_plot()
        #tree.cs2_migration()
        #tree.ac_migration()
        #state = tree.get_state()
        #print(state)

if __name__ == "__main__":
    App.main()
