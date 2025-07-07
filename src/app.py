from src.fat_tree import FatTree
import numpy as np
import matplotlib.pyplot as plt
class App:
    def __init__(self):
        pass

    @staticmethod
    def placement_compare_plot():
        #first generate a plot where  the y axis is total communication cost of configuration
        #x axis is number of VM pairs
        #vary the number of VM pairs with 500, 1000, 1500, 2000
        #3 VNFs, k=16, PM capacity 
        ff_results = np.empty((4,10))
        pal_results = np.empty((4,10))
        for i in range(4):
            for j in range(10):
                tree = FatTree(k=16, vm_pair_count=500 * (i + 1), vnf_capacity=3, vnf_count=3, pm_capacity=200)
                tree.set_traffic_range(0, 1000)
                ff_results[i][j]=tree.create_sized_pairs_ff_place(lower_bound=1, upper_bound=10)
                pal_results[i][j]=tree.create_pairs_sized_pal_place(lower_bound=1, upper_bound=10)
        #plot results as histogram, avergaing the ten runs for each x value
        
        # Compute means
        ff_means = ff_results.mean(axis=1)
        pal_means = pal_results.mean(axis=1)

        # Define x-axis labels
        x_labels = [500, 1000, 1500, 2000]
        x = np.arange(len(x_labels))  # the label locations
        width = 0.35  # the width of the bars

        # Create the plot
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, ff_means, width, label='First Fit')
        rects2 = ax.bar(x + width/2, pal_means, width, label='PAL')

        # Add labels, title, and legend
        ax.set_ylabel('Average Communication Cost')
        ax.set_xlabel('Number of VM Pairs')
        ax.set_title('Comparison of Placement Algorithms')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()

        # Show plot
        plt.tight_layout()
        plt.show()
        
        
    @staticmethod
    def main():
        # Creating an instance of FatTree
        #tree = FatTree(8, 100, 3, 3, 40)
        #tree.set_traffic_range(0, 1000)
        #tree.create_pairs_pal_place()
        #tree.create_sized_pairs_ff_place(lower_bound=1, upper_bound=10)
        #tree.create_pairs_sized_pal_place(lower_bound=1, upper_bound=10)
        App.placement_compare_plot()
        #tree.cs2_migration()
        #tree.ac_migration()
        #state = tree.get_state()
        #print(state)

if __name__ == "__main__":
    App.main()
