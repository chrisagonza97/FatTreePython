from src.fat_tree import FatTree
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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
                tree = FatTree(k=16, vm_pair_count=1000 * (i + 1), vnf_capacity=3, vnf_count=3, pm_capacity=80)
                tree.set_traffic_range(0, 1000)
                ff_results[i][j]=tree.create_sized_pairs_ff_place(lower_bound=1, upper_bound=10)
                pal_results[i][j]=tree.create_pairs_sized_pal_place(lower_bound=1, upper_bound=10)

        # Compute means and standard errors
        ff_means = ff_results.mean(axis=1)
        pal_means = pal_results.mean(axis=1)

        # Calculate standard errors
        ff_std = ff_results.std(axis=1)
        pal_std = pal_results.std(axis=1)

        # Calculate 95% confidence intervals
        # For 10 samples, use t-distribution with 9 degrees of freedom
        # t-value for 95% CI with df=9 is approximately 2.262
        
        t_value = stats.t.ppf(0.975, df=9)  # 0.975 for two-tailed 95% CI
        ff_ci = t_value * ff_std / np.sqrt(10)
        pal_ci = t_value * pal_std / np.sqrt(10)

        # Define x-axis labels
        x_labels = [1000, 2000, 3000, 4000]
        x = np.arange(len(x_labels))  # the label locations
        width = 0.35  # the width of the bars

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, ff_means, width, yerr=ff_ci, label='First-Fit', capsize=5)
        rects2 = ax.bar(x + width/2, pal_means, width, yerr=pal_ci, label='Next-Fit', capsize=5)

        # Add labels, title, and legend
        ax.set_ylabel('Average Communication Cost')
        ax.set_xlabel('Number of VM Pairs')
        ax.set_title('Comparison of Placement Algorithms')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()

        # Add grid for better readability
        ax.grid(True, axis='y', alpha=0.3)

        # Show plot
        plt.tight_layout()
        plt.show()

        # Save plot to png
        fig.savefig('placement_comparison_pairs.png', dpi=300)
        
    @staticmethod
    def placement_compare_plot_capacity():
        # Generate a plot where y axis is total communication cost of configuration
        # x axis is PM capacity
        # vary the PM capacity with 20, 40, 60, 80
        # 3 VNFs, k=16, 1000 VM pairs
        ff_results = np.empty((4,10))
        pal_results = np.empty((4,10))
        capacities = [20, 40, 60, 80]
        
        for i in range(4):
            for j in range(10):
                tree = FatTree(k=16, vm_pair_count=1000, vnf_capacity=3, vnf_count=3, pm_capacity=capacities[i])
                tree.set_traffic_range(0, 1000)
                ff_results[i][j] = tree.create_sized_pairs_ff_place(lower_bound=1, upper_bound=10)
                pal_results[i][j] = tree.create_pairs_sized_pal_place(lower_bound=1, upper_bound=10)

        # Compute means and standard errors
        ff_means = ff_results.mean(axis=1)
        pal_means = pal_results.mean(axis=1)

        # Calculate standard errors
        ff_std = ff_results.std(axis=1)
        pal_std = pal_results.std(axis=1)

        # Calculate 95% confidence intervals
        # For 10 samples, use t-distribution with 9 degrees of freedom
        t_value = stats.t.ppf(0.975, df=9)  # 0.975 for two-tailed 95% CI
        ff_ci = t_value * ff_std / np.sqrt(10)
        pal_ci = t_value * pal_std / np.sqrt(10)

        # Define x-axis labels
        x_labels = capacities
        x = np.arange(len(x_labels))  # the label locations
        width = 0.35  # the width of the bars

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, ff_means, width, yerr=ff_ci, label='First-Fit', capsize=5)
        rects2 = ax.bar(x + width/2, pal_means, width, yerr=pal_ci, label='Next-Fit', capsize=5)

        # Add labels, title, and legend
        ax.set_ylabel('Average Communication Cost')
        ax.set_xlabel('PM Capacity')
        ax.set_title('Comparison of Placement Algorithms (1000 VM pairs)')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()

        # Add grid for better readability
        ax.grid(True, axis='y', alpha=0.3)

        # Show plot
        plt.tight_layout()
        plt.show()

        # Save plot to png
        fig.savefig('placement_comparison_capacity.png', dpi=300)
        
    @staticmethod
    def main():
        # Creating an instance of FatTree
        #tree = FatTree(8, 100, 3, 3, 40)
        #tree.set_traffic_range(0, 1000)
        #tree.create_pairs_pal_place()
        #tree.create_sized_pairs_ff_place(lower_bound=1, upper_bound=10)
        #tree.create_pairs_sized_pal_place(lower_bound=1, upper_bound=10)
        App.placement_compare_plot()
        App.placement_compare_plot_capacity()
        #tree.cs2_migration()
        #tree.ac_migration()
        #state = tree.get_state()
        #print(state)

if __name__ == "__main__":
    App.main()
