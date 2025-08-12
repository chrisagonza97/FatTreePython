import csv
from src.fat_tree import FatTree
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
class App:
    def __init__(self):
        pass

    @staticmethod
    def placement_compare_plot():
        # first generate a plot where the y axis is total communication cost of configuration
        # x axis is number of VM pairs
        # vary the number of VM pairs with 200, 500, 800, 1000
        # 3 VNFs, k=16, PM capacity
        ff_results = np.empty((4,10))
        pal_results = np.empty((4,10))

        ff_results_active_pms = np.empty((4,10))
        pal_results_active_pms = np.empty((4,10))

        x_counts = [200, 500, 800, 1000]
        for i in range(4):
            for j in range(10):
                tree = FatTree(k=16, vm_pair_count=x_counts[i], vnf_capacity=3, vnf_count=3, pm_capacity=12)
                tree.set_traffic_range(0, 1000)
                ff_results[i][j], ff_results_active_pms[i][j] = tree.create_sized_pairs_ff_place(lower_bound=1, upper_bound=8)
                pal_results[i][j], pal_results_active_pms[i][j] = tree.create_pairs_sized_pal_place(lower_bound=1, upper_bound=8)

        # --- Cost plot stats ---
        ff_means = ff_results.mean(axis=1)
        pal_means = pal_results.mean(axis=1)
        ff_std = ff_results.std(axis=1, ddof=1)
        pal_std = pal_results.std(axis=1, ddof=1)

        # --- Active PMs plot stats ---
        ff_active_means = ff_results_active_pms.mean(axis=1)
        pal_active_means = pal_results_active_pms.mean(axis=1)
        ff_active_std = ff_results_active_pms.std(axis=1, ddof=1)
        pal_active_std = pal_results_active_pms.std(axis=1, ddof=1)

        # 95% CI via t (df=9)
        t_value = stats.t.ppf(0.975, df=9)
        ff_ci = t_value * ff_std / np.sqrt(10)
        pal_ci = t_value * pal_std / np.sqrt(10)
        ff_active_ci = t_value * ff_active_std / np.sqrt(10)
        pal_active_ci = t_value * pal_active_std / np.sqrt(10)
        
        
        # === DAT export for GNUplot ===
        with open('CostOverPairs.dat', 'w') as f:
            f.write("# vm_pairs FF_mean FF_CI PAL_mean PAL_CI\n")
            for xi, ffm, ffc, palm, palc in zip(x_counts, ff_means, ff_ci, pal_means, pal_ci):
                f.write(f"{xi} {ffm:.6f} {ffc:.6f} {palm:.6f} {palc:.6f}\n")

        with open('ActiveOverPairs.dat', 'w') as f:
            f.write("# vm_pairs FF_mean FF_CI PAL_mean PAL_CI\n")
            for xi, ffm, ffc, palm, palc in zip(x_counts, ff_active_means, ff_active_ci, pal_active_means, pal_active_ci):
                f.write(f"{xi} {ffm:.6f} {ffc:.6f} {palm:.6f} {palc:.6f}\n")


        # --- Cost figure ---
        x = np.arange(len(x_counts))
        width = 0.35

        fig_cost, ax_cost = plt.subplots(figsize=(10, 6))
        ax_cost.bar(x - width/2, ff_means, width, yerr=ff_ci, label='First-Fit', capsize=5)
        ax_cost.bar(x + width/2, pal_means, width, yerr=pal_ci, label='Next-Fit (PAL)', capsize=5)
        ax_cost.set_ylabel('Average Communication Cost')
        ax_cost.set_xlabel('Number of VM Pairs')
        ax_cost.set_title('Comparison of Placement Algorithms: Cost')
        ax_cost.set_xticks(x)
        ax_cost.set_xticklabels(x_counts)
        ax_cost.legend()
        ax_cost.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        fig_cost.savefig('placement_comparison_pairs_cost.png', dpi=300)

        # --- Active PMs figure ---
        fig_pm, ax_pm = plt.subplots(figsize=(10, 6))
        ax_pm.bar(x - width/2, ff_active_means, width, yerr=ff_active_ci, label='First-Fit', capsize=5)
        ax_pm.bar(x + width/2, pal_active_means, width, yerr=pal_active_ci, label='Next-Fit (PAL)', capsize=5)
        ax_pm.set_ylabel('Average Active PMs (≥1 VM)')
        ax_pm.set_xlabel('Number of VM Pairs')
        ax_pm.set_title('Active PMs vs. VM Pair Count')
        ax_pm.set_xticks(x)
        ax_pm.set_xticklabels(x_counts)
        ax_pm.legend()
        ax_pm.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        fig_pm.savefig('placement_comparison_pairs_active_pms.png', dpi=300)

    @staticmethod
    def placement_compare_plot_capacity():
        # Generate plots vs PM capacity, 500 VM pairs
        ff_results = np.empty((4,10))
        pal_results = np.empty((4,10))

        ff_results_active_pms = np.empty((4,10))
        pal_results_active_pms = np.empty((4,10))

        capacities = [8, 10, 15, 20]
        for i in range(4):
            for j in range(10):
                tree = FatTree(k=16, vm_pair_count=500, vnf_capacity=3, vnf_count=3, pm_capacity=capacities[i])
                tree.set_traffic_range(0, 1000)
                ff_results[i][j], ff_results_active_pms[i][j] = tree.create_sized_pairs_ff_place(lower_bound=1, upper_bound=8)
                pal_results[i][j], pal_results_active_pms[i][j] = tree.create_pairs_sized_pal_place(lower_bound=1, upper_bound=8)

        # --- Cost plot stats ---
        ff_means = ff_results.mean(axis=1)
        pal_means = pal_results.mean(axis=1)
        ff_std = ff_results.std(axis=1, ddof=1)
        pal_std = pal_results.std(axis=1, ddof=1)

        # --- Active PMs plot stats ---
        ff_active_means = ff_results_active_pms.mean(axis=1)
        pal_active_means = pal_results_active_pms.mean(axis=1)
        ff_active_std = ff_results_active_pms.std(axis=1, ddof=1)
        pal_active_std = pal_results_active_pms.std(axis=1, ddof=1)

        # 95% CI via t (df=9)
        t_value = stats.t.ppf(0.975, df=9)
        ff_ci = t_value * ff_std / np.sqrt(10)
        pal_ci = t_value * pal_std / np.sqrt(10)
        ff_active_ci = t_value * ff_active_std / np.sqrt(10)
        pal_active_ci = t_value * pal_active_std / np.sqrt(10)
        
        
        # === DAT export for GNUplot ===
        with open('CostOverCapacity.dat', 'w') as f:
            f.write("# pm_capacity FF_mean FF_CI PAL_mean PAL_CI\n")
            for cap, ffm, ffc, palm, palc in zip(capacities, ff_means, ff_ci, pal_means, pal_ci):
                f.write(f"{cap} {ffm:.6f} {ffc:.6f} {palm:.6f} {palc:.6f}\n")

        with open('ActiveOverCapacity.dat', 'w') as f:
            f.write("# pm_capacity FF_mean FF_CI PAL_mean PAL_CI\n")
            for cap, ffm, ffc, palm, palc in zip(capacities, ff_active_means, ff_active_ci, pal_active_means, pal_active_ci):
                f.write(f"{cap} {ffm:.6f} {ffc:.6f} {palm:.6f} {palc:.6f}\n")


        # --- Cost figure ---
        x = np.arange(len(capacities))
        width = 0.35

        fig_cost, ax_cost = plt.subplots(figsize=(10, 6))
        ax_cost.bar(x - width/2, ff_means, width, yerr=ff_ci, label='First-Fit', capsize=5)
        ax_cost.bar(x + width/2, pal_means, width, yerr=pal_ci, label='Next-Fit (PAL)', capsize=5)
        ax_cost.set_ylabel('Average Communication Cost')
        ax_cost.set_xlabel('PM Capacity')
        ax_cost.set_title('Comparison of Placement Algorithms (500 VM pairs): Cost')
        ax_cost.set_xticks(x)
        ax_cost.set_xticklabels(capacities)
        ax_cost.legend()
        ax_cost.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        fig_cost.savefig('placement_comparison_capacity_cost.png', dpi=300)

        # --- Active PMs figure ---
        fig_pm, ax_pm = plt.subplots(figsize=(10, 6))
        ax_pm.bar(x - width/2, ff_active_means, width, yerr=ff_active_ci, label='First-Fit', capsize=5)
        ax_pm.bar(x + width/2, pal_active_means, width, yerr=pal_active_ci, label='Next-Fit (PAL)', capsize=5)
        ax_pm.set_ylabel('Average Active PMs (≥1 VM)')
        ax_pm.set_xlabel('PM Capacity')
        ax_pm.set_title('Active PMs vs. PM Capacity (500 VM pairs)')
        ax_pm.set_xticks(x)
        ax_pm.set_xticklabels(capacities)
        ax_pm.legend()
        ax_pm.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        fig_pm.savefig('placement_comparison_capacity_active_pms.png', dpi=300)

        
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
