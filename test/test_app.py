import unittest
from src.fat_tree import FatTree
from src.core_switch import CoreSwitch
from src.agg_switch import AggregateSwitch
from src.edge_switch import EdgeSwitch
from src.phys_machine import PhysicalMachine
from src.vm_pair import VmPair

class TestApp(unittest.TestCase):

    def contains_element(self, arr, target):
        return target in arr

    def test_check_node_location(self):
        test_tree = FatTree(4, 10, 3, 3, 10)
        
        # first node is of type CoreSwitch
        self.assertTrue(isinstance(test_tree.tree[0], CoreSwitch))
        # fourth node is of type CoreSwitch
        self.assertTrue(isinstance(test_tree.tree[3], CoreSwitch))
        # fifth node is of type AggregateSwitch
        self.assertTrue(isinstance(test_tree.tree[4], AggregateSwitch))
        # 12th is also AggregateSwitch
        self.assertTrue(isinstance(test_tree.tree[11], AggregateSwitch))
        # 13th is EdgeSwitch
        self.assertTrue(isinstance(test_tree.tree[12], EdgeSwitch))
        # 20th node is EdgeSwitch
        self.assertTrue(isinstance(test_tree.tree[19], EdgeSwitch))
        # 21st node is PhysicalMachine
        self.assertTrue(isinstance(test_tree.tree[20], PhysicalMachine))
        self.assertEqual(test_tree.first_pm, 20)
        # 36th node is PhysicalMachine
        self.assertTrue(isinstance(test_tree.tree[35], PhysicalMachine))
        self.assertEqual(test_tree.last_pm, 35)
        # There are 16 PMs
        self.assertEqual(test_tree.pm_count, 16)

    def test_check_core_edges(self):
        test_tree = FatTree(4, 10, 3, 3, 10)
        
        first_core = test_tree.tree[0]
        last_core = test_tree.tree[3]
        
        self.assertTrue(self.contains_element(first_core.aggr_edges, 4))
        self.assertFalse(self.contains_element(first_core.aggr_edges, 5))
        self.assertTrue(self.contains_element(last_core.aggr_edges, 5))
        self.assertFalse(self.contains_element(last_core.aggr_edges, 4))

    def test_check_aggr_edges(self):
        test_tree = FatTree(4, 10, 3, 3, 10)
        
        first_aggr = test_tree.tree[4]
        last_aggr = test_tree.tree[11]
        
        self.assertTrue(self.contains_element(first_aggr.core_edges, 1))
        self.assertFalse(self.contains_element(first_aggr.core_edges, 5))
        self.assertFalse(self.contains_element(first_aggr.core_edges, 2))
        self.assertTrue(self.contains_element(last_aggr.core_edges, 2))
        self.assertFalse(self.contains_element(last_aggr.core_edges, 1))

    def test_vm_pair_size(self):
        test_tree = FatTree(4, 10, 3, 3, 10)
        self.assertTrue(len(test_tree.vm_pairs)==10)

    def test_check_dists(self):
        test_tree = FatTree(4, 10, 3, 3, 10)
        
        # Check if types are correct
        self.assertTrue(isinstance(test_tree.tree[0], CoreSwitch))
        self.assertTrue(isinstance(test_tree.tree[20], PhysicalMachine))
        
        # Check distances
        self.assertEqual(FatTree.distance(test_tree.tree[0], test_tree.tree[20]), 3)
        self.assertEqual(FatTree.distance(test_tree.tree[0], test_tree.tree[1]), 2)
        self.assertEqual(FatTree.distance(test_tree.tree[0], test_tree.tree[2]), 4)
        
        self.assertEqual(FatTree.distance(test_tree.tree[20], test_tree.tree[21]), 2)
        self.assertEqual(FatTree.distance(test_tree.tree[20], test_tree.tree[22]), 4)
        self.assertEqual(FatTree.distance(test_tree.tree[20], test_tree.tree[24]), 6)
        self.assertEqual(FatTree.distance(test_tree.tree[20], test_tree.tree[6]), 4)
        
        # Check distances for switches
        self.assertEqual(FatTree.distance(test_tree.tree[6], test_tree.tree[7]), 2)
        self.assertEqual(FatTree.distance(test_tree.tree[6], test_tree.tree[8]), 2)
        self.assertEqual(FatTree.distance(test_tree.tree[6], test_tree.tree[9]), 4)

if __name__ == '__main__':
    unittest.main()
