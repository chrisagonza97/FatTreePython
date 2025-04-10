from src.fat_tree import FatTree


class App:
    def __init__(self):
        pass

    @staticmethod
    def main():
        # Creating an instance of FatTree
        tree = FatTree(8, 10, 3, 3, 40)
        tree.set_traffic_range(0, 1000)
        tree.create_vm_pairs()
        #tree.cs2_migration()
        tree.ac_migration()
        #state = tree.get_state()
        #print(state)

if __name__ == "__main__":
    App.main()
