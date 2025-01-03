from src.fat_tree import FatTree


class App:
    def __init__(self):
        pass

    @staticmethod
    def main():
        # Creating an instance of FatTree
        tree = FatTree(4, 10, 3, 3, 10)
        tree.set_traffic_range(100, 4000)
        tree.create_vm_pairs()
        tree.cs2_migration()
        #state = tree.get_state()
        #print(state)

if __name__ == "__main__":
    App.main()
