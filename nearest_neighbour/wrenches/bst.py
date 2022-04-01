# simple implementation of binary search tree used in the algorithms

class Node:
    def __init__(self, key, value=-1):
        self.key = key
        self.value = value
        self.left = None
        self.right = None

    def __str__(self):
        return f'Node({self.key}, {self.value})'


class BST:
    def __init__(self, items=None):
        self.root = None
        if items:
            for item in items:
                self.insert(item[0], item[1])

    def insert(self, key, value=-1):
        node = Node(key, value)
        if not self.root:
            self.root = node
        else:
            parent = self.root
            while True:
                if key < parent.key:
                    if parent.left:
                        parent = parent.left
                    else:
                        parent.left = node
                        break
                elif key > parent.key:
                    if parent.right:
                        parent = parent.right
                    else:
                        parent.right = node
                        break
                else:
                    break

    def inorder_traversal(self):
        if not self.root:
            print("Empty Tree.")
        else:
            stack = []
            node = self.root
            while stack or node:
                while node:
                    stack.append(node)
                    node = node.left
                node = stack.pop()
                print(node)
                node = node.right


if __name__ == '__main__':

    test = [(5.3, 1), (7, 2), (3, 3), (4, 4), (2.9, 5), (8, 6), (6, 7), (1, 8)]
    bst = BST()
    for item in test:
        bst.insert(item[0], item[1])
    bst.inorder_traversal()
