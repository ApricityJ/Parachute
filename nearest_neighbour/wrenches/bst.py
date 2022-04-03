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

    def knn_search(self, k, target):
        if k < 1 or target < 0:
            print("Parameter error.")
        k_neighbors = []
        self._knn_search(k, target, self.root, k_neighbors)
        return k_neighbors

    def _knn_search(self, k, target, node, k_neighbors):
        if not node:
            return
        else:
            distance = abs(target - node.key)
            if (len(k_neighbors) < k) or (distance < k_neighbors[-1][0]):
                self._knn_search(k, target, node.left, k_neighbors)
                self._knn_search(k, target, node.right, k_neighbors)

                k_neighbors_length = len(k_neighbors)
                if k_neighbors_length == 0:
                    k_neighbors.append((distance, node))
                else:
                    for i in range(k_neighbors_length):
                        if distance >= k_neighbors[i][0]:
                            i += 1
                            if i == k_neighbors_length and k_neighbors_length < k:
                                k_neighbors.append((distance, node))
                        else:
                            if k_neighbors_length < k:
                                k_neighbors.insert(i, (distance, node))
                                break
                            else:
                                k_neighbors[i+1:] = k_neighbors[i:-1]
                                k_neighbors[i] = (distance, node)
                                break
        return


if __name__ == '__main__':

    test = [(5.3, 1), (7, 2), (3.7, 3), (4, 4), (2.9, 5), (8, 6), (6.9, 7), (1, 8)]
    bst = BST()
    for item in test:
        bst.insert(item[0], item[1])
    bst.inorder_traversal()

    print('----------')

    res = bst.knn_search(8, 3.7)
    for item in res:
        print(f'{item[0]}, {item[1]}')
        print(f'{item[1].key}, {item[1].value}')
