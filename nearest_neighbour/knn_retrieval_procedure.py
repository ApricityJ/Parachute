# Implementation of k-nearest neighbour retrieval procedure (Algo2)

import numpy as np
from wrenches.bst import BST


# parameters:
#     q: the query point, T_jl: binary search trees from construction algo, U_jl: associated projection vectors,
#     epsilon: the maximum tolerable failure probability, D: dataset, k: k-nearest
# return:
#     nodes: k points in the union of S_l closest in Euclidean distance to q
def knn_retrieval_procedure(q, T_jl, U_jl, epsilon, D, k):

    # Define the number of composite indices
    L = len(T_jl)
    # Define the number of simple indices
    m = len(T_jl[0])
    # Define n as the number of samples
    n = len(D)

    # Define C_l
    C_l = [[0 for j in range(n)] for i in range(L)]
    # Define q_jl_bar
    q = np.asarray(q)
    q_jl_bar = np.linalg.norm(q - U_jl, axis=2)
    # Define S_l
    S_l = [set() for i in range(L)]

    for i in range(n):
        for l in range(L):
            for j in range(m):
                # To find the node whose key is the ith closest to q_jl_bar[l][j]
                # i+1 because i begins from 0, i+1+1 because q_jl_bar[l][j] itself should be excluded
                result = T_jl[l][j].knn_search(i+2, q_jl_bar[l][j])
                # To exclude q_jl_bar[l][j] itself, res_index begins from 1
                for res_index in range(1, len(result)):
                    # Get the node index
                    h_jl_i = result[res_index][1][1]
                    # Minus 1 because the range of each C_l[l] is 0 to n-1 while the range of h_jl_i is 1 to n
                    C_l[l][h_jl_i - 1] = C_l[l][h_jl_i - 1] + 1
            # The loop here is a little different from the description in the paper
            for n_index in range(n):
                if C_l[l][n_index] == m:
                    # Turn the n_index back to 1 to n
                    S_l[l].add(n_index + 1)

        # If satisfy the stopping condition, then break
        if is_stopping_condition_satisfied(i, S_l, epsilon):
            break

    # Define the union of all s_l
    indexes = set()
    for item in S_l:
        indexes = indexes.union(item)
    indexes = list(indexes)
    # Calculate the euclidean distances of all the nodes in the union and the query point
    euclidean_distances = [np.linalg.norm(q - D[index - 1]) for index in indexes]
    # Find the k closest nodes
    knn_indexes = [i for _, i in sorted(zip(euclidean_distances, indexes))][:k]
    nodes = [D[index] for index in knn_indexes]

    return nodes


# To be completed
def is_stopping_condition_satisfied(i, S_l, epsilon):
    pass
