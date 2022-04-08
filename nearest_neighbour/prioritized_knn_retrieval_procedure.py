# Implementation of k-nearest neighbour querying procedure (Algo4)

import math
import heapq
import numpy as np
from wrenches.bst import BST


# parameters:
#     q: the query point, T_jl: binary search trees from construction algo, U_jl: associated projection vectors,
#     k0: the number of points to retrieve, k1: the number of points to visit, D: dataset, k: k-nearest
#     (k0 and k1 can be estimated by the formulas in Theorem 10 on Page 44 )
# return:
#     nodes: k points in the union of S_l closest in Euclidean distance to q
def prioritized_knn_retrieval_procedure(q, T_jl, U_jl, k0, k1, D, k):
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
    # Compute inner product
    q_jl_bar = np.dot(U_jl, q)
    # Define S_l
    S_l = [set() for i in range(L)]
    # Define the priority queues
    P_l = [[] for i in range(L)]

    for l in range(L):
        for j in range(m):
            # Find the node whose key is the closest to q_jl_bar[l][j]
            result = T_jl[l][j].knn_search(1, q_jl_bar[l][j])
            # Insert the node to corresponding priority queue with its priority
            # The priority is -(abs(p_jl_bar_(1) - q_jl_bar))
            # Insert the current j to help the procedure below
            heapq.heappush(P_l[l], (-(abs(result[1][0] - q_jl_bar[l][j])), result[1], j))

    for i_prime in range(k1 - 1):
        for l in range(L):
            if len(S_l[l]) < k0:
                # Get and remove the node with the highest priority in P_l[l]
                current_node = heapq.heappop(P_l[l])
                current_j = current_node[2]
                # Find the node whose key is the next closest to q_jl_bar
                # i_prime + 2 because i_prime begins from 0
                # Slice [i_prime + 1] to get the exact i+1th closest
                result = T_jl[l][current_j].knn_search(i_prime + 2, q_jl_bar[l][current_j])[i_prime + 1]
                # Insert to the priority queue
                heapq.heappush(P_l[l], (-(abs(result[1][0] - q_jl_bar[l][current_j])), result[1], current_j))

                # Get the node which removed from the priority queue's index
                h_jl_i = current_node[1][1]
                # Minus 1 because the range of each C_l[l] is 0 to n-1 while the range of h_jl_i is 1 to n
                C_l[l][h_jl_i - 1] = C_l[l][h_jl_i - 1] + 1

                if C_l[l][h_jl_i - 1] == m:
                    # Add to the candidate set
                    S_l[l].add(h_jl_i)

    # Define the union of all s_l
    indexes = set()
    for item in S_l:
        indexes = indexes.union(item)
    indexes = list(indexes)
    # Calculate the euclidean distances of all the nodes and the query point
    euclidean_distances = [np.linalg.norm(q - D[index - 1]) for index in indexes]
    # Find the final k closest nodes
    knn_indexes = [i for _, i in sorted(zip(euclidean_distances, indexes))][:k]
    nodes = [D[index - 1] for index in knn_indexes]

    return nodes
