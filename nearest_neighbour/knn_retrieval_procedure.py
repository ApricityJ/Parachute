# Implementation of k-nearest neighbour retrieval procedure (Algo2)

import math
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
                # To exclude q_jl_bar[l][j] itself, slice [1:]
                result = T_jl[l][j].knn_search(i + 2, q_jl_bar[l][j])[1:]
                for res_item in result:
                    # Get the node index
                    h_jl_i = res_item[1][1]
                    # Minus 1 because the range of each C_l[l] is 0 to n-1 while the range of h_jl_i is 1 to n
                    C_l[l][h_jl_i - 1] = C_l[l][h_jl_i - 1] + 1

            # The loop here is a little different from the description in the paper
            for n_index in range(n):
                if C_l[l][n_index] == m:
                    # Turn the n_index back to 1 to n
                    S_l[l].add(n_index + 1)

        # If satisfy the stopping condition, then break
        if is_stopping_condition_satisfied_data_dependent(i, S_l, epsilon, D, q, k, m):
            break

    # Define the union of all s_l
    indexes = set()
    for item in S_l:
        indexes = indexes.union(item)
    # Find the final k closest nodes
    nodes = knn_helper(D, indexes, q, k, target='All')

    return nodes


# Function to find the node which is the kth nearest or all k nearest neighbors to the query q in the candidate indexes
# parameters:
#     D: dataset, indexes: all the candidate indexes, q: the query point, k: k-nearest, target: one or all
# return:
#     nodes: the kth nearest (target='One') or all k nearest neighbors (target='All')
def knn_helper(D, indexes, q, k, target='All'):
    indexes = list(indexes)
    # Calculate the euclidean distances of all the nodes and the query point
    euclidean_distances = [np.linalg.norm(q - D[index - 1]) for index in indexes]
    # Find the k closest nodes
    knn_indexes = [i for _, i in sorted(zip(euclidean_distances, indexes))][:k]
    nodes = [D[index] for index in knn_indexes]
    # Return all k nodes
    if target == 'All':
        return nodes
    # Return the kth node
    elif target == 'One':
        return nodes[k - 1]
    else:
        return None


# parameters:
#     i: current number of the outermost iteration, S_l: sets of indexes, epsilon: maximum tolerable failure probability
#     D: dataset, q: the query point, k: k-nearest, m: the number of simple indices
# return: Ture or False
def is_stopping_condition_satisfied_data_dependent(i, S_l, epsilon, D, q, k, m):
    # Define the union of all s_l
    indexes = set()
    for item in S_l:
        indexes = indexes.union(item)

    # When the number of all candidates >= k
    if len(indexes) >= k:
        # Find the ith closest candidate point to q, i+1 for i begins from 0
        p_k_tilde = knn_helper(D, indexes, q, i+1, target='One')
        p_l_max_tilde_record = []
        # Find the farthest candidate point from each lth composite index
        for item in S_l:
            farthest_node = knn_helper(D, item, q, len(item), target='One')
            p_l_max_tilde_record.append(farthest_node)
        # Define and calculate the probability
        probability = 1
        for p_l_max_tilde in p_l_max_tilde_record:
            probability *= 1 - (2 / math.pi * math.acos((np.linalg.norm(p_k_tilde - q)) / (np.linalg.norm(p_l_max_tilde - q))))**m
        if probability <= epsilon:
            return True
        else:
            return False
    else:
        return False


# This way is not recommended.
# parameters:
#     i: current number of the outermost iteration, k: k-nearest, n: the number of samples,
#     gamma: the global relative sparsity of the dataset (which is rarely known a priori)
# return: Ture or False
def is_stopping_condition_satisfied_data_independent(i, k, n, gamma):
    return i > max(k * math.log2(n / k)), k * (n / k) ** (1 - math.log2(gamma))
