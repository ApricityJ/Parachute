# Implementation of Data structure construction procedure (Algo1)

import numpy as np
from wrenches.bst import BST


# parameters:
#     D: dataset, m: the number of simple indices, L: the number of composite indices
# return:
#     T_jl: binary search trees, U_jl: random unit vectors in R^d
def construct(D, m, L):

    # Define n as the number of samples
    n = len(D)
    # Define d as the dimension of each sample
    d = len(D[0])

    # Define and init u_jl, T_jl
    U_jl = []
    T_jl = []
    for i in range(L):
        U_l = []
        T_l = []
        for j in range(m):
            # Random unit vector
            v = np.random.rand(d)
            v_norm = np.linalg.norm(v)
            U_l.append(v / v_norm)
            # Empty binary search tree
            T_l.append(BST())
        U_jl.append(U_l)
        T_jl.append(T_l)

    # Construct the binary search trees
    for j in range(m):
        for l in range(L):
            for i in range(n):
                # Compute Euclidean distance
                p_jl_bar = np.linalg.norm(D[i] - U_jl[l][j])
                # Insert into corresponding bst with p_jl_bar being the key and i being the value (i from 1 to n)
                T_jl[l][j].insert(p_jl_bar, i + 1)

    return T_jl, U_jl
