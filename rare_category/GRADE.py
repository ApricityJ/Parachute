# Implementation of Graph-based Rare Category Detection (GRADE)

import math
import heapq
# Retrieve kd-trees from scikit learn
from sklearn.neighbors import KDTree
import numpy as np


# require samples (S), probabilities of all classes (P) and parameter alpha (alpha)
# NOTICE: in this algorithm, P is different from the previous ones
# return the set I of selected examples and the set L of their labels (I, L)
def grade(S, P, alpha):
    # Define n as the number of samples
    n = len(S)
    # Define m as the number of rare classes, a little difference from the paper's algorithm
    m = len(P) - 1
    # Create KDTree from data
    tree = KDTree(S)

    # Step 1
    # Exclude p1
    K = math.ceil(n * max(P[1:]))

    # Step 2
    # Calculate sigma
    sigma = float("inf")
    # For each example, calculate the distance to it's K_th neighbor
    for example in S:
        distances, indices = tree.query(example, k=K)
        # Set sigma to the minimum distance
        for dist in distances:
            if dist != 0 and dist < sigma:
                sigma = dist

    # Step 3
    # Initialize matrix W_prime
    W_prime = np.zeros([n, n], dtype=float)
    # Create the helper matrix
    S_matrix = []
    for sample in S:
        S_matrix.append(sample)
    S_matrix = np.asarray(S_matrix)
    # Construct the pair-wise similarity matrix W_prime according to equation 3.12
    for i in range(n):
        for k in range(n):
            if i != k:
                W_prime[i][k] = math.exp(- np.linalg.norm(S_matrix[i] - S_matrix[k]) / (2*sigma**2))

    # Step 4
    # Initialize the diagonal matrix D
    D = np.zeros([n, n], dtype=float)
    # Construct D
    for i in range(n):
        D_ii = 0
        for k in range(n):
            D_ii += W_prime[i][k]
        D[i][i] = D_ii

    # Step 5
    # Calculate D^-1/2, maybe too redundant here
    V, Q = np.linalg.eig(D)
    V = np.diag(V ** (-0.5))
    D_prime = np.dot(np.dot(Q, V), np.linalg.inv(Q))
    # Calculate the normalized matrix W
    W = np.dot(np.dot(D_prime, W_prime), D_prime)

    # Step 6
    # Calculate the global similarity matrix A
    i_matrix = np.identity(n)
    A = np.linalg.inv(i_matrix - alpha * W)

    # Step 7 to 11
    # Record all a_c
    A_c = []
    # Record all n_i_c
    N_i_C = []
    for c in range(m):
        a_c = float("-inf")
        K_c = n * P[c+1]
        for row in A:
            # Find the K_c largest element
            K_c_largest = heapq.nlargest(K_c, row)
            for element in K_c_largest:
                if element > a_c:
                    a_c = element
        A_c.append(a_c)

        # For each rare class, calculate N_i
        N_i = []
        # Loop for every sample
        for i in range(n):
            # Initialize n_i_c
            n_i_c = 0
            for j in range(n):
                # According to step 10, A(x, x_i)>= a_c
                if A[j][i] >= a_c:
                    n_i_c += 1
            N_i.append(n_i_c)
        N_i_C.append(N_i)

    # Step 12 to 19
    # Initialize all the classes as undiscovered
    discovered_list = [0 for item in P]
    # Record the examples have been selected
    I = []
    # Record the example's index
    I_index = []
    # Record their labels
    L = []
    # In the loop, i refer to the index of each rare class (as c = 2:m in the paper)
    for i in range(m):
        # If class c has not been discovered, +1 for rare class
        if discovered_list[i+1] == 0:
            # Conduct the for loop to increase t
            for t in range(2, n + 1):
                # Record every s_i_c
                S_i_c = []
                # The loop t_a_c
                t_a_c = A_c[i] / t
                # In the loop, k refer to the index of each sample
                for k, example in enumerate(S):
                    continue_calculate_flag = True
                    # The example has been selected and labeled
                    if k in I_index:
                        S_i_c.append(float("-inf"))
                        continue
                    # According to step 15, if A(x_i, x_k)>= a^yi, exclude the example
                    for j, example_j in enumerate(I):
                        # Since the paper not mention a_1, skip the example labeled 1
                        if L[j] == 1:
                            continue
                        # Use I_index[j] to get the original index of example_j in A
                        if A[I_index[j]][k] >= A_c[L[j]-2]:
                            S_i_c.append(float("-inf"))
                            # Flag the following calculation skipped
                            continue_calculate_flag = False
                            # Find anyone can break
                            break
                    if continue_calculate_flag:
                        # The original calculation for example not selected
                        s_i_c = calculate_s_i(S, A, N_i_C[i], k, t_a_c)
                        S_i_c.append(s_i_c)

                # Then Query x = argmax S_i, with x_i in S
                # Here only consider the first max one
                query_index = S_i_c.index(max(S_i_c))

                label = query_by_oracle(S[query_index])

                # Post-processing
                I.append(S[query_index])
                I_index.append(query_index)
                L.append(label)
                # If label of x is c, which is i + 2 here, then break
                if label == i + 2:
                    discovered_list[i + 1] = 1
                    break
                else:
                    # Mark the class that x belongs to as discovered
                    discovered_list[label - 1] = 1
    return I, L


# function : help calculate each s_i = max (n_i_c - n_k_c) with x_k in NN(x_i, a_c/t)
# parameters : samples(S), global similarity matrix(A), N_i_C of current rare class(N_i_c),
#              current example index(example_index), loop a_c/t(t_a_c)
def calculate_s_i(S, A, N_i_c, example_index, t_a_c):
    # Record all similar examples' n_i_c
    N_k_c = []
    for k, example_k in enumerate(S):
        if A[k][example_index] >= t_a_c:
            # Get n_k_c
            N_k_c.append(N_i_c[k])
    # For each fixed n_i_c, find max (n_i_c - n_k_c) equivalent to find min n_k_c
    min_n_k_c = min(N_k_c)
    return N_i_c[example_index] - min_n_k_c


# just example, return class label
def query_by_oracle(example):
    return 1

