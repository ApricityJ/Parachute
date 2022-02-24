# Implementation of Graph-based Rare Category Detection with Less Information (GRADE-LI)

import math
# Retrieve kd-trees from scikit learn
from sklearn.neighbors import KDTree
import numpy as np


# require: samples (S), an upper bound p on the proportions of all the minority classes (p), parameter alpha (alpha)
#          and the number of all classes (m)
# return:  the set I of selected examples and the set L of their labels (I, L)
def grade_li(S, p, alpha, m):
    # Define n as the number of samples
    n = len(S)
    # Create KDTree from data
    tree = KDTree(S)

    # Step 1
    K = math.ceil(n * p)

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
    # Construct the pair-wise similarity matrix W_prime as algo5
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

    # Step 7
    # Initialize a
    a = float("-inf")
    for row in A:
        # Here find the K_th largest element can be converted to find the largest element
        largest = max(row)
        if largest > a:
            a = largest

    # Step 8
    # Record all n_i
    N_i = []
    # Loop for every sample
    for i in range(n):
        # Initialize n_i
        n_i = 0
        for j in range(n):
            # Find A(x, x_i)>= a
            if A[j][i] >= a:
                n_i += 1
        N_i.append(n_i)

    # Step 9 to 15
    # Record the examples have been selected
    I = []
    # Record the example's index
    I_index = []
    # Record their labels
    L = []
    # Define the number of rare classes that have been discovered
    m_found = 0
    # While not all rare classes have been discovered
    while m_found < m - 1:
        # Conduct the for loop to increase t
        for t in range(2, n + 1):
            # Record all s_i
            S_i = []
            # The loop t_a
            t_a = a / t
            # In the loop, k refer to the index of each sample
            for k, example in enumerate(S):
                continue_calculate_flag = True
                # The example has been selected and labeled
                if k in I_index:
                    S_i.append(float("-inf"))
                    continue
                # According to step 11, if A(x_i, x_k)>= a, exclude the example
                for j, example_j in enumerate(I):
                    # Use I_index[j] to get the original index of example_j in A
                    if A[I_index[j]][k] >= a:
                        S_i.append(float("-inf"))
                        # Flag the following calculation skipped
                        continue_calculate_flag = False
                        # Find anyone can break
                        break
                if continue_calculate_flag:
                    # The original calculation for example not selected
                    s_i = calculate_s_i(S, A, N_i, k, t_a)
                    S_i.append(s_i)

            # Then Query x = argmax S_i, with x_i in S
            # Here only consider the first max one
            query_index = S_i.index(max(S_i))

            label = query_by_oracle(S[query_index])

            # Post-processing
            I.append(S[query_index])
            I_index.append(query_index)
            L.append(label)
            # If the example belongs to rare classes, m_found+1
            if label != 1:
                m_found += 1
    return I, L


# function : help calculate each s_i = max (n_i - n_k) with x_k in NN(x_i, a/t)
# parameters : samples(S), global similarity matrix(A), N_i calculated above (N_i),
#              current example index(example_index), loop a/t(t_a)
def calculate_s_i(S, A, N_i, example_index, t_a):
    # Record all similar examples' n_i
    N_k = []
    for k, example_k in enumerate(S):
        if A[k][example_index] >= t_a:
            # Get n_k
            N_k.append(N_i[k])
    # For each fixed n_i, find max (n_i - n_k) equivalent to find min n_k
    min_n_k = min(N_k)
    return N_i[example_index] - min_n_k


# just example, return class label
def query_by_oracle(example):
    return 1

