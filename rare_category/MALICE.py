# Implementation of Modified Active Learning for Initial Class Exploration (MALICE)

# Retrieve kd-trees from scikit learn
from sklearn.neighbors import KDTree
import numpy as np


# require samples (S) and probabilities of rare classes (P)
def alice(S, P):
    # Define n as the number of samples
    n = len(S)
    # Define m as the number of rare classes, a little difference from the paper's algorithm
    m = len(P)
    # Initialize all the rare classes as undiscovered
    discovered_list = [0 for item in P]
    # Create KDTree from data
    tree = KDTree(S)

    # Step 2 to 5
    # Create a list of all r_prime_c
    R_prime_c = []
    for p_c in P:
        # Define K_c = n*p_c, where n is the number of samples, and p_c is the probability of the current rare class
        K_c = n * p_c
        # Define r_prime_c
        r_prime_c = float("inf")
        # For each example, calculate the distance to it's K_cth neighbor
        for example in S:
            distances, indices = tree.query(example, k=K_c)
            # Set r_prime_c to the minimum distance
            for dist in distances:
                if dist != 0 and dist < r_prime_c:
                    r_prime_c = dist
        R_prime_c.append(r_prime_c)

    # Step 6
    # let r_prime_1 = max(r_prime_c)
    r_prime_1 = max(R_prime_c)
    # Put r_prime_1 the head of R_prime_c
    R_prime_c.index(0, r_prime_1)

    # Step 7 to 9
    # Create a list of each rare class's N_i
    N_i_C = []
    # For each rare class, calculate N_i
    for i in range(m):
        # Create a cardinality list N_i
        N_i = []
        # For each example, create a hyperball with radius r_prime_c
        for example in S:
            indices = tree.query_radius(example, r=R_prime_c[i])
            # Get the number of closest other examples within r_prime_c radius
            N_i.append(len(indices))
        N_i_C.append(N_i)

    # Step 10 to 17
    # Record the index of sample that has been selected
    selected_set = set()
    # Record the class of each sample, 0 means not labeled, 0 for all at the beginning
    samples_y = [0 for item in S]
    # In the loop, i refer to the index of each rare class (as c = 2:m in the paper)
    for i in range(m):
        # If class c has not been discovered
        if discovered_list[i] == 0:
            # Conduct the for loop to increase hyperball radius
            for t in range(2, n + 1):
                # Define s_i_c = max (N_i_c - N_k_c) with x_k in NN(x_i, t*r_prime_c)
                S_i_c = []
                # Here use (t - 1) according to step 15 in Algorithm MALICE
                t_r_prime_c = t * R_prime_c[i] if t == 2 else (t-1) * R_prime_c[i]
                # In the loop, j refer to the index of each sample
                for j, example in enumerate(S):
                    # For every example in S - that hasn't been selected
                    if j not in selected_set:
                        continue_calculate_flag = True
                        # For every index of labeled sample
                        for x_i in selected_set:
                            # Find the r_prime of the class of the labeled sample
                            # Minus one because class begins from 1 while R_prime_c begins from 0
                            r_prime_y_of_x_i = R_prime_c[samples_y[x_i] - 1]
                            # Calculate the Euclidean distance of the current example and every labeled sample
                            # Paper step 13: ||x_i - x_k|| <= r_prime_yi
                            if np.linalg.norm(np.array(example) - np.array(S[x_i])) <= r_prime_y_of_x_i:
                                S_i_c.append(float("-inf"))
                                # Flag the following calculation skipped
                                continue_calculate_flag = False
                                # Find anyone can break
                                break
                        if continue_calculate_flag:
                            # The original calculation for example not selected
                            s_i_c = calculate_s_i(S, tree, example, N_i_C[i][j], t_r_prime_c)
                            S_i_c.append(s_i_c)
                    else:
                        # Else set s_i_c = float("-inf")
                        S_i_c.append(float("-inf"))
                # Then Query x = argmax S_i, with x_i in S
                # Here only consider the first max one
                query_index = S_i_c.index(max(S_i_c))

                label = query_by_oracle(S[query_index])
                # If label of x is c, which is i + 2 here, then break
                if label == i + 2:
                    # Add to the selected_set
                    selected_set.add(query_index)
                    # Record the label of current sample
                    samples_y[query_index] = label
                    break
                else:
                    # Mark the class that x belongs to as discovered
                    discovered_list[label - 2] = 1
                    # Add to the selected_set and begin next loop
                    selected_set.add(query_index)
                    # Record the label of current sample
                    samples_y[query_index] = label


# the same as calculate_s_i(S, tree, example, n_i, r) in NNDB
# function : help calculate each s_i = max (n_i - n_k) with x_k in NN(x_i, t*r_prime)
# parameters : samples(S), created KDTree(tree), current example(example), n_i of current sample(n_i), loop t*r_prime(r)
def calculate_s_i(S, tree, example, n_i, r):
    # Find neighbors of example within r radius
    indices = tree.query_radius(example, r=r)
    # Record all neighbors' number of neighbors within r radius
    N_k = []
    # Loop for each neighbor of the current example
    for k in indices:
        indices_n_k = tree.query_radius(S[k], r=r)
        N_k.append(len(indices_n_k))
    # For each fixed n_i, find max (n_i - n_k) equivalent to find min n_k
    min_n_k = min(N_k)
    return n_i - min_n_k


# just example, return class label
def query_by_oracle(example):
    return 1
