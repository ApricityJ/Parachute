# Implementation of SEmiparametric Density Estimation based Rare category detection (SEDER)

import math
# LeaveOneOut cross validation
from sklearn.model_selection import LeaveOneOut
import numpy as np


# require samples (S) and the number of rare classes (m)
# return the set I of selected examples and the set L of their labels (I, L)
def seder(S: list, m: int):
    # Step 1 : Initialize I and L
    I = []
    L = []

    # Step 2 to 5
    # Define n as the number of samples
    n = len(S)
    # Define d as the dimensionality of the feature space
    d = len(S[0])
    # Record all sigma_j
    sigma = []
    # Record all b_hat_j
    b_hat = []
    # Create the helper matrix
    S_matrix = []
    for sample in S:
        S_matrix.append(sample)
    S_matrix = np.asarray(S_matrix)
    # LeaveOneOut cross validation
    loo = LeaveOneOut()
    # Loop for every feature column
    for j in range(d):
        # The current feature column
        fea_j = S_matrix[:, j]
        # Record all LeaveOneOut std
        res = []
        for others, one_out in loo.split(fea_j):
            # Calculate std
            res.append(np.std(fea_j[others]))
        # The final bandwidth sigma_j using cross validation
        sigma_j = sum(res) / n
        sigma.append(sigma_j)

        # Calculate b_hat_j
        A = 0
        C = 0
        # The outermost sum of A
        for example_k in S_matrix:
            # Initialize the numerator and denominator of A
            numerator = 0
            denominator = 0
            # Calculate the numerator of A
            for example_i in S_matrix:
                numerator += math.exp(- (example_k[j] - example_i[j]) ** 2 / (2 * (sigma_j ** 2))) * (example_i[j]) ** 2
            # Calculate the denominator of A
            for example_i in S_matrix:
                denominator += math.exp(- (example_k[j] - example_i[j]) ** 2 / (2 * (sigma_j ** 2)))
            A += numerator / denominator
            C += example_k[j] ** 2

        # Final A
        A = A / n
        # Calculate B
        B = sigma_j ** 2
        # Final C
        C = C / n
        # Calculate b_hat_j according to equation 3.10
        b_hat_j = (-B + math.sqrt(B ** 2 + 4 * A * C)) / (2 * A)
        b_hat.append(b_hat_j)

    # Step 6 to 8
    # Record all s_k
    score = []
    # Loop for every example to calculate its score
    for example_k in S_matrix:
        # Initialize s_k
        s_k = 0
        # The outermost sum
        for l in range(d):
            numerator = 0
            # Calculate the numerator in the sqrt of equation 3.11
            for example_i in S_matrix:
                D_i_x_k = 1 / n
                # Calculate D_i(xk)
                for j_in_D_i in range(d):
                    # The equation in Theorem 6
                    D_i_x_k *= (1 / (math.sqrt(2 * math.pi * b_hat[j_in_D_i]) * sigma[j_in_D_i])) \
                               * math.exp(-(example_k[j_in_D_i] - b_hat[j_in_D_i] * example_i[j_in_D_i]) ** 2 / (
                                2 * (sigma[j_in_D_i] ** 2) * b_hat[j_in_D_i]))
                # Calculate numerator after D_i_x_k
                numerator += D_i_x_k * (example_k[l] - b_hat[l] * example_i[l]) ** 2
            # Calculate the denominator in the sqrt of equation 3.11
            denominator = (sigma[l] ** 2 * b_hat[l]) ** 2
            s_k += numerator / denominator
        # The final s_k
        s_k = math.sqrt(s_k)
        score.append(s_k)

    # Step 9 to 13
    # Record the number of rare classes have been found
    m_found = 0
    # While not find all rare classes
    while m_found < m:
        # Record all samples meet the criteria of step 10
        S_prime = []
        # Record the scores of the selected samples
        score_prime = []
        # Loop for all samples to find those meet the criteria
        for index, example in enumerate(S):
            # Exclude flag
            exclude_flag = False
            # Loop for labeled sample
            for example_i in I:
                # Loop for every feature dimension
                for j in range(d):
                    if (abs(example[j] - example_i[j])) <= 3*sigma[j]:
                        exclude_flag = True
                        break
                if exclude_flag:
                    break
            # If meet the criteria, then append the sample and its score
            if not exclude_flag:
                S_prime.append(example)
                score_prime.append(score[index])

        # Then Query x = argmax score_i, with x_i in S_prime
        # Here only consider the first max one
        query_index = score_prime.index(max(score_prime))
        label = query_by_oracle(S_prime[query_index])

        # If label of rare classes, then the number of rare classes have been found + 1
        if label != 1:
            m_found += 1
        # Append to selected examples I and L of their labels
        I.append(S_prime[query_index])
        L.append(label)

    return I, L


# just example, return class label
def query_by_oracle(example):
    return 1
