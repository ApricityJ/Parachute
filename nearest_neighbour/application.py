# Sample of application  (Algo6 Implicit maximum likelihood estimation IMLE)

import numpy as np
from knn_retrieval_procedure import knn_retrieval_procedure


# just pseudo-code
# parameters:
#     D: dataset, theta: parameters of the target distribution P, K and L: the loop numbers
#     T_jl: binary search trees from construction algo, U_jl: associated projection vectors,
#     epsilon: the maximum tolerable failure probability, batch_size
# return:
#     theta: parameters of the target distribution P
def IMLE(D, theta, K, L, T_jl, U_jl, epsilon, batch_size):

    # Initialize theta to a random vector

    for k in range(K):
        # Draw i.i.d samples x_theta_1 to x_theta_m from P_theta (define as S_theta)
        S_theta = []
        # Pick a random batch S from dataset D
        S = np.random.choice(D, batch_size)
        # Find the nearest node of sampled data
        # Use knn_retrieval_procedure or prioritized_knn_retrieval_procedure
        Sigma_i = []
        for item in S:
            nearest_node = knn_retrieval_procedure(item, T_jl, U_jl, epsilon, S_theta, 1)
            Sigma_i.append(nearest_node)

        for l in range(L):
            # Pick a random mini-batch S_tilde from S and corresponding Sigma_i_tidle from Sigma_i
            S_tilde = []
            Sigma_i_tidle = []
            loss = 0
            for item1, item2 in zip(S_tilde, Sigma_i_tidle):
                loss += np.linalg.norm(item1 - Sigma_i_tidle)**2
            loss = loss * len(D) / len(S_tilde)
            theta = theta - opt_algo(loss)

    return theta


def opt_algo(object):
    pass
