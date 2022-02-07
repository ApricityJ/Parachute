# Implementation of Nearest-Neighbor-Based Rare Category Detection

# Retrieve kd-trees from scikit learn
from sklearn.neighbors import KDTree


# require samples (S) and probability of rare class (p)
def nndb(S, p):
    # Define n as the number of samples
    n = len(S)
    # Define r_prime
    r_prime = float("inf")
    # Define K = np, where n is the number of samples, and p is the probability of the rare class
    K = n * p
    # Create KDTree from data
    tree = KDTree(S)
    # For each example, calculate the distance to it's Kth neighbor
    for example in S:
        distances, indices = tree.query(example, k=K)
        # Set r_prime to the minimum distance
        for dist in distances:
            if dist != 0 and dist < r_prime:
                r_prime = dist
    # Create a cardinality list N_i
    N_i = []
    # For each example, create a hyperball with radius r_prime
    for example in S:
        indices = tree.query_radius(example, r=r_prime)
        # Get the number of closest other examples within r_prime radius
        N_i.append(len(indices))
    # Conduct the for loop to increase hyperball radius
    for t in xrange(1, n + 1):
        # TODO:
        # For every example in S - that hasn't been selected
        # Define S_i = max (N_i - N_j) with x_j i NN(x_i, t*r_prime)
        # Else set S_i = float("-inf")
        # Then Query x = argmax S_i, with x_i in S
        # If label of x is 2, break
        pass
