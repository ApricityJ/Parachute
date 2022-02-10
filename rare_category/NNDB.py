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

    # Record the examples have been selected
    selected_set = set()
    # Conduct the for loop to increase hyperball radius
    for t in range(1, n + 1):
        # Define S_i = max (N_i - N_k) with x_k i NN(x_i, t*r_prime)
        S_i = []
        t_r_prime = t*r_prime
        for i, example in enumerate(S):
            # For every example in S - that hasn't been selected
            if example not in selected_set:
                s_i = calculate_s_i(S, tree, example, N_i[i], t_r_prime)
                S_i.append(s_i)
            else:
                # Else set s_i = float("-inf")
                S_i.append(float("-inf"))
        # Then Query x = argmax S_i, with x_i in S
        # Here only consider the first max one
        x_to_query = S.index(max(S_i))
        # x_to_query_index_list = [i for i,x in enumerate(S_i) if x==max(S_i)]

        label = query_by_oracle(x_to_query)
        # If label of x is 2, break
        if label == 2:
            break
        else:
            # Add to the selected_set and begin next loop
            selected_set.add(tuple(x_to_query))


# Function : Help calculate each s_i = max (n_i - n_k) with x_k i NN(x_i, t*r_prime)
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


# just example
def query_by_oracle(example):
    return 1

