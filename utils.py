import torch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import itertools

# -----------------------------------------------
def softmax(matrix,axis = None,T=1):
    exp_matrix = torch.exp(matrix/T - torch.max(matrix, dim=axis, keepdim=True).values)
    softmax_matrix = exp_matrix / torch.sum(exp_matrix, dim=axis, keepdim=True)
    return softmax_matrix


def normalize_axis(to_normalize, axis=None):
    axis_sum = torch.sum(to_normalize, dim=axis, keepdim=True)
    axis_sum = torch.clamp(axis_sum, min=1e-5)#Some small epsilon for preventing nan's
    # Normalize only if the sum is greater than 1
    if torch.any(axis_sum > 1):
        return to_normalize / axis_sum
    return to_normalize


def normalize_matrix_axis(matrix,axis = None):
    slices = [normalize_axis(to_normalize=matrix.select(axis, i), axis=None) for i in range(matrix.size(axis))]
    return torch.stack(slices, dim=axis)


def get_axis_sums(matrix,axis = None):
    return torch.sum(matrix, dim=axis)

# Helper function to check for NaN or Inf values
def check_for_nan_inf(tensor, name):
    print_flag = False
    if isinstance(tensor, torch.Tensor):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            print_flag = True
        if not torch.isfinite(tensor).all():
            print(f"Inf or -Inf detected in {name}")
            print_flag = True
    elif isinstance(tensor, float):
        if math.isnan(tensor):
            print(f"NaN detected in {name} (float value)")
            print_flag = True
    if print_flag:
        print(f"Problematic Nan value Name: {name}")
        print(f"Value: {tensor}")

def primal_projection(matrix, axis=None):
    # Check for NaNs or Infs in the input matrix
    if torch.isnan(matrix).any() or not torch.isfinite(matrix).all():
        print("Input matrix contains NaNs or Infs")

    softmax_matrix = softmax(matrix=matrix, axis=axis)
    # check_for_nan_inf(softmax_matrix, "softmax_matrix")

    normalized_matrix = normalize_matrix_axis(matrix=softmax_matrix, axis=axis).clone()
    # check_for_nan_inf(normalized_matrix, "normalized_matrix")

    ret_mat = normalized_matrix

    r = get_axis_sums(ret_mat, axis=1)
    # check_for_nan_inf(r, "r (row sums)")

    c = get_axis_sums(ret_mat, axis=0)
    # check_for_nan_inf(c, "c (column sums)")

    r_tild = torch.ones(len(ret_mat), device=matrix.device) - r
    # check_for_nan_inf(r_tild, "r_tild")

    c_tild = torch.ones(len(ret_mat[0]), device=matrix.device) - c
    # check_for_nan_inf(c_tild, "c_tild")

    c_tild_sum = torch.sum(c_tild)
    if c_tild_sum <= 1e-5:
        c_tild_sum = 1e-5  # Small epsilon to prevent division by zero
    # check_for_nan_inf(torch.tensor([c_tild_sum]), "c_tild_sum")

    outer_product = torch.outer(r_tild, c_tild)
    # check_for_nan_inf(outer_product, "outer_product")

    ret_mat = ret_mat + (1 / c_tild_sum) * outer_product
    # check_for_nan_inf(ret_mat, "ret_mat (final result)")

    return ret_mat

# ---------------------------------------------------

def primal_sol(out_put_mat, distances_mat):
    # Step 1: Apply primal_projection
    bi_stoch_mat = primal_projection(out_put_mat, axis=0)
    # check_for_nan_inf(bi_stoch_mat, "bi_stoch_mat (after primal_projection)")

    # Step 2: Flatten the matrices
    B = bi_stoch_mat.view(-1)
    D = (distances_mat**2).view(-1)

    # # Step 3: Check NaN in B and D before dot product
    # check_for_nan_inf(B, "B (flattened bi_stoch_mat)")
    # check_for_nan_inf(D, "D (flattened distances_mat squared)")

    # # If NaNs are detected in B or D, return NaN tensor immediately
    # if torch.isnan(B).any():
    #     print("NaN detected in B before dot product")
    #     return torch.tensor(float('nan'), device=out_put_mat.device)
    # if torch.isnan(D).any():
    #     print("NaN detected in D before dot product")
    #     return torch.tensor(float('nan'), device=out_put_mat.device)

    # Step 4: Compute the dot product
    dot_product = torch.dot(B, D)
    # check_for_nan_inf(dot_product, "dot_product")

    # Return the result
    return torch.sqrt(torch.abs(dot_product))

# ---------------------------------------------------

# def primal_sol(out_put_mat,distances_mat):
#     # Flatten the matrices
#     bi_stoch_mat = primal_projection(out_put_mat,axis = 0)
#     B = bi_stoch_mat.view(-1)
#     D = (distances_mat**2).view(-1)
#     if torch.isnan(B).any():
#         print("NaN detected in B  before dot product")
#         return torch.tensor(float('nan'), device=out_put_mat.device)  # Handle NaN gracefully
#     if torch.isnan(D).any():
#         print("NaN detected in D  before dot product")
#         return torch.tensor(float('nan'), device=out_put_mat.device)  # Handle NaN gracefully

#     # Compute the dot product of the flattened matrices
#     dot_product = torch.dot(B, D)
#     return torch.sqrt(torch.abs(dot_product))
# # -----------------------------------------------------
# def primal_projection(matrix,axis = None):
#     ret_mat = normalize_matrix_axis(matrix = softmax(matrix = matrix,axis = axis),axis = axis).clone()
#     r = get_axis_sums(ret_mat,axis = 1)
#     c = get_axis_sums(ret_mat,axis = 0)
#     r_tild = torch.ones(len(ret_mat), device=matrix.device) - r
#     c_tild = torch.ones(len(ret_mat[0]), device=matrix.device) - c
#     ret_mat = ret_mat + (1 / torch.sum(c_tild)) * torch.outer(r_tild, c_tild)
#     return ret_mat
    
def dual_projection(f, g, cost):
    """
    Perform dual projection to obtain f_ret and g_ret.
    
    Parameters:
    f (torch.Tensor): A 1D tensor with the f values.
    g (torch.Tensor): A 1D tensor with the g values.
    cost (torch.Tensor): A 2D tensor with cost values where cost[i][k] is the cost value.

    Returns:
    torch.Tensor: f_ret tensor.
    torch.Tensor: g_ret tensor.
    """
    # Compute g_ret (which is simply g)
    g_ret = g.clone()
    
    # Compute the cost matrix subtraction
    cost_minus_g = cost - g.unsqueeze(1)  # Broadcasting to subtract g from each column in cost
    
    # Compute min_over_k(cost[i][k] - g[k])
    min_cost_minus_g = torch.min(cost_minus_g, dim=1).values
    
    # Compute f_ret
    f_ret = torch.minimum(f, min_cost_minus_g)
    
    return f_ret, g_ret

def dual_sol(f, g, distances_mat):
    projected_f, projected_g = dual_projection(f=f, g=g, cost=(distances_mat**2))  # Changed distances to squared
    sum_result = torch.sum(projected_f + projected_g)
    max_val = torch.maximum(sum_result, torch.tensor(0.0))  # Take the max of the sum and zero
    return torch.sqrt(max_val)
    
def generate_vector_batches(count: int, dim: int, n: int, device: torch.device, coord_max: float, seed: int = 42):
    """
    Generates a single tensor with shape (count, 2, n, dim) on the specified device,
    with each coordinate having values in the range [-coord_max, coord_max].
    
    Parameters:
    count (int): Number of batches to generate.
    dim (int): Dimension of each vector.
    n (int): Number of vectors in each batch.
    device (torch.device): The device on which to create the tensors.
    coord_max (float): The maximum absolute value for each coordinate.
    seed (int, optional): Random seed for reproducibility.
    
    Returns:
    torch.Tensor: A tensor with shape (count, 2, n, dim), with values in the range [-coord_max, coord_max].
    """
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate random values in the range [-coord_max, coord_max]
    batches = (2 * torch.rand((count, 2, n, dim), device=device) - 1) * coord_max
    
    return batches


def wass_hungarian(batch):
    """
    Given two sets of d-dimensional vectors, each of size n, computes the Wasserstein p=2 distance between them.
    
    This distance is found by using the Hungarian algorithm to find the optimal matching 
    that minimizes the sum of squared distances between corresponding vectors.
    
    Parameters:
    - batch: tensor of shape (2, n, d), where batch[0] and batch[1] are the two sets of n vectors to be compared.
    
    Returns:
    - Wasserstein distance (p=2) between the two sets of vectors.
    """
    
    # Extract the two sets of vectors
    vectors1 = batch[0]  # Shape: (n, d)
    vectors2 = batch[1]  # Shape: (n, d)

    # Compute the squared Euclidean distance matrix using broadcasting
    distance_matrix = torch.sum((vectors1[:, None, :] - vectors2[None, :, :]) ** 2, dim=-1)

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix.cpu().numpy())

    # Calculate the total minimum distance
    min_squared_distance = distance_matrix[row_ind, col_ind].sum()

    # Return the Wasserstein distance (p=2)
    return torch.sqrt(min_squared_distance)

def wass_permutations(batch, device):
    """
    Given two lists of d-dimensional vectors, each of size n, computes the Wasserstein p=2 distance between them.
    
    This distance is found by checking every possible permutation of vector couplings and selecting the one 
    that minimizes the sum of squared distances between corresponding vectors.
    
    Parameters:
    - batch: tensor of shape (2, n, d), where batch[0] and batch[1] are the two sets of n vectors to be compared.
    - device: 'cpu' or 'cuda' for GPU computation.
    
    Returns:
    - Wasserstein distance (p=2) between the two sets of vectors.
    """
    # Move batch to specified device (CPU or GPU)
    batch = batch.to(device)
    
    vectors1 = batch[0]
    vectors2 = batch[1]
    n = vectors1.shape[0]

    # Generate all permutations of size n
    all_permutations = list(itertools.permutations(range(n)))

    min_squared_distance = float('inf')

    # Loop over all permutations
    for perm in all_permutations:
        # Create an index tensor for the current permutation
        perm_tensor = torch.tensor(perm, device=device)

        # Compute the squared distance for this permutation
        squared_distance_sum = torch.sum((vectors1 - vectors2[perm_tensor]) ** 2)

        # Update minimum distance
        if squared_distance_sum < min_squared_distance:
            min_squared_distance = squared_distance_sum.item()  # Convert to Python float for comparison

    # Return the Wasserstein distance (sqrt of minimum squared distance)
    return torch.sqrt(torch.tensor(min_squared_distance, device=device))


