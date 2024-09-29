import torch
import numpy as np
import pandas as pd
# -----------------------------------------------
def softmax(matrix,axis = None,T=1):
    exp_matrix = torch.exp(matrix/T - torch.max(matrix, dim=axis, keepdim=True).values)
    softmax_matrix = exp_matrix / torch.sum(exp_matrix, dim=axis, keepdim=True)
    return softmax_matrix


def normalize_axis(to_normalize, axis=None):
    axis_sum = torch.sum(to_normalize, dim=axis, keepdim=True)
    
    # Normalize only if the sum is greater than 1
    if torch.any(axis_sum > 1):
        return to_normalize / axis_sum
    return to_normalize


def normalize_matrix_axis(matrix,axis = None):
    slices = [normalize_axis(to_normalize=matrix.select(axis, i), axis=None) for i in range(matrix.size(axis))]
    return torch.stack(slices, dim=axis)


def get_axis_sums(matrix,axis = None):
    return torch.sum(matrix, dim=axis)


def primal_projection(matrix,axis = None):
    ret_mat = normalize_matrix_axis(matrix = softmax(matrix = matrix,axis = axis),axis = axis).clone()
    r = get_axis_sums(ret_mat,axis = 1)
    c = get_axis_sums(ret_mat,axis = 0)
    r_tild = torch.ones(len(ret_mat), device=matrix.device) - r
    c_tild = torch.ones(len(ret_mat[0]), device=matrix.device) - c
    ret_mat = ret_mat + (1 / torch.sum(c_tild)) * torch.outer(r_tild, c_tild)
    return ret_mat

def primal_sol(out_put_mat,distances_mat):
    # Flatten the matrices
    bi_stoch_mat = primal_projection(out_put_mat,axis = 0)
    B = bi_stoch_mat.view(-1)
    D = (distances_mat**2).view(-1)
    
    # Compute the dot product of the flattened matrices
    dot_product = torch.dot(B, D)
    return torch.sqrt(dot_product)

def primal_projection(matrix,axis = None):
    ret_mat = normalize_matrix_axis(matrix = softmax(matrix = matrix,axis = axis),axis = axis).clone()
    r = get_axis_sums(ret_mat,axis = 1)
    c = get_axis_sums(ret_mat,axis = 0)
    r_tild = torch.ones(len(ret_mat), device=matrix.device) - r
    c_tild = torch.ones(len(ret_mat[0]), device=matrix.device) - c
    ret_mat = ret_mat + (1 / torch.sum(c_tild)) * torch.outer(r_tild, c_tild)
    return ret_mat
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
    # Ensure input tensors are on the same device
    device = f.device
    
    # Compute g_ret (which is simply g)
    g_ret = g.clone()
    
    # Compute the cost matrix subtraction
    cost_minus_g = cost - g.unsqueeze(1)  # Broadcasting to subtract g from each column in cost
    
    # Compute min_over_k(cost[i][k] - g[k])
    min_cost_minus_g = torch.min(cost_minus_g, dim=1).values
    
    # Compute f_ret
    f_ret = torch.minimum(f, min_cost_minus_g)
    
    return f_ret, g_ret

def dual_sol(f,g,distances_mat):
    projected_f,projected_g = dual_projection(f=f, g=g , cost =distances_mat)
    return torch.sum(projected_f + projected_g) 
    
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


