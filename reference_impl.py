"""
reference implementation of the butterfly matrix multiplication algorithm using torch and matmuls
used to verify the correctness of the optimized implementation
"""

import math, torch

def build_dense_from_stages(W_par, rightmost = True):
    """
    Convert butterfly stage parameters into a full dense LxL matrix.
    
    This function constructs the complete matrix W by multiplying together all
    butterfly stages. Each stage B_i is a sparse matrix that applies 2x2 transformations
    to element pairs with stride 2^i.
    
    Note: This is O(L^2) complexity and intended only for correctness verification.
    The optimized implementation avoids constructing the full matrix.
    
    Args:
        W_par: Butterfly parameters of shape (e, L, 2) where e = log2(L)
        rightmost: If True, multiply stages left-to-right (W_par[0] closest to input)
                   If False, multiply stages right-to-left (W_par[-1] closest to input)
    
    Returns:
        W_dense: Dense matrix of shape (L, L) representing the full transformation
    """    
    e, L, _ = W_par.shape
    identity = torch.eye(L, dtype=W_par.dtype, device=W_par.device)
    stages = range(e) if rightmost else reversed(range(e))

    W_dense = identity

    for i in stages: # construct and multiply each B_i
        stride = 1 << i # 2^i
        B_i = identity.clone()

        # For each block of 2*stride elements, we apply 2x2 transformations
        for base in range(0, L, 2*stride):
            # Create indices for the paired elements in this block
            idx = torch.arange(stride, device=W_par.device)  # [0, 1, ..., stride-1]
            rows0 = base + idx      # First element of each pair: [base, base+1, ..., base+stride-1]
            rows1 = base + stride + idx  # Second element of each pair: [base+stride, base+stride+1, ..., base+2*stride-1]

            # Extract butterfly coefficients for this block
            a0a1 = W_par[i, rows0]  # Coefficients [a0, a1] for first elements, shape (stride, 2)
            b0b1 = W_par[i, rows1]  # Coefficients [b0, b1] for second elements, shape (stride, 2)

            # Apply the 2x2 butterfly transformation to each pair
            # For each pair (j, j+stride), the transformation is:
            # [y_j, y_{j+stride}] = [a0, a1; b0, b1] * [x_j, x_{j+stride}]
            
            # Set the 2x2 blocks in B_i matrix
            B_i[rows0, rows0] = a0a1[:, 0]  # Top-left diagonal: a0 coefficients
            B_i[rows0, rows1] = a0a1[:, 1]  # Top-right diagonal: a1 coefficients  
            B_i[rows1, rows0] = b0b1[:, 0]  # Bottom-left diagonal: b0 coefficients
            B_i[rows1, rows1] = b0b1[:, 1]  # Bottom-right diagonal: b1 coefficients
            
        # Multiply the current dense matrix by this butterfly stage
        W_dense = W_dense @ B_i  # Right multiplication: W = W * B_i

    return W_dense

def butterfly_mm_ref(X, W_par, rightmost=True):
    """
    Compute butterfly matrix multiplication using reference implementation.
    
    This function computes Y = X · W, where W is represented as a product of log2(L)
    butterfly matrices. Each butterfly stage applies sparse 2x2 transformations to
    element pairs with increasing stride.
    
    Args:
        X: Input tensor of shape (B, F, L) where L must be a power of 2
        W_par: Butterfly parameters of shape (e, L, 2) where e = log2(L)
        rightmost: If True, apply stages left-to-right (W_par[0] closest to X)
                   If False, apply stages right-to-left (W_par[-1] closest to X)
    
    Returns:
        Y: Output tensor of shape (B, F, L)
    """

    B, F, L = X.shape
    e = int(math.log2(L))
    
    # Validate input shapes and constraints
    assert W_par.shape == (e, L, 2), f"W_par has wrong shape: expected ({e}, {L}, 2), got {W_par.shape}"
    assert 1 << e == L, f"L={L} must be a power of 2 (expected 2^{e}={1<<e})"

    Y = X.clone()
    stage_order = range(e) if rightmost else reversed(range(e))

    for i in stage_order:
        stride = 1 << i # distance between paired indices
        for base in range(0, L, 2*stride):
            # Define slices for the paired element groups
            j0 = slice(base, base + stride)      # First group: [base, base+1, ..., base+stride-1]
            j1 = slice(base + stride, base + 2*stride)  # Second group: [base+stride, base+stride+1, ..., base+2*stride-1]

            # Extract the paired elements for all batches and features
            v0 = Y[:, :, j0].clone()  # Shape (B, F, stride)
            v1 = Y[:, :, j1].clone()  # Shape (B, F, stride)

            # Extract butterfly coefficients and broadcast to match tensor dimensions
            # Each coefficient needs to be broadcast to (B, F, stride) for element-wise operations
            a0 = W_par[i, j0, 0].unsqueeze(0).unsqueeze(0)  # Add batch and feature dimensions
            a1 = W_par[i, j0, 1].unsqueeze(0).unsqueeze(0)
            b0 = W_par[i, j1, 0].unsqueeze(0).unsqueeze(0)
            b1 = W_par[i, j1, 1].unsqueeze(0).unsqueeze(0)

            # Apply the 2x2 butterfly transformation to each element pair
            # [y_j0, y_j1] = [a0*v0 + b0*v1, a1*v0 + b1*v1]
            Y[:, :, j0] = a0 * v0 + b0 * v1
            Y[:, :, j1] = a1 * v0 + b1 * v1
            
    return Y


if __name__ == "__main__":
    # Simple correctness test: compare butterfly multiplication with dense matrix multiplication
    torch.manual_seed(0)
    B, F, L = 3, 5, 16
    e = int(math.log2(L))

    # Generate random test data
    X = torch.randn(B, F, L)
    W_par = torch.randn(e, L, 2)

    # Compute result using butterfly algorithm
    Y_ref = butterfly_mm_ref(X, W_par)
    
    # Compute result using dense matrix multiplication for verification
    W_dense = build_dense_from_stages(W_par)
    Y_check = X @ W_dense  # Standard matrix multiplication: (B, F, L) * (L, L) = (B, F, L)

    # Check correctness
    max_error = (Y_ref - Y_check).abs().max().item()
    print(f"Maximum absolute difference: {max_error:.2e}")
    print("Test passed!" if max_error < 1e-6 else "Test failed!")