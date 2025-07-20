"""
optimized implementation of the butterfly matrix multiplication algorithm using triton
"""

import math
import torch
import triton
import triton.language as tl

def butterfly_mm_triton(X, W_par, rightmost=True):
    B, F, L = X.shape
    e = int(math.log2(L))
    stage_order = range(e) if rightmost else reversed(range(e))

    Y = X.clone()
    for i in stage_order:
        # Stride for this stage: 2^i determines which elements are paired
        # Stage 0: pairs adjacent elements (stride=1)
        # Stage 1: pairs elements with distance 2 (stride=2)
        # Stage 2: pairs elements with distance 4 (stride=4), etc.
        stride = 1 << i
        apply_one_stage(Y, W_par[i], stride)
    return Y

def apply_one_stage(X, w_stage, stride, BLOCK_BF=64):
    """
    Apply a single butterfly stage using Triton kernel.
    
    Each stage performs a sparse matrix multiplication where elements are paired
    according to the stride. For stride s, element j is paired with element j+s.
    The transformation is: [y_j, y_{j+s}] = [a0, a1; b0, b1] * [x_j, x_{j+s}]
    
    Args:
        X: Input/output tensor of shape (B, F, L), must be contiguous in L dimension
        w_stage: Butterfly coefficients for this stage, shape (L, 2)
        stride: Distance between paired elements (2^i for stage i)
        BLOCK_BF: Number of rows (B*F) processed by each Triton program
    """
    B, F, L = X.shape
    BF = B * F
    # Reshape to 2D for easier indexing in the kernel
    # This is a view operation, no memory copy occurs
    X_2d = X.view(BF, L)

    # Grid configuration for Triton kernel
    # - First dimension: number of 2*stride blocks to process
    # - Second dimension: number of row blocks to process
    grid = (L // (2 * stride), (BF + BLOCK_BF - 1) // BLOCK_BF)

    butterfly_stage_kernel[grid](
        X_2d, w_stage,
        BF, L,
        stride_BF=L,  # Row stride in X_2d (each row has L elements)
        BLOCK_BF=BLOCK_BF,
        STRIDE=stride  # Distance between paired elements
    )

@triton.jit
def butterfly_stage_kernel(
        x_ptr,            # Pointer to BF × L tensor (row-major, contiguous in L)
        w_ptr,            # Pointer to L × 2 tensor (row-major)
        BF: tl.constexpr, # Total number of rows = B*F
        L:  tl.constexpr, # Length of each row
        stride_BF: tl.constexpr,  # Row stride in x = L
        BLOCK_BF: tl.constexpr,   # Number of rows each program processes
        STRIDE: tl.constexpr):    # Distance between paired elements = 2^stage

    # Program ID determines which block of work this program handles
    pid_block = tl.program_id(axis=0)  # Which block of element pairs to process
    pid_row   = tl.program_id(axis=1)  # Which block of rows to process

    # ---- Compute absolute indices for this program ----
    # Each program processes a block of STRIDE element pairs
    base = pid_block * 2 * STRIDE  # Starting index for this block of pairs
    row0 = pid_row * BLOCK_BF + tl.arange(0, BLOCK_BF)  # Row indices for this program
    
    # Create indices for all element pairs in this block
    block_indices = tl.arange(0, STRIDE)
    # j0 indices: [base, base+1, ..., base+stride-1]
    j0_indices = base + block_indices
    # j1 indices: [base+stride, base+stride+1, ..., base+2*stride-1]
    j1_indices = base + STRIDE + block_indices

    # Mask to handle the last partial block (when BF is not divisible by BLOCK_BF)
    mask = row0 < BF

    # ---- Load input values for the entire block ----
    # Load v0 values for all paired elements in this block
    # v0_offsets shape: (BLOCK_BF, STRIDE) - each row gets STRIDE elements
    v0_offsets = row0[:, None] * stride_BF + j0_indices[None, :]
    v0 = tl.load(x_ptr + v0_offsets, mask=mask[:, None])
    
    # Load v1 values for all paired elements in this block
    v1_offsets = row0[:, None] * stride_BF + j1_indices[None, :]
    v1 = tl.load(x_ptr + v1_offsets, mask=mask[:, None])

    # ---- Load butterfly coefficients for the entire block ----
    # Each element pair (j0, j1) has coefficients [a0, a1] and [b0, b1]
    # such that: [y_j0, y_j1] = [a0, a1; b0, b1] * [v0, v1]
    
    # Load coefficients for j0 indices: [a0, a1] for each j0
    a0_offsets = j0_indices * 2 + 0  # Even indices in w_stage
    a1_offsets = j0_indices * 2 + 1  # Odd indices in w_stage
    a0 = tl.load(w_ptr + a0_offsets)  # Shape: (STRIDE,)
    a1 = tl.load(w_ptr + a1_offsets)  # Shape: (STRIDE,)
    
    # Load coefficients for j1 indices: [b0, b1] for each j1
    b0_offsets = j1_indices * 2 + 0  # Even indices in w_stage
    b1_offsets = j1_indices * 2 + 1  # Odd indices in w_stage
    b0 = tl.load(w_ptr + b0_offsets)  # Shape: (STRIDE,)
    b1 = tl.load(w_ptr + b1_offsets)  # Shape: (STRIDE,)

    # Broadcast coefficients to match v0/v1 shapes for element-wise operations
    # From (STRIDE,) to (1, STRIDE) to broadcast with (BLOCK_BF, STRIDE)
    a0 = a0[None, :]  # Shape: (1, STRIDE)
    a1 = a1[None, :]  # Shape: (1, STRIDE)
    b0 = b0[None, :]  # Shape: (1, STRIDE)
    b1 = b1[None, :]  # Shape: (1, STRIDE)

    # ---- Apply butterfly transformation for the entire block ----
    # Compute: [y0, y1] = [a0*v0 + b0*v1, a1*v0 + b1*v1]
    # This applies the 2x2 butterfly matrix to each element pair
    y0 = a0 * v0 + b0 * v1  # Shape: (BLOCK_BF, STRIDE)
    y1 = a1 * v0 + b1 * v1  # Shape: (BLOCK_BF, STRIDE)
    
    # ---- store results ----
    tl.store(x_ptr + v0_offsets, y0, mask=mask[:, None])
    tl.store(x_ptr + v1_offsets, y1, mask=mask[:, None])