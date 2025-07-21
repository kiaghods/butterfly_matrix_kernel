"""
butterfly kernel with num_warps tuning (4, 8, 16 warps)
Includes benchmarking logic to compare performance for different num_warps values and outputs results as CSV
"""
import torch
import time
import triton
import triton.language as tl
import math
import os
import csv

@triton.jit
def butterfly_stage_kernel_4_warps(
        x_ptr, w_ptr,
        BF: tl.constexpr, L: tl.constexpr,
        stride_BF: tl.constexpr, STRIDE: tl.constexpr,
        BLOCK_BF: tl.constexpr):
    """
    Kernel for butterfly stage computation with 4 warps.
    Computes y0 and y1 for a given block of rows (BLOCK_BF) and updates x_ptr.
    """
    pid_block = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    base = pid_block * 2 * STRIDE
    row0 = pid_row * BLOCK_BF + tl.arange(0, BLOCK_BF)
    mask = row0 < BF
    j0_indices = base + tl.arange(0, STRIDE)
    j1_indices = base + STRIDE + tl.arange(0, STRIDE)
    v0_offsets = row0[:, None] * stride_BF + j0_indices[None, :]
    v0 = tl.load(x_ptr + v0_offsets, mask=mask[:, None])
    v1_offsets = row0[:, None] * stride_BF + j1_indices[None, :]
    v1 = tl.load(x_ptr + v1_offsets, mask=mask[:, None])
    a0 = tl.load(w_ptr + (j0_indices * 2 + 0)); a1 = tl.load(w_ptr + (j0_indices * 2 + 1))
    b0 = tl.load(w_ptr + (j1_indices * 2 + 0)); b1 = tl.load(w_ptr + (j1_indices * 2 + 1))
    y0_computed = a0[None, :] * v0 + b0[None, :] * v1
    y1_computed = a1[None, :] * v0 + b1[None, :] * v1
    y0 = tl.where(mask[:, None], y0_computed, v0)
    y1 = tl.where(mask[:, None], y1_computed, v1)
    tl.store(x_ptr + v0_offsets, y0, mask=mask[:, None])
    tl.store(x_ptr + v1_offsets, y1, mask=mask[:, None])

@triton.jit
def butterfly_stage_kernel_8_warps(
        x_ptr, w_ptr,
        BF: tl.constexpr, L: tl.constexpr,
        stride_BF: tl.constexpr, STRIDE: tl.constexpr,
        BLOCK_BF: tl.constexpr):
    """
    Kernel for butterfly stage computation with 8 warps.
    Computes y0 and y1 for a given block of rows (BLOCK_BF) and updates x_ptr.
    """
    pid_block = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    base = pid_block * 2 * STRIDE
    row0 = pid_row * BLOCK_BF + tl.arange(0, BLOCK_BF)
    mask = row0 < BF
    j0_indices = base + tl.arange(0, STRIDE)
    j1_indices = base + STRIDE + tl.arange(0, STRIDE)
    v0_offsets = row0[:, None] * stride_BF + j0_indices[None, :]
    v0 = tl.load(x_ptr + v0_offsets, mask=mask[:, None])
    v1_offsets = row0[:, None] * stride_BF + j1_indices[None, :]
    v1 = tl.load(x_ptr + v1_offsets, mask=mask[:, None])
    a0 = tl.load(w_ptr + (j0_indices * 2 + 0)); a1 = tl.load(w_ptr + (j0_indices * 2 + 1))
    b0 = tl.load(w_ptr + (j1_indices * 2 + 0)); b1 = tl.load(w_ptr + (j1_indices * 2 + 1))
    y0_computed = a0[None, :] * v0 + b0[None, :] * v1
    y1_computed = a1[None, :] * v0 + b1[None, :] * v1
    y0 = tl.where(mask[:, None], y0_computed, v0)
    y1 = tl.where(mask[:, None], y1_computed, v1)
    tl.store(x_ptr + v0_offsets, y0, mask=mask[:, None])
    tl.store(x_ptr + v1_offsets, y1, mask=mask[:, None])

@triton.jit
def butterfly_stage_kernel_16_warps(
        x_ptr, w_ptr,
        BF: tl.constexpr, L: tl.constexpr,
        stride_BF: tl.constexpr, STRIDE: tl.constexpr,
        BLOCK_BF: tl.constexpr):
    """
    Kernel for butterfly stage computation with 16 warps.
    Computes y0 and y1 for a given block of rows (BLOCK_BF) and updates x_ptr.
    """
    pid_block = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    base = pid_block * 2 * STRIDE
    row0 = pid_row * BLOCK_BF + tl.arange(0, BLOCK_BF)
    mask = row0 < BF
    j0_indices = base + tl.arange(0, STRIDE)
    j1_indices = base + STRIDE + tl.arange(0, STRIDE)
    v0_offsets = row0[:, None] * stride_BF + j0_indices[None, :]
    v0 = tl.load(x_ptr + v0_offsets, mask=mask[:, None])
    v1_offsets = row0[:, None] * stride_BF + j1_indices[None, :]
    v1 = tl.load(x_ptr + v1_offsets, mask=mask[:, None])
    a0 = tl.load(w_ptr + (j0_indices * 2 + 0)); a1 = tl.load(w_ptr + (j0_indices * 2 + 1))
    b0 = tl.load(w_ptr + (j1_indices * 2 + 0)); b1 = tl.load(w_ptr + (j1_indices * 2 + 1))
    y0_computed = a0[None, :] * v0 + b0[None, :] * v1
    y1_computed = a1[None, :] * v0 + b1[None, :] * v1
    y0 = tl.where(mask[:, None], y0_computed, v0)
    y1 = tl.where(mask[:, None], y1_computed, v1)
    tl.store(x_ptr + v0_offsets, y0, mask=mask[:, None])
    tl.store(x_ptr + v1_offsets, y1, mask=mask[:, None])

def apply_one_stage_numwarp_tuning(X, w_stage, stride, BLOCK_BF: int, num_warps: int):
    """
    Wrapper for the numwarp-tuning butterfly kernel. Selects kernel based on num_warps.
    Args:
        X: Input/output tensor of shape (B, F, L)
        w_stage: Butterfly coefficients for this stage, shape (L, 2)
        stride: Distance between paired elements (2^i for stage i)
        BLOCK_BF: Number of rows processed per block (fixed to 64 for best perf)
        num_warps: Number of warps (4, 8, or 16)
    """
    B, F, L = X.shape
    BF = B * F
    X_2d = X.view(BF, L)
    grid = (L // (2 * stride), (BF + BLOCK_BF - 1) // BLOCK_BF)
    if num_warps == 4:
        butterfly_stage_kernel_4_warps[grid](
            X_2d, w_stage, BF, L, L, stride, BLOCK_BF=BLOCK_BF, num_warps=4
        )
    elif num_warps == 8:
        butterfly_stage_kernel_8_warps[grid](
            X_2d, w_stage, BF, L, L, stride, BLOCK_BF=BLOCK_BF, num_warps=8
        )
    elif num_warps == 16:
        butterfly_stage_kernel_16_warps[grid](
            X_2d, w_stage, BF, L, L, stride, BLOCK_BF=BLOCK_BF, num_warps=16
        )
    else:
        raise ValueError("Unsupported num_warps. Must be 4, 8, or 16.")

def butterfly_mm_numwarp_tuning(X, W_par, BLOCK_BF=64, num_warps=4):
    """
    Full butterfly matmul using the numwarp-tuning kernel.
    Args:
        X: Input tensor of shape (B, F, L)
        W_par: List of butterfly coefficients for each stage
        BLOCK_BF: Number of rows per block (default 64)
        num_warps: Number of warps (default 4)
    Returns:
        Output tensor of shape (B, F, L)
    """
    Y = X.clone()
    e = int(math.log2(X.shape[2]))
    for i in range(e):
        apply_one_stage_numwarp_tuning(Y, W_par[i], 1 << i, BLOCK_BF, num_warps)
    return Y

def main():
    """
    Benchmark num_warps for a range of L and print & save results as CSV.
    """
    B, F = 8, 8
    BLOCK_BF = 64
    num_warps_options = [4, 8, 16]
    L_values = [128, 256, 512, 1024, 2048]
    print("NumWarp Tuning Benchmark:")
    csv_dir = "results/optimization"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "numwarp_tuning_results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["L", "num_warps", "time_ms"])
        for L in L_values:
            X = torch.randn(B, F, L, device='cuda', dtype=torch.float32)
            w_stage = torch.randn(L, 2, device='cuda', dtype=torch.float32)
            times = {}
            for nw in num_warps_options:
                # Warmup
                for _ in range(3):
                    apply_one_stage_numwarp_tuning(X, w_stage, stride=1, BLOCK_BF=BLOCK_BF, num_warps=nw)
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(10):
                    apply_one_stage_numwarp_tuning(X, w_stage, stride=1, BLOCK_BF=BLOCK_BF, num_warps=nw)
                torch.cuda.synchronize()
                elapsed = (time.time() - start) / 10 * 1000  # ms
                times[nw] = elapsed
                writer.writerow([L, nw, elapsed])
            print(f"L={L}: " + ", ".join([f"num_warps={nw}: {times[nw]:.3f} ms" for nw in num_warps_options]))
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    main() 