"""
Pipelined butterfly kernel: moves the next load earlier to test load–compute–store pipelining
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import triton
import triton.language as tl
import time
import csv

@triton.jit
def butterfly_stage_kernel_pipelined(
    x_ptr, w_ptr,
    BF: tl.constexpr, L: tl.constexpr,
    stride_BF: tl.constexpr, BLOCK_BF: tl.constexpr, STRIDE: tl.constexpr
):
    pid_block = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    base = pid_block * 2 * STRIDE
    row0 = pid_row * BLOCK_BF + tl.arange(0, BLOCK_BF)
    mask = row0 < BF

    # Pipelined version: unroll over stride, prefetch next pair's data
    if STRIDE >= 2:
        # Prefetch first pair
        j0 = base + 0
        j1 = base + STRIDE + 0
        v0_offsets = row0 * stride_BF + j0
        v1_offsets = row0 * stride_BF + j1
        v0 = tl.load(x_ptr + v0_offsets, mask=mask)
        v1 = tl.load(x_ptr + v1_offsets, mask=mask)
        a0 = tl.load(w_ptr + (j0 * 2 + 0))
        a1 = tl.load(w_ptr + (j0 * 2 + 1))
        b0 = tl.load(w_ptr + (j1 * 2 + 0))
        b1 = tl.load(w_ptr + (j1 * 2 + 1))
        for k in range(STRIDE - 1):
            # Prefetch next
            j0n = base + (k + 1)
            j1n = base + STRIDE + (k + 1)
            v0n_offsets = row0 * stride_BF + j0n
            v1n_offsets = row0 * stride_BF + j1n
            v0n = tl.load(x_ptr + v0n_offsets, mask=mask)
            v1n = tl.load(x_ptr + v1n_offsets, mask=mask)
            a0n = tl.load(w_ptr + (j0n * 2 + 0))
            a1n = tl.load(w_ptr + (j0n * 2 + 1))
            b0n = tl.load(w_ptr + (j1n * 2 + 0))
            b1n = tl.load(w_ptr + (j1n * 2 + 1))
            # Compute/store current
            y0 = a0 * v0 + b0 * v1
            y1 = a1 * v0 + b1 * v1
            tl.store(x_ptr + v0_offsets, y0, mask=mask)
            tl.store(x_ptr + v1_offsets, y1, mask=mask)
            # Move next to current
            v0, v1 = v0n, v1n
            a0, a1, b0, b1 = a0n, a1n, b0n, b1n
            v0_offsets, v1_offsets = v0n_offsets, v1n_offsets
        # Compute/store last
        y0 = a0 * v0 + b0 * v1
        y1 = a1 * v0 + b1 * v1
        tl.store(x_ptr + v0_offsets, y0, mask=mask)
        tl.store(x_ptr + v1_offsets, y1, mask=mask)
    else:
        # Fallback to original logic for STRIDE == 1
        j0 = base + 0
        j1 = base + STRIDE + 0
        v0_offsets = row0 * stride_BF + j0
        v1_offsets = row0 * stride_BF + j1
        v0 = tl.load(x_ptr + v0_offsets, mask=mask)
        v1 = tl.load(x_ptr + v1_offsets, mask=mask)
        a0 = tl.load(w_ptr + (j0 * 2 + 0))
        a1 = tl.load(w_ptr + (j0 * 2 + 1))
        b0 = tl.load(w_ptr + (j1 * 2 + 0))
        b1 = tl.load(w_ptr + (j1 * 2 + 1))
        y0 = a0 * v0 + b0 * v1
        y1 = a1 * v0 + b1 * v1
        tl.store(x_ptr + v0_offsets, y0, mask=mask)
        tl.store(x_ptr + v1_offsets, y1, mask=mask)


def apply_one_stage_pipelined(X, w_stage, stride, BLOCK_BF=64):
    B, F, L = X.shape
    BF = B * F
    X_2d = X.view(BF, L)
    grid = (L // (2 * stride), (BF + BLOCK_BF - 1) // BLOCK_BF)
    butterfly_stage_kernel_pipelined[grid](
        X_2d, w_stage,
        BF, L,
        stride_BF=L,
        BLOCK_BF=BLOCK_BF,
        STRIDE=stride
    )

def butterfly_mm_pipelined(X, W_par, rightmost=True, BLOCK_BF=64):
    B, F, L = X.shape
    e = int(math.log2(L))
    stage_order = range(e) if rightmost else reversed(range(e))
    Y = X.clone()
    for i in stage_order:
        stride = 1 << i
        apply_one_stage_pipelined(Y, W_par[i], stride, BLOCK_BF=BLOCK_BF)
    return Y

# ===================== Benchmarking =====================
def benchmark_pipelined_vs_baseline(B=4, F=64, L=1024, dtype=torch.float32, device='cuda', reps=100):
    from triton_kernel import butterfly_mm_triton
    torch.manual_seed(42)
    e = int(math.log2(L))
    X = torch.randn(B, F, L, device=device, dtype=dtype)
    Wp = torch.randn(e, L, 2, device=device, dtype=dtype)
    # Warmup
    for _ in range(5):
        _ = butterfly_mm_triton(X, Wp)
        _ = butterfly_mm_pipelined(X, Wp)
        torch.cuda.synchronize()
    # Baseline timing
    t0 = time.time()
    for _ in range(reps):
        Y0 = butterfly_mm_triton(X, Wp)
    torch.cuda.synchronize()
    t_baseline = (time.time() - t0) / reps
    # Pipelined timing
    t0 = time.time()
    for _ in range(reps):
        Y1 = butterfly_mm_pipelined(X, Wp)
    torch.cuda.synchronize()
    t_pipelined = (time.time() - t0) / reps
    # Check correctness
    max_diff = (Y0 - Y1).abs().max().item()
    print(f"[B={B} F={F} L={L}] Baseline: {t_baseline*1e3:.2f} ms | Pipelined: {t_pipelined*1e3:.2f} ms | Max diff: {max_diff:.2e}")
    return t_baseline, t_pipelined, max_diff

if __name__ == "__main__":
    configs = [
        (4, 64, 256),
        (4, 64, 1024),
        (4, 128, 2048),
        (8, 64, 4096),
    ]
    csv_dir = "results/optimization"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "pipelined_vs_baseline_results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["B", "F", "L", "baseline_ms", "pipelined_ms", "max_diff"])
        for B, F, L in configs:
            t_base, t_pipe, max_diff = benchmark_pipelined_vs_baseline(B, F, L)
            writer.writerow([B, F, L, t_base * 1e3, t_pipe * 1e3, max_diff])
    print(f"\nResults saved to {csv_path}") 