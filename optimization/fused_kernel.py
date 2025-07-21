"""
fused two-stage butterfly kernel: applies two butterfly stages in a single Triton kernel for small L (L=8)
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
def butterfly_stage_kernel_fused_L8(
    x_ptr, w1_ptr, w2_ptr,
    BF: tl.constexpr, stride_BF: tl.constexpr
):
    L = 8
    pid_row = tl.program_id(axis=0)
    row = pid_row
    if row >= BF:
        return
    offsets = row * stride_BF + tl.arange(0, 8)
    x_row = tl.load(x_ptr + offsets)
    idx = tl.arange(0, 8)

    # ---- First stage: pairs (0,1), (2,3), (4,5), (6,7) ----
    y_row = x_row
    # (0, 1)
    mask0 = idx == 0
    mask1 = idx == 1
    a0 = tl.load(w1_ptr + 0 * 2 + 0)
    a1 = tl.load(w1_ptr + 0 * 2 + 1)
    b0 = tl.load(w1_ptr + 1 * 2 + 0)
    b1 = tl.load(w1_ptr + 1 * 2 + 1)
    v0 = tl.sum(x_row * mask0, axis=0)
    v1 = tl.sum(x_row * mask1, axis=0)
    y0 = a0 * v0 + b0 * v1
    y1 = a1 * v0 + b1 * v1
    y_row = tl.where(mask0, y0, y_row)
    y_row = tl.where(mask1, y1, y_row)
    # (2, 3)
    mask0 = idx == 2
    mask1 = idx == 3
    a0 = tl.load(w1_ptr + 2 * 2 + 0)
    a1 = tl.load(w1_ptr + 2 * 2 + 1)
    b0 = tl.load(w1_ptr + 3 * 2 + 0)
    b1 = tl.load(w1_ptr + 3 * 2 + 1)
    v0 = tl.sum(x_row * mask0, axis=0)
    v1 = tl.sum(x_row * mask1, axis=0)
    y0 = a0 * v0 + b0 * v1
    y1 = a1 * v0 + b1 * v1
    y_row = tl.where(mask0, y0, y_row)
    y_row = tl.where(mask1, y1, y_row)
    # (4, 5)
    mask0 = idx == 4
    mask1 = idx == 5
    a0 = tl.load(w1_ptr + 4 * 2 + 0)
    a1 = tl.load(w1_ptr + 4 * 2 + 1)
    b0 = tl.load(w1_ptr + 5 * 2 + 0)
    b1 = tl.load(w1_ptr + 5 * 2 + 1)
    v0 = tl.sum(x_row * mask0, axis=0)
    v1 = tl.sum(x_row * mask1, axis=0)
    y0 = a0 * v0 + b0 * v1
    y1 = a1 * v0 + b1 * v1
    y_row = tl.where(mask0, y0, y_row)
    y_row = tl.where(mask1, y1, y_row)
    # (6, 7)
    mask0 = idx == 6
    mask1 = idx == 7
    a0 = tl.load(w1_ptr + 6 * 2 + 0)
    a1 = tl.load(w1_ptr + 6 * 2 + 1)
    b0 = tl.load(w1_ptr + 7 * 2 + 0)
    b1 = tl.load(w1_ptr + 7 * 2 + 1)
    v0 = tl.sum(x_row * mask0, axis=0)
    v1 = tl.sum(x_row * mask1, axis=0)
    y0 = a0 * v0 + b0 * v1
    y1 = a1 * v0 + b1 * v1
    y_row = tl.where(mask0, y0, y_row)
    y_row = tl.where(mask1, y1, y_row)

    # ---- Second stage: pairs (0,2), (1,3), (4,6), (5,7) ----
    z_row = y_row
    # (0, 2)
    mask0 = idx == 0
    mask1 = idx == 2
    a0 = tl.load(w2_ptr + 0 * 2 + 0)
    a1 = tl.load(w2_ptr + 0 * 2 + 1)
    b0 = tl.load(w2_ptr + 2 * 2 + 0)
    b1 = tl.load(w2_ptr + 2 * 2 + 1)
    v0 = tl.sum(y_row * mask0, axis=0)
    v1 = tl.sum(y_row * mask1, axis=0)
    y0 = a0 * v0 + b0 * v1
    y1 = a1 * v0 + b1 * v1
    z_row = tl.where(mask0, y0, z_row)
    z_row = tl.where(mask1, y1, z_row)
    # (1, 3)
    mask0 = idx == 1
    mask1 = idx == 3
    a0 = tl.load(w2_ptr + 1 * 2 + 0)
    a1 = tl.load(w2_ptr + 1 * 2 + 1)
    b0 = tl.load(w2_ptr + 3 * 2 + 0)
    b1 = tl.load(w2_ptr + 3 * 2 + 1)
    v0 = tl.sum(y_row * mask0, axis=0)
    v1 = tl.sum(y_row * mask1, axis=0)
    y0 = a0 * v0 + b0 * v1
    y1 = a1 * v0 + b1 * v1
    z_row = tl.where(mask0, y0, z_row)
    z_row = tl.where(mask1, y1, z_row)
    # (4, 6)
    mask0 = idx == 4
    mask1 = idx == 6
    a0 = tl.load(w2_ptr + 4 * 2 + 0)
    a1 = tl.load(w2_ptr + 4 * 2 + 1)
    b0 = tl.load(w2_ptr + 6 * 2 + 0)
    b1 = tl.load(w2_ptr + 6 * 2 + 1)
    v0 = tl.sum(y_row * mask0, axis=0)
    v1 = tl.sum(y_row * mask1, axis=0)
    y0 = a0 * v0 + b0 * v1
    y1 = a1 * v0 + b1 * v1
    z_row = tl.where(mask0, y0, z_row)
    z_row = tl.where(mask1, y1, z_row)
    # (5, 7)
    mask0 = idx == 5
    mask1 = idx == 7
    a0 = tl.load(w2_ptr + 5 * 2 + 0)
    a1 = tl.load(w2_ptr + 5 * 2 + 1)
    b0 = tl.load(w2_ptr + 7 * 2 + 0)
    b1 = tl.load(w2_ptr + 7 * 2 + 1)
    v0 = tl.sum(y_row * mask0, axis=0)
    v1 = tl.sum(y_row * mask1, axis=0)
    y0 = a0 * v0 + b0 * v1
    y1 = a1 * v0 + b1 * v1
    z_row = tl.where(mask0, y0, z_row)
    z_row = tl.where(mask1, y1, z_row)

    tl.store(x_ptr + offsets, z_row)

def apply_two_stage_fused(X, w1, w2):
    B, F, L = X.shape
    assert L == 8
    BF = B * F
    X_2d = X.view(BF, L)
    grid = (BF,)
    butterfly_stage_kernel_fused_L8[grid](
        X_2d, w1, w2,
        BF, stride_BF=L
    )

def butterfly_mm_two_stage_fused(X, W_par):
    # Only fuses the first two stages, then returns output
    B, F, L = X.shape
    e = int(math.log2(L))
    assert e >= 2 and L == 8, "Need L=8 for the hardcoded fused kernel"
    Y = X.clone()
    apply_two_stage_fused(Y, W_par[0], W_par[1])
    return Y

# ===================== Benchmarking =====================
def benchmark_fused_vs_baseline(B=4, F=64, L=8, dtype=torch.float32, device='cuda', reps=100):
    from triton_kernel import butterfly_mm_triton
    torch.manual_seed(42)
    e = int(math.log2(L))
    X = torch.randn(B, F, L, device=device, dtype=dtype)
    Wp = torch.randn(e, L, 2, device=device, dtype=dtype)
    # Warmup
    for _ in range(5):
        _ = butterfly_mm_triton(X, Wp)
        _ = butterfly_mm_two_stage_fused(X, Wp)
        torch.cuda.synchronize()
    # Baseline timing (first two stages only)
    def baseline_two_stage(X, Wp):
        Y = X.clone()
        from triton_kernel import apply_one_stage
        apply_one_stage(Y, Wp[0], 1, BLOCK_BF=64)
        apply_one_stage(Y, Wp[1], 2, BLOCK_BF=64)
        return Y
    t0 = time.time()
    for _ in range(reps):
        Y0 = baseline_two_stage(X, Wp)
    torch.cuda.synchronize()
    t_baseline = (time.time() - t0) / reps
    # Fused timing
    t0 = time.time()
    for _ in range(reps):
        Y1 = butterfly_mm_two_stage_fused(X, Wp)
    torch.cuda.synchronize()
    t_fused = (time.time() - t0) / reps
    # Check correctness
    max_diff = (Y0 - Y1).abs().max().item()
    print(f"[B={B} F={F} L={L}] Baseline: {t_baseline*1e3:.2f} ms | Fused: {t_fused*1e3:.2f} ms | Max diff: {max_diff:.2e}")
    return t_baseline, t_fused, max_diff

if __name__ == "__main__":
    # Only run for L=8
    configs = [
        (4, 64, 8),
    ]
    csv_dir = "results/optimization"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "fused_vs_baseline_results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["B", "F", "L", "baseline_ms", "fused_ms", "max_diff"])
        for B, F, L in configs:
            t_base, t_fused, max_diff = benchmark_fused_vs_baseline(B, F, L)
            writer.writerow([B, F, L, t_base * 1e3, t_fused * 1e3, max_diff])
    print(f"\nResults saved to {csv_path}") 