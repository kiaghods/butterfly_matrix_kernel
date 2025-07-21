#!/usr/bin/env python3
"""
minimal radix-2 FFT kernel (complex, power-of-two length).

- Uses a custom Triton kernel for each FFT stage, performing the butterfly and complex multiply on the GPU.
- Input: (B, N, 2) float32 tensor, where the last dim is (real, imag).
- Output matches torch.fft.fft (within numerical precision).

Usage:
    python triton_fft_kernel.py            # runs a quick self-test
    python triton_fft_kernel.py --bench    # benchmark vs torch.fft
"""

import math, argparse, time
import torch
import triton
import triton.language as tl

# --- Bit-reversal permutation helper ---
def _bit_reverse_permutation(N: int, device) -> torch.Tensor:
    """Return LongTensor of indices that bit-reverse numbers in [0, N)."""
    bits = N.bit_length() - 1
    idx = torch.arange(N, device=device, dtype=torch.long)
    rev = torch.zeros_like(idx)
    for i in range(bits):
        rev |= ((idx >> i) & 1) << (bits - 1 - i)
    return rev

# --- Triton kernel for a single FFT butterfly stage (complex) ---
@triton.jit
def fft_butterfly_stage_kernel(
    x_ptr,                # pointer to (N, 2) tensor (real/imag pairs)
    twiddle_ptr,          # pointer to (N//2, 2) tensor (real/imag twiddles)
    stage: tl.constexpr,  # current stage (0-based)
    N: tl.constexpr,      # FFT length
    BLOCK: tl.constexpr   # block size
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N // 2

    # Butterfly span for this stage
    span = 1 << (stage + 1)
    half = span // 2

    # Indices for the butterfly pairs
    i0 = (offs // half) * span + (offs % half)
    i1 = i0 + half

    # Load real/imag pairs
    x0r = tl.load(x_ptr + i0 * 2 + 0, mask=mask)
    x0i = tl.load(x_ptr + i0 * 2 + 1, mask=mask)
    x1r = tl.load(x_ptr + i1 * 2 + 0, mask=mask)
    x1i = tl.load(x_ptr + i1 * 2 + 1, mask=mask)

    # Load twiddle factors (real/imag)
    wr = tl.load(twiddle_ptr + offs * 2 + 0, mask=mask)
    wi = tl.load(twiddle_ptr + offs * 2 + 1, mask=mask)

    # Complex multiply x1 by twiddle: (x1r + j*x1i) * (wr + j*wi)
    t1r = x1r * wr - x1i * wi
    t1i = x1r * wi + x1i * wr

    # Butterfly
    y0r = x0r + t1r
    y0i = x0i + t1i
    y1r = x0r - t1r
    y1i = x0i - t1i

    # Store results
    tl.store(x_ptr + i0 * 2 + 0, y0r, mask=mask)
    tl.store(x_ptr + i0 * 2 + 1, y0i, mask=mask)
    tl.store(x_ptr + i1 * 2 + 0, y1r, mask=mask)
    tl.store(x_ptr + i1 * 2 + 1, y1i, mask=mask)

# --- Main FFT function using the above kernel ---
def fft_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Compute 1D complex FFT along last dimension using Triton.
    Args:
        x: (B, N, 2) float32 tensor; real/imag packed, N power-of-two
    Returns:
        (B, N, 2) tensor (real, imag)
    """
    assert x.dtype == torch.float32 and x.shape[-1] == 2
    B, N, _ = x.shape
    device = x.device
    stages = int(math.log2(N))

    # Bit-reverse the input for in-place Cooley-Tukey
    rev = _bit_reverse_permutation(N, device)
    x = x.index_select(1, rev).contiguous()

    # Flatten for kernel: (B, N, 2) -> (B, N, 2)
    x_flat = x.view(B, N, 2)

    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(N // 2, meta['BLOCK']),)

    for stage in range(stages):
        span = 1 << (stage + 1)
        half = span // 2
        # Twiddle factors for this stage (N//2, 2)
        k = torch.arange(N // 2, device=device)
        m = span
        angle = -2 * math.pi * (k % half) / m
        wr = torch.cos(angle)
        wi = torch.sin(angle)
        twiddles = torch.stack([wr, wi], dim=1).contiguous().view(-1)
        # Launch kernel for each batch
        for b in range(B):
            # Pass a contiguous (N, 2) tensor for each batch
            fft_butterfly_stage_kernel[grid] (
                x_flat[b],
                twiddles,
                stage,
                N,
                BLOCK
            )
    # Reshape back to (B, N, 2)
    return x_flat

# --- Quick self-test ---
def quick_test():
    """Run a quick correctness test against torch.fft.fft."""
    torch.manual_seed(0)
    B, N = 4, 1024
    x = torch.randn(B, N, 2, device="cuda", dtype=torch.float32)
    y = fft_triton(x.clone())
    y_ref = torch.view_as_real(torch.fft.fft(torch.view_as_complex(x)))  # (B, N, 2)
    err = (y - y_ref).abs().max().item()
    print(f"max |err| = {err:.3e}")
    assert err < 1e-3, "FFT mismatch – something is wrong!"

# --- Benchmark ---
def bench():
    """Benchmark Triton FFT vs torch.fft.fft for a few sizes."""
    shapes = [256, 1024, 4096]
    for N in shapes:
        x = torch.randn(4, N, 2, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        t0 = time.time(); y = fft_triton(x); torch.cuda.synchronize(); t1 = time.time()
        t_triton = t1 - t0

        t0 = time.time(); y2 = torch.view_as_real(torch.fft.fft(torch.view_as_complex(x))); torch.cuda.synchronize(); t1 = time.time()
        t_torch = t1 - t0

        err = (y - y2).abs().max().item()
        print(f"N={N:5d}  triton={t_triton*1e3:6.2f} ms   torch={t_torch*1e3:6.2f} ms   err={err:.2e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    args = parser.parse_args()

    if args.bench:
        bench()
    else:
        quick_test()
