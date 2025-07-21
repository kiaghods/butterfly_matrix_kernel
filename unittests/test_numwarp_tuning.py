"""
unit tests for the numwarp-tuning butterfly kernel (optimization/numwarp_tuning.py).
"""
import torch
import sys
import os
import math
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimization.numwarp_tuning import butterfly_mm_numwarp_tuning
from testing.correctness_harness import check_case
from reference_impl import butterfly_mm_ref

SHAPES = [
    (1, 1, 16),
    (2, 8, 32),
    (4, 64, 256),
    (8, 128, 512),
    (16, 512, 1024),
    (16, 2048, 2048),
]
DTYPES = [torch.float32]
NUM_WARPS = [4, 8, 16]

@pytest.mark.parametrize("B,F,L", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_warps", NUM_WARPS)
def test_numwarp_tuning_correctness(B, F, L, dtype, num_warps):
    passed, err, t_ref, t_kern = check_case(
        B, F, L,
        ref_fn=butterfly_mm_ref,
        kern_fn=lambda X, W_par: butterfly_mm_numwarp_tuning(X, W_par, BLOCK_BF=64, num_warps=num_warps),
        dtype=dtype,
        quiet=True
    )
    print(f"[pytest] B={B} F={F} L={L} dtype={dtype} num_warps={num_warps}  err={err:.2e}  "
          f"t_ref={t_ref:.2f} ms  t_kern={t_kern:.2f} ms")
    assert passed