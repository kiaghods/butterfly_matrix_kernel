"""
unit tests for butterfly matrix multiplication correctness
"""

import math, pytest, torch
from correctness_harness import check_case
from reference_impl import butterfly_mm_ref
from triton_kernel import butterfly_mm_triton

SHAPES = [
    (1, 1, 16),
    (2, 8, 32),
    (4, 64, 256),
    (8, 128, 512),
    (16, 512, 1024),
    (16, 2048, 2048),
]

DTYPES = [torch.float32]

@pytest.mark.parametrize("B,F,L", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_butterfly_kernel(B, F, L, dtype):
    """
    Test butterfly kernel correctness for a specific input configuration.
        
    Args:
        B: Batch size
        F: Feature dimension  
        L: Sequence length (must be a power of 2)
        dtype: Data type for the test tensors
    
    Raises:
        AssertionError: If the kernel output differs significantly from reference
    """
    passed, err, t_ref, t_kern = check_case(
        B, F, L,
        ref_fn=butterfly_mm_ref,
        kern_fn=butterfly_mm_triton,
        dtype=dtype,
        quiet=True
    )
    # force a print even if it passed
    print(f"[pytest] B={B} F={F} L={L} dtype={dtype}  err={err:.2e}  "
          f"t_ref={t_ref:.2f} ms  t_kern={t_kern:.2f} ms")
    assert passed