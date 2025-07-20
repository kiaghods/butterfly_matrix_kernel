"""
independent tester for butterfly matrix multiplication
"""

import math, time, itertools
import torch


# ------------------------------------------------------------------------
# 1. random-instance generator
# ------------------------------------------------------------------------
def rand_case(B, F, L, *, dtype=torch.float32, device="cuda"):
    """
    Generate a random test case for butterfly matrix multiplication.
    
    Creates random input tensor X and butterfly parameters W_par with the
    specified dimensions. The input L must be a power of 2 for the butterfly
    algorithm to work correctly.
    
    Args:
        B: Batch size
        F: Feature dimension
        L: Sequence length (must be a power of 2)
        dtype: Data type for the tensors
        device: Device to place tensors on
    
    Returns:
        X: Random input tensor of shape (B, F, L)
        W_par: Random butterfly parameters of shape (e, L, 2) where e = log2(L)
    """
    e = int(math.log2(L))
    X = torch.randn(B, F, L, device=device, dtype=dtype)
    W_par = torch.randn(e, L, 2, device=device, dtype=dtype)
    return X, W_par


# ------------------------------------------------------------------------
# 2. Single test case validation
# ------------------------------------------------------------------------
def check_case(B, F, L, *,
               ref_fn,
               kern_fn,   
               dtype=torch.float32,
               atol=1e-5, rtol=1e-5,
               quiet=False):
    """
    Validate a single test case by comparing reference and kernel implementations.
    
    This function generates a random test case and compares the outputs of
    the reference and kernel implementations. It measures both accuracy (error)
    and performance (execution time) for the given test case.
    
    Args:
        B, F, L: Test case dimensions (batch, features, sequence length)
        ref_fn: Reference implementation function
        kern_fn: Kernel implementation function to test
        dtype: Data type for the test
        atol: Absolute tolerance for numerical comparison
        rtol: Relative tolerance for numerical comparison
        quiet: If True, suppress output unless test fails
    
    Returns:
        passed: Boolean indicating if the test passed
        err: Maximum absolute error between implementations
        t_ref: Reference implementation execution time (ms)
        t_kern: Kernel implementation execution time (ms)
    """

    # Generate random test case
    X, W_par = rand_case(B, F, L, dtype=dtype)
    torch.cuda.synchronize()

    # reference
    t0 = time.perf_counter()
    Y_ref = ref_fn(X, W_par)
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) * 1e3   # Convert to milliseconds

    # kernel
    t0 = time.perf_counter()
    Y_kern = kern_fn(X, W_par)
    torch.cuda.synchronize()
    t_kern = (time.perf_counter() - t0) * 1e3  # Convert to milliseconds

    err   = (Y_ref - Y_kern).abs().max().item()
    scale = Y_ref.abs().max().item()
    passed = err < atol + rtol * scale

    if not quiet:
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}]  B={B:2d} F={F:4d} L={L:4d}  "
              f"err={err:.2e}  t_ref={t_ref:6.2f} ms  t_kern={t_kern:6.2f} ms")

    return passed, err, t_ref, t_kern


# ------------------------------------------------------------------------
# 3. Batch testing runner
# ------------------------------------------------------------------------
def run_suite(ref_fn,
              kern_fn,
              shapes=None,
              dtypes=(torch.float32,),
              atol=1e-5, rtol=1e-5):
    """
    Run a test suite across multiple test cases.
    
    This function tests the kernel implementation against the reference
    implementation across a variety of input shapes and data types.
    It provides detailed reporting of any failures and summary statistics.
    
    Args:
        ref_fn: Reference implementation function
        kern_fn: Kernel implementation function to test
        shapes: List of (B, F, L) tuples to test. If None, uses default test cases
        dtypes: List of data types to test
        atol: Absolute tolerance for numerical comparison
        rtol: Relative tolerance for numerical comparison
    
    Raises:
        AssertionError: If any test cases fail
    """

    # Default test cases covering various scenarios
    if shapes is None:
        shapes = [
            (1,   1,    16),
            (2,   8,    32),
            (4,  64,   256),
            (8, 128,   512),
            (16, 512, 1024),
            (16, 2048, 2048),
        ]

    failures = []

    for dtype, (B, F, L) in itertools.product(dtypes, shapes):
        passed, err, *_ = check_case(
            B, F, L,
            ref_fn=ref_fn,
            kern_fn=kern_fn,
            dtype=dtype,
            atol=atol,
            rtol=rtol,
            quiet=False,
        )
        if not passed:
            failures.append((B, F, L, dtype, err))

    # summary
    if not failures:
        print("\n All cases passed.")
    else:
        print(f"\n {len(failures)} case(s) failed:")
        for B, F, L, dt, err in failures:
            print(f"   • B={B} F={F} L={L} dtype={dt}  |err|={err:.2e}")
        raise AssertionError("Some test cases failed; see list above.")


# ------------------------------------------------------------------------
# 4. Self-test when run as script
# ------------------------------------------------------------------------
if __name__ == "__main__":
    from reference_impl import butterfly_mm_ref  
    from triton_kernel import butterfly_mm_triton

    run_suite(butterfly_mm_ref, butterfly_mm_triton)