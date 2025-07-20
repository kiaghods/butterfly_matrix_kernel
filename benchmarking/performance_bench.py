"""
timing & GFLOP/s comparison for butterfly vs. dense matmul
"""

import math, os, csv, time
import sys
import torch
import triton
import triton.language as tl
import pandas as pd
import numpy as np
import logging

# Add parent directory to Python path to import modules from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import your implementations
try:
    from reference_impl import build_dense_from_stages, butterfly_mm_ref
    from triton_kernel import butterfly_mm_triton
    logger.info("Successfully imported butterfly implementations")
except ImportError as e:
    logger.error(f"Could not import implementations: {e}")
    raise

# --------------------------- benchmark config --------------------------
# Focus on power-of-2 L values that make sense for butterfly operations
# and reasonable B, F combinations for practical use cases
SHAPES = [
    # Small scale tests
    (4,   64,   256),    # e=8
    (4,   64,   512),    # e=9
    (4,   64,  1024),    # e=10
    (4,   64,  2048),    # e=11
    (4,   64,  4096),    # e=12
    
    # Different feature dimensions
    (4,  128,   512),    # e=9
    (4,  128,  1024),    # e=10
    (4,  256,  1024),    # e=10
    (4,  256,  2048),    # e=11
    
    # Different batch sizes
    (8,   64,   512),    # e=9
    (8,   64,  1024),    # e=10
    (8,   64,  2048),    # e=11
    (16,  64,   512),    # e=9
    (16,  64,  1024),    # e=10
    
    # Larger scales (if memory permits)
    (4,   64,  8192),    # e=13
    (8,   64,  4096),    # e=12
]
DTYPE = torch.float32
DEVICE = "cuda"
OUTPUT_DIR = "results/benchmarks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global storage for metrics
metrics_store = {}

# Helper functions for accuracy comparison
def max_diff(a, b):
    """Calculate maximum absolute difference between tensors"""
    if a is None or b is None: 
        return float('inf')
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor): 
        return float('inf')
    if a.shape != b.shape: 
        return float('inf')
    return (a - b).abs().max().item()

def avg_diff(a, b):
    """Calculate average absolute difference between tensors"""
    if a is None or b is None: 
        return float('inf')
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor): 
        return float('inf')
    if a.shape != b.shape: 
        return float('inf')
    return (a - b).abs().mean().item()

def relative_error(a, b):
    """Calculate relative error as percentage"""
    if a is None or b is None: 
        return float('inf')
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor): 
        return float('inf')
    if a.shape != b.shape: 
        return float('inf')
    
    epsilon = torch.finfo(a.dtype).eps if a.dtype.is_floating_point else 1e-6
    denominator = torch.max(a.abs(), b.abs())
    safe_denominator = torch.where(denominator < epsilon, epsilon, denominator)
    return ((a - b).abs() / safe_denominator).mean().item() * 100

def benchmark_butterfly_impl(B, F, L, provider, dtype=DTYPE, device=DEVICE):
    """
    Benchmark a single butterfly implementation with proper warmup
    Returns: (tflops, time_s, peak_mem_gb, max_diff_vs_ref, avg_rel_err)
    """
    e = int(math.log2(L))
    
    # Validate L is power of 2
    if 2**e != L:
        logger.warning(f"Skipping L={L} as it's not a power of 2")
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    
    # Generate test data
    torch.manual_seed(42)  # Fixed seed for reproducibility
    X = torch.randn(B, F, L, device=device, dtype=dtype)
    Wp = torch.randn(e, L, 2, device=device, dtype=dtype)
    
    # GPU warmup - run a few operations to ensure proper initialization
    logger.debug(f"Warming up GPU for shape (B={B}, F={F}, L={L})...")
    for _ in range(5):
        _ = torch.matmul(torch.randn(100, 100, device=device), torch.randn(100, 100, device=device))
        torch.cuda.synchronize()
    
    # Provider-specific warmup
    if provider == "triton":
        warmup_fn = lambda: butterfly_mm_triton(X, Wp)
    elif provider == "reference":
        warmup_fn = lambda: butterfly_mm_ref(X, Wp)
    elif provider == "dense":
        W_dense = build_dense_from_stages(Wp)
        warmup_fn = lambda: X @ W_dense
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    # Run provider-specific warmup
    logger.debug(f"Warming up {provider} implementation...")
    for _ in range(3):
        try:
            _ = warmup_fn()
            torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Warmup failed for {provider}: {e}")
            break
    
    # Accuracy check against reference
    max_diff_val = float('nan')
    avg_rel_err_val = float('nan')
    
    if provider != "reference":
        try:
            # Get reference result
            ref_result = butterfly_mm_ref(X.clone(), Wp.clone())
            
            # Get provider result
            if provider == "triton":
                provider_result = butterfly_mm_triton(X.clone(), Wp.clone())
            elif provider == "dense":
                W_dense = build_dense_from_stages(Wp.clone())
                provider_result = X.clone() @ W_dense
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            # Calculate accuracy metrics
            max_diff_val = max_diff(provider_result, ref_result)
            avg_rel_err_val = relative_error(provider_result, ref_result)
            
            logger.debug(f"Accuracy [{provider} vs reference | B={B}, F={F}, L={L}] "
                        f"Max Diff: {max_diff_val:.2e}, Rel Err: {avg_rel_err_val:.2f}%")
            
        except Exception as e:
            logger.error(f"Error in accuracy check for {provider}: {e}")
    
    # Clear any cached memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Define the function to benchmark
    if provider == "triton":
        bench_fn = lambda: butterfly_mm_triton(X, Wp)
    elif provider == "reference":
        bench_fn = lambda: butterfly_mm_ref(X, Wp)
    elif provider == "dense":
        W_dense = build_dense_from_stages(Wp)
        bench_fn = lambda: X @ W_dense
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    # Benchmark using Triton's robust timing with more aggressive warmup
    try:
        # Increased warmup iterations for more stable timing
        ms = triton.testing.do_bench(bench_fn, warmup=50, rep=100, quantiles=None)
        time_s = ms / 1000.0
    except Exception as e:
        logger.error(f"Error in timing {provider}: {e}")
        return float('nan'), float('nan'), float('nan'), max_diff_val, avg_rel_err_val
    
    # Memory usage
    torch.cuda.synchronize()
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    
    # FLOP calculation for butterfly matrix multiplication
    # Butterfly: B * F * L * log2(L) operations approximately
    # Dense equivalent: B * F * L * L operations
    if provider == "dense":
        total_flops = 2.0 * B * F * L * L  # Dense matrix multiply
    else:
        total_flops = 2.0 * B * F * L * e  # Butterfly multiply (log factor)
    
    tflops = total_flops * 1e-12 / time_s if time_s > 0 else float('nan')
    
    logger.debug(f"Benchmark [{provider} | B={B}, F={F}, L={L}] "
                f"Time: {time_s:.4f}s, Peak Mem: {peak_mem_gb:.3f}GB, TFLOP/s: {tflops:.2f}")
    
    return tflops, time_s, peak_mem_gb, max_diff_val, avg_rel_err_val

def validate_shapes(shapes):
    """Validate and filter shapes to only include power-of-2 L values"""
    valid_shapes = []
    for B, F, L in shapes:
        e = int(math.log2(L))
        if 2**e == L:
            valid_shapes.append((B, F, L))
            logger.debug(f"Valid shape: (B={B}, F={F}, L={L}, e={e})")
        else:
            logger.warning(f"Skipping invalid shape: (B={B}, F={F}, L={L}) - L must be power of 2")
    
    logger.info(f"Using {len(valid_shapes)} valid shapes out of {len(shapes)} total")
    return valid_shapes

def generate_benchmark_configs():
    """Generate Triton benchmark configurations with validated shapes"""
    # Validate shapes first
    valid_shapes = validate_shapes(SHAPES)
    
    providers = {
        "triton": ("Triton Kernel", ("red", "-")),
        "reference": ("Reference Implementation", ("blue", "--")),
        "dense": ("Dense Baseline", ("green", "-.")),
    }
    
    configs = []
    
    # Create a config for each valid shape
    for B, F, L in valid_shapes:
        configs.append(
            triton.testing.Benchmark(
                x_names=["L"],
                x_vals=[L],
                line_arg="provider",
                line_vals=list(providers.keys()),
                line_names=[p[0] for p in providers.values()],
                styles=[p[1] for p in providers.values()],
                ylabel="TFLOP/s",
                plot_name=f"butterfly-b{B}-f{F}",
                args={
                    "B": B,
                    "F": F,
                    "dtype": DTYPE,
                    "device": DEVICE
                },
            )
        )
    
    logger.info(f"Generated {len(configs)} benchmark configurations")
    return configs

@triton.testing.perf_report(generate_benchmark_configs())
def benchmark_all_butterfly_impls(B, F, L, provider, dtype=DTYPE, device=DEVICE):
    """Main benchmark function called by Triton's perf_report"""
    tflops, time_s, peak_mem_gb, max_diff_val, avg_rel_err_val = benchmark_butterfly_impl(
        B, F, L, provider, dtype, device
    )
    
    # Store additional metrics
    run_key = (provider, B, F, L)
    metrics_store[run_key] = {
        "Time (s)": time_s,
        "Peak Memory (GB)": peak_mem_gb,
        "TFLOP/s": tflops,
        "Max Abs Diff (vs Reference)": max_diff_val,
        "Avg Relative Error (%)": avg_rel_err_val,
    }
    
    return tflops

def save_detailed_results():
    """Save detailed results to CSV"""
    if not metrics_store:
        logger.warning("No metrics collected, skipping CSV save")
        return
    
    # Convert metrics store to DataFrame
    rows = []
    for (provider, B, F, L), metrics in metrics_store.items():
        row = {
            'Provider': provider,
            'B': B,
            'F': F,
            'L': L,
            'Log2(L)': int(math.log2(L)),
            **metrics
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    col_order = ['Provider', 'B', 'F', 'L', 'Log2(L)', 'Time (s)', 'Peak Memory (GB)', 
                 'TFLOP/s', 'Max Abs Diff (vs Reference)', 'Avg Relative Error (%)']
    df = df[col_order]
    
    # Sort by B, F, L, then provider
    df = df.sort_values(['B', 'F', 'L', 'Provider'])
    
    csv_path = os.path.join(OUTPUT_DIR, "butterfly_detailed_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Detailed results saved to: {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Group by shape and compare providers
    for (B, F, L) in sorted(set((row['B'], row['F'], row['L']) for row in rows)):
        shape_data = df[(df['B'] == B) & (df['F'] == F) & (df['L'] == L)]
        if len(shape_data) == 0:
            continue
            
        print(f"\nShape (B={B}, F={F}, L={L}):")
        print("-" * 60)
        
        for _, row in shape_data.iterrows():
            provider = row['Provider']
            tflops = row['TFLOP/s']
            time_s = row['Time (s)']
            mem_gb = row['Peak Memory (GB)']
            max_diff = row['Max Abs Diff (vs Reference)']
            rel_err = row['Avg Relative Error (%)']
            
            print(f"  {provider:20s}: {tflops:8.2f} TFLOP/s, {time_s*1000:8.1f} ms, "
                  f"{mem_gb:6.3f} GB")
            if provider != "reference" and not np.isnan(max_diff):
                print(f"                        Accuracy: max_diff={max_diff:.2e}, "
                      f"rel_err={rel_err:.2f}%")

def main():
    """Main benchmark execution with warmup"""
    logger.info("Starting butterfly matrix multiplication benchmarks...")
    
    # Validate shapes at startup
    valid_shapes = validate_shapes(SHAPES)
    if not valid_shapes:
        logger.error("No valid shapes found! All L values must be powers of 2.")
        return
    
    logger.info(f"Will benchmark {len(valid_shapes)} shapes with enhanced warmup")
    logger.info("Valid shapes: " + ", ".join([f"(B={B},F={F},L={L})" for B, F, L in valid_shapes[:5]]) + 
                ("..." if len(valid_shapes) > 5 else ""))
    
    # Global GPU warmup
    logger.info("Performing global GPU warmup...")
    torch.cuda.empty_cache()
    for _ in range(10):
        _ = torch.matmul(torch.randn(512, 512, device=DEVICE), torch.randn(512, 512, device=DEVICE))
        torch.cuda.synchronize()
    
    try:
        # Run the benchmarks (Triton will handle the execution)
        logger.info("Starting benchmark execution...")
        benchmark_all_butterfly_impls.run(save_path=None, print_data=True, show_plots=False)
        
        # Save our detailed results
        save_detailed_results()
        
    except Exception as e:
        logger.error(f"Error during benchmark execution: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("Benchmark execution completed!")

if __name__ == "__main__":
    torch.cuda.manual_seed(42)
    main()