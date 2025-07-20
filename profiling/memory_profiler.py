"""
memory profiling and analysis for butterfly matrix multiplication implementations
"""

import torch
import gc
import time
import logging
import math
import os
import sys
import csv
import pandas as pd
from collections import defaultdict

# Add parent directory to Python path to import modules from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your implementations
try:
    from triton_kernel import butterfly_mm_triton
    from reference_impl import butterfly_mm_ref, build_dense_from_stages
    IMPLEMENTATIONS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import implementations: {e}")
    IMPLEMENTATIONS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("butterfly_mem")

GB = 1024 ** 3
MB = 1024 ** 2

def gb(x): 
    return x / GB

def mb(x): 
    return x / MB

# --------------------------- Memory Analysis Helpers --------------------------
def print_mem(tag):
    log.info(f"{tag:<28} "
             f"allocated={gb(torch.cuda.memory_allocated()):5.2f} GB   "
             f"reserved={gb(torch.cuda.memory_reserved()):5.2f} GB")

def top_blocks(snapshot, k=8):
    """Get top k memory blocks from snapshot"""
    blocks = [b for seg in snapshot for b in seg["blocks"] if b["state"] == "active_allocated"]
    blocks.sort(key=lambda b: b["size"], reverse=True)
    return blocks[:k]

def segment_breakdown(snapshot):
    """
    Break down memory usage by segment type.
    
    Provides insights into how memory is organized:
    - cached: Memory cached by PyTorch for reuse
    - allocated: Memory currently in use
    - reserved: Memory reserved by PyTorch but not necessarily used
    """
    seg_bytes = defaultdict(int)
    for seg in snapshot:
        seg_bytes[seg["segment_type"]] += seg["active_size"]
    return {k: gb(v) for k, v in seg_bytes.items()}

def memory_diff(before, after):
    """Calculate memory difference between two snapshots"""
    return {
        "allocated_diff": gb(after["allocated"] - before["allocated"]),
        "reserved_diff": gb(after["reserved"] - before["reserved"])
    }

def get_memory_snapshot():
    """Get current memory state"""
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved()
    }

# --------------------------- Theoretical Memory Calculations --------------------------
def theoretical_memory_usage(B, F, L, dtype=torch.float16):
    """Calculate theoretical memory usage for butterfly operations"""
    e = int(math.log2(L))
    dtype_size = 2 if dtype == torch.float16 else 4  # bytes per element
    
    # Input tensors
    X_bytes = B * F * L * dtype_size  # Input tensor: (B, F, L)
    Wp_bytes = e * L * 2 * dtype_size  # Butterfly parameters: (e, L, 2)
    
    # Output tensor
    Y_bytes = B * F * L * dtype_size  # Output tensor: (B, F, L)
    
    # Intermediate storage (rough estimate for butterfly stages)
    # Each stage processes data and may need temporary storage
    # This is a conservative estimate for in-place operations
    intermediate_bytes = B * F * L * dtype_size
    
    total_theoretical = X_bytes + Wp_bytes + Y_bytes + intermediate_bytes
    
    return {
        "input_X_mb": mb(X_bytes),
        "weights_Wp_mb": mb(Wp_bytes),
        "output_Y_mb": mb(Y_bytes),
        "intermediate_mb": mb(intermediate_bytes),
        "total_theoretical_mb": mb(total_theoretical),
        "stages": e
    }

# --------------------------- Main Memory Profiler --------------------------
def profile_butterfly_implementation(impl_fn, impl_name, B=4, F=64, L=1024, dtype=torch.float16):
    """Profile a single butterfly implementation"""
    if not IMPLEMENTATIONS_AVAILABLE:
        log.error("Implementations not available, skipping profiling")
        return None
        
    e = int(math.log2(L))
    if 2**e != L:
        log.error(f"L={L} is not a power of 2, skipping")
        return None
    
    # Clear memory before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    log.info(f"\n--- Profiling {impl_name} (B={B}, F={F}, L={L}, e={e}) ---")
    print_mem("startup")
    
    # Get theoretical estimates
    theory = theoretical_memory_usage(B, F, L, dtype)
    log.info(f"theoretical memory usage  = {theory['total_theoretical_mb']:6.1f} MB "
             f"(X:{theory['input_X_mb']:.1f} + Wp:{theory['weights_Wp_mb']:.1f} + "
             f"Y:{theory['output_Y_mb']:.1f} + temp:{theory['intermediate_mb']:.1f})")
    
    # Create input tensors
    mem_before_tensors = get_memory_snapshot()
    
    X = torch.randn(B, F, L, dtype=dtype, device="cuda")
    Wp = torch.randn(e, L, 2, dtype=dtype, device="cuda")
    
    # For dense baseline, pre-build the dense matrix
    W_dense = None
    if impl_name == "Dense":
        W_dense = build_dense_from_stages(Wp)
    
    mem_after_tensors = get_memory_snapshot()
    tensor_mem_diff = memory_diff(mem_before_tensors, mem_after_tensors)
    
    print_mem("after tensor setup")
    log.info(f"tensor allocation overhead = {tensor_mem_diff['allocated_diff']:5.2f} GB allocated, "
             f"{tensor_mem_diff['reserved_diff']:5.2f} GB reserved")
    
    # Warmup run to initialize kernels and allocate any persistent memory
    log.info("performing warmup...")
    try:
        with torch.no_grad():
            if impl_name == "Triton":
                _ = butterfly_mm_triton(X[:1], Wp)  # Small warmup
            elif impl_name == "Reference":
                _ = butterfly_mm_ref(X[:1], Wp)
            elif impl_name == "Dense":
                _ = X[:1] @ W_dense
            else:
                raise ValueError(f"Unknown implementation: {impl_name}")
    except Exception as e:
        log.error(f"Warmup failed for {impl_name}: {e}")
        return None
    
    torch.cuda.synchronize()
    print_mem("after warmup")
    
    # Reset memory stats for the actual profiling run
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    # Actual profiling run
    mem_before_run = get_memory_snapshot()
    start_time = time.time()
    
    try:
        with torch.no_grad():
            if impl_name == "Triton":
                out = butterfly_mm_triton(X, Wp)
            elif impl_name == "Reference":
                out = butterfly_mm_ref(X, Wp)
            elif impl_name == "Dense":
                out = X @ W_dense
            else:
                raise ValueError(f"Unknown implementation: {impl_name}")
    except Exception as e:
        log.error(f"Execution failed for {impl_name}: {e}")
        return None
    
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    mem_after_run = get_memory_snapshot()
    run_mem_diff = memory_diff(mem_before_run, mem_after_run)
    
    # Get peak memory stats
    alloc_peak = gb(torch.cuda.max_memory_allocated())
    reserved_peak = gb(torch.cuda.max_memory_reserved())
    alloc_after = gb(torch.cuda.memory_allocated())
    reserved_after = gb(torch.cuda.memory_reserved())
    
    print_mem("after execution")
    log.info(f"execution memory overhead = {run_mem_diff['allocated_diff']:5.2f} GB allocated, "
             f"{run_mem_diff['reserved_diff']:5.2f} GB reserved")
    log.info(f"peak allocated            = {alloc_peak:5.2f} GB")
    log.info(f"peak reserved             = {reserved_peak:5.2f} GB")
    log.info(f"execution wall-time       = {elapsed_time:6.3f} s")
    
    # Calculate efficiency metrics
    actual_vs_theoretical = (alloc_peak * GB) / (theory['total_theoretical_mb'] * MB)
    
    log.info(f"memory efficiency         = {actual_vs_theoretical:5.2f}x theoretical")
    
    # Clean up
    del X, Wp, out
    if W_dense is not None:
        del W_dense
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        "impl_name": impl_name,
        "shape": (B, F, L),
        "stages": e,
        "alloc_after": alloc_after,
        "reserved_after": reserved_after,
        "alloc_peak": alloc_peak,
        "reserved_peak": reserved_peak,
        "execution_time": elapsed_time,
        "tensor_alloc_overhead": tensor_mem_diff['allocated_diff'],
        "run_alloc_overhead": run_mem_diff['allocated_diff'],
        "theoretical_mb": theory['total_theoretical_mb'],
        "memory_efficiency": actual_vs_theoretical
    }

def profile_butterfly_scaling(impl_name, base_B=4, base_F=64, scale_factors=[1, 2, 4, 8]):
    """Profile how implementation scales with different parameters"""
    log.info(f"\n--- Scaling Analysis for {impl_name} ---")
    
    results = []
    
    # Scale sequence length (most important for butterfly)
    log.info("Scaling sequence length (L):")
    base_L = 1024
    for factor in scale_factors:
        L = base_L * factor
        if L <= 16384:  # Reasonable limit
            result = profile_butterfly_implementation(
                get_impl_fn(impl_name), impl_name, 
                B=base_B, F=base_F, L=L
            )
            if result:
                results.append(result)
    
    # Scale batch size
    log.info("\nScaling batch size (B):")
    for factor in scale_factors:
        B = base_B * factor
        if B <= 32:  # Reasonable limit
            result = profile_butterfly_implementation(
                get_impl_fn(impl_name), impl_name,
                B=B, F=base_F, L=base_L
            )
            if result:
                results.append(result)
    
    return results

def get_impl_fn(impl_name):
    """Get implementation function by name"""
    if impl_name == "Triton":
        return butterfly_mm_triton
    elif impl_name == "Reference":
        return butterfly_mm_ref
    elif impl_name == "Dense":
        return lambda X, Wp: X @ build_dense_from_stages(Wp)
    else:
        raise ValueError(f"Unknown implementation: {impl_name}")

def compare_implementations(B=4, F=64, L=1024):
    """Compare all available implementations"""
    implementations = ["Triton", "Reference", "Dense"]
    results = {}
    
    for impl_name in implementations:
        try:
            result = profile_butterfly_implementation(
                get_impl_fn(impl_name), impl_name, B, F, L
            )
            if result:
                results[impl_name] = result
        except Exception as e:
            log.error(f"Failed to profile {impl_name}: {e}")
    
    return results

def save_results_to_csv(all_results, scaling_results=None):
    """Save profiling results to CSV files"""
    # Ensure results/profiles directory exists
    os.makedirs("results/profiles", exist_ok=True)
    
    # Convert main results to DataFrame
    rows = []
    for seq_len, implementations in all_results.items():
        for impl_name, result in implementations.items():
            if result:
                row = {
                    'sequence_length': seq_len,
                    'implementation': impl_name,
                    'batch_size': result['shape'][0],
                    'feature_dim': result['shape'][1],
                    'stages': result['stages'],
                    'alloc_after_gb': result['alloc_after'],
                    'reserved_after_gb': result['reserved_after'],
                    'alloc_peak_gb': result['alloc_peak'],
                    'reserved_peak_gb': result['reserved_peak'],
                    'execution_time_s': result['execution_time'],
                    'tensor_alloc_overhead_gb': result['tensor_alloc_overhead'],
                    'run_alloc_overhead_gb': result['run_alloc_overhead'],
                    'theoretical_mb': result['theoretical_mb'],
                    'memory_efficiency': result['memory_efficiency']
                }
                rows.append(row)
    
    # Save main results
    if rows:
        df_main = pd.DataFrame(rows)
        main_csv_path = os.path.join("results/profiles", "butterfly_memory_profiling.csv")
        df_main.to_csv(main_csv_path, index=False)
        log.info(f"Main results saved to: {main_csv_path}")
    
    # Save scaling results if available
    if scaling_results:
        scaling_rows = []
        for result in scaling_results:
            row = {
                'implementation': result['impl_name'],
                'batch_size': result['shape'][0],
                'feature_dim': result['shape'][1],
                'sequence_length': result['shape'][2],
                'stages': result['stages'],
                'alloc_peak_gb': result['alloc_peak'],
                'reserved_peak_gb': result['reserved_peak'],
                'execution_time_s': result['execution_time'],
                'memory_efficiency': result['memory_efficiency']
            }
            scaling_rows.append(row)
        
        if scaling_rows:
            df_scaling = pd.DataFrame(scaling_rows)
            scaling_csv_path = os.path.join("results/profiles", "butterfly_scaling_analysis.csv")
            df_scaling.to_csv(scaling_csv_path, index=False)
            log.info(f"Scaling results saved to: {scaling_csv_path}")

# --------------------------- Main Execution ---------------------------
if __name__ == "__main__":
    if not IMPLEMENTATIONS_AVAILABLE:
        log.error("Required implementations not available. Please ensure imports work.")
        exit(1)
    
    # Test different sequence lengths
    seq_lens = [512, 1024, 2048, 4096, 8192]
    batch_sizes = [4, 8, 16]
    feature_dims = [64, 128, 256]
    
    all_results = {}
    
    # Main comparison across sequence lengths
    log.info("="*80)
    log.info("BUTTERFLY MATRIX MEMORY PROFILING")
    log.info("="*80)
    
    for seq_len in seq_lens:
        log.info(f"\nTesting sequence length: {seq_len}")
        all_results[seq_len] = compare_implementations(B=4, F=64, L=seq_len)
    
    # Summary table
    log.info("\n" + "="*100)
    log.info("MEMORY USAGE SUMMARY")
    log.info("="*100)
    log.info(f"{'Seq Len':>8} | {'Implementation':<12} | {'Peak Alloc (GB)':>15} | {'Peak Reserv (GB)':>16} | {'Time (s)':>10} | {'Mem Eff':>8}")
    log.info("-"*100)
    
    for seq_len in seq_lens:
        if seq_len in all_results:
            for impl_name, result in all_results[seq_len].items():
                if result:
                    log.info(f"{seq_len:8d} | {impl_name:<12} | {result['alloc_peak']:15.3f} | "
                            f"{result['reserved_peak']:16.3f} | {result['execution_time']:10.3f} | "
                            f"{result['memory_efficiency']:8.2f}x")
    
    # Detailed scaling analysis for Triton implementation
    scaling_results = None
    if any("Triton" in results for results in all_results.values()):
        log.info("\n" + "="*80)
        log.info("DETAILED SCALING ANALYSIS (Triton Implementation)")
        log.info("="*80)
        scaling_results = profile_butterfly_scaling("Triton")
    
    # Save results to CSV
    save_results_to_csv(all_results, scaling_results)
    
    log.info("\nMemory profiling completed!")