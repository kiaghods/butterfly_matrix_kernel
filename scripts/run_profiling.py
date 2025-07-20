"""
wrapper script for running all profiling tools
"""

import sys
import os
import subprocess
import argparse

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_torch_profiler():
    """Run PyTorch profiler with TensorBoard output."""
    print("Running PyTorch profiler...")
    try:
        subprocess.run([sys.executable, "profiling/torch_profiler.py"], check=True)
        print("PyTorch profiler completed")
    except subprocess.CalledProcessError as e:
        print(f"PyTorch profiler failed: {e}")

def run_memory_profiler():
    """Run memory profiling."""
    print("Running memory profiler...")
    try:
        subprocess.run([sys.executable, "profiling/memory_profiler.py"], check=True)
        print("Memory profiler completed")
    except subprocess.CalledProcessError as e:
        print(f"Memory profiler failed: {e}")

def run_nsys_profiler():
    """Run Nsight Systems profiler."""
    print("Running Nsight Systems profiler...")
    try:
        # Run Nsight Systems profiling
        subprocess.run(["nsys", "profile", "-o", "results/profiles/butterfly_profile", 
                       sys.executable, "profiling/torch_profiler.py"], check=True)
        
        # Export to SQLite for analysis
        print("Exporting to SQLite...")
        subprocess.run(["nsys", "export", "results/profiles/butterfly_profile.nsys-rep", 
                       "--type", "sqlite", "-o", "results/profiles/butterfly_profile.sqlite"], check=True)
        
        print("Nsight Systems profiler completed")
    except subprocess.CalledProcessError as e:
        print(f"Nsight Systems profiler failed: {e}")

def run_visualization():
    """Run visualization of profiling results."""
    print("Running visualization...")
    try:
        subprocess.run([sys.executable, "profiling/visualization.py"], check=True)
        print("Visualization completed")
    except subprocess.CalledProcessError as e:
        print(f"Visualization failed: {e}")

def main():
    """Run profiling tools based on command line arguments."""
    parser = argparse.ArgumentParser(description="Run butterfly matrix profiling tools")
    parser.add_argument("--torch", action="store_true", help="Run PyTorch profiler")
    parser.add_argument("--memory", action="store_true", help="Run memory profiler")
    parser.add_argument("--nsys", action="store_true", help="Run Nsight Systems profiler")
    parser.add_argument("--viz", action="store_true", help="Run visualization")
    parser.add_argument("--all", action="store_true", help="Run all profiling tools")
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs("results/profiles", exist_ok=True)
    
    if args.all or not any([args.torch, args.memory, args.nsys, args.viz]):
        # Run all by default
        run_torch_profiler()
        run_memory_profiler()
        run_nsys_profiler()
        run_visualization()
    else:
        if args.torch:
            run_torch_profiler()
        if args.memory:
            run_memory_profiler()
        if args.nsys:
            run_nsys_profiler()
        if args.viz:
            run_visualization()
    
    print("\n🎉 Profiling completed!")

if __name__ == "__main__":
    main() 