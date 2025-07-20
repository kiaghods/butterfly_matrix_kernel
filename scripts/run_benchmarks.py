"""
wrapper script for running performance benchmarks
"""

import sys
import os
import subprocess
import argparse

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_performance_benchmarks():
    """Run performance benchmarks."""
    print("Running performance benchmarks...")
    try:
        subprocess.run([sys.executable, "benchmarking/performance_bench.py"], check=True)
        print("Performance benchmarks completed")
    except subprocess.CalledProcessError as e:
        print(f"Performance benchmarks failed: {e}")

def main():
    """Run benchmarking tools."""
    parser = argparse.ArgumentParser(description="Run butterfly matrix benchmarking tools")
    parser.add_argument("--performance", action="store_true", help="Run performance benchmarks")
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs("results/benchmarks", exist_ok=True)
    
    # Run performance benchmarks by default
    run_performance_benchmarks()
    
    print("\nBenchmarking completed!")

if __name__ == "__main__":
    main() 