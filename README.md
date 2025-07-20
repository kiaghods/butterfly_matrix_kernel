# Butterfly Matrix Multiplication Kernel

A high-performance implementation of butterfly matrix multiplication using Triton GPU kernels, with comprehensive testing, profiling, and benchmarking tools.

## Overview

This project implements butterfly matrix multiplication, which decomposes matrix multiplication into a series of structured operations. The implementation includes:

- **Triton GPU kernel** for high-performance computation
- **Reference implementation** for correctness validation
- **Comprehensive testing framework** with both pytest and standalone testing
- **Profiling tools** for performance analysis and optimization
- **Benchmarking suite** for performance comparison

## Project Structure

```
butterfly_matrix_kernel/
├── triton_kernel.py              # Main Triton GPU kernel implementation
├── reference_impl.py             # Reference torch implementation for validation
├── unittests/                    # Pytest-based unit tests
│   ├── __init__.py
│   └── test_correctness.py       # Unit tests using correctness_harness
├── testing/                      # Standalone testing framework
│   └── correctness_harness.py    # Comprehensive test suite
├── profiling/                    # Performance profiling tools
│   ├── torch_profiler.py         # PyTorch profiler with NVTX markers
│   ├── memory_profiler.py        # Memory usage profiling
│   └── visualization.py          # SQLite visualization for Nsight Systems
├── benchmarking/                 
│   └── performance_bench.py      # Performance comparison benchmarks
├── results/                      
│   ├── profiles/                 # Profiling results
│   ├── benchmarks/               # Benchmark results
│   └── plots/                    # Generated plots
└── scripts/                      # wrapper scripts
    ├── run_profiling.py          # Run all profiling tools
    └── run_benchmarks.py         # Run performance benchmarks
```

## Quick Start

### Prerequisites

- Python 3.8+
- uv (Python package manager) - [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
- NVIDIA GPU with compute capability 7.0+

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd butterfly_matrix_kernel

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate

uv pip install -r requirements.txt
```


## Testing

### Unit Testing (pytest) - Correctness & Numerical Precision

Run the pytest-based unit tests:

```bash
# Run all tests
pytest unittests/

# Run specific test file
pytest unittests/test_correctness.py

# Run with verbose output
pytest unittests/test_correctness.py -v
```

The pytest tests use the same `check_case` function from `correctness_harness.py` and tests various input configurations.

## Profiling

### PyTorch Profiler

Profile the implementation using PyTorch's built-in profiler:

```bash
python profiling/torch_profiler.py
```

This generates:
- Profiler summary with operator statistics
- NVTX markers for kernel identification

### Memory Profiling

Profile memory usage patterns:

```bash
python profiling/memory_profiler.py
```

This tracks:
- GPU memory allocation/deallocation
- Peak memory usage
- Memory efficiency metrics

### Nsight Systems Profiling

For detailed kernel-level profiling:

```bash
# Profile with Nsight Systems
nsys profile -o results/profiles/butterfly_profile python profiling/torch_profiler.py

# Export to SQLite for analysis
nsys export results/profiles/butterfly_profile.qdrep --type sqlite results/profiles/butterfly_profile.sqlite

# Visualize the results
python profiling/visualization.py results/profiles/butterfly_profile.sqlite
```

### Convenience Scripts

Use the wrapper scripts for easy profiling:

```bash
# Run all profiling tools
python scripts/run_profiling.py

# Run specific profiling tool
python scripts/run_profiling.py --tool torch
python scripts/run_profiling.py --tool memory
python scripts/run_profiling.py --tool nsight
```

## Benchmarking

### Performance Benchmarks

Run performance benchmarks to compare implementations:

```bash
python benchmarking/performance_bench.py
```

This compares:
- Triton kernel vs reference implementation vs dense matmul
- Performance (runtime, memory) across different input sizes

### Convenience Script

```bash
python scripts/run_benchmarks.py
```

## Results Analysis

### Profiling Results

Profiling results are stored in `results/profiles/`:
- SQLite databases (`.sqlite`) for Nsight Systems analysis
- Memory profiling reports

### Benchmark Results

Benchmark results are stored in `results/benchmarks/`:
- Performance comparison data
- Timing statistics
- Memory usage reports

### Visualization

The `profiling/visualization.py` script provides:
- Kernel duration distribution plots
- Timeline visualization
- Performance tier analysis

