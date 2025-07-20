"""
nsight systems profiler for butterfly matrix multiplication
"""

import torch, math
import sys
import os

# Add parent directory to Python path to import modules from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triton_kernel import butterfly_mm_triton
import time

# Test configuration
B, F, L = 16, 2048, 2048
e = int(math.log2(L))

# Generate test data
X = torch.randn(B, F, L, device='cuda', dtype=torch.float16)
W = torch.randn(e, L, 2, device='cuda', dtype=torch.float16)

# Warmup run
print("Warming up...")
_ = butterfly_mm_triton(X, W)
torch.cuda.synchronize()

# Simple timing
print("Running timing test...")
start = time.time()
result = butterfly_mm_triton(X, W)
torch.cuda.synchronize()
end = time.time()
print(f"Execution time: {(end-start)*1000:.2f} ms")

print("\nTo profile with Nsight Systems, run:")
print("nsys profile -o butterfly_profile python new_profiler.py")
print("Then view the .qdrep file with: nsys-ui butterfly_profile.qdrep")