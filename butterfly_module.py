"""
PyTorch nn.Module wrapper for the Triton butterfly matrix multiplication kernel
"""
import torch
import torch.nn as nn
import math
from triton_kernel import butterfly_mm_triton 

class ButterflyLayer(nn.Module):
    def __init__(self, F, L, rightmost=True):
        super().__init__()
        self.F = F
        self.L = L
        self.rightmost = rightmost
        e = int(math.log2(L))
        # Butterfly parameters: one (L, 2) matrix per stage
        self.W_par = nn.ParameterList([
            nn.Parameter(torch.randn(L, 2)) for _ in range(e)
        ])

    def forward(self, X):
        # X: (B, F, L)
        W_par = [w for w in self.W_par]
        return butterfly_mm_triton(X, W_par, rightmost=self.rightmost)