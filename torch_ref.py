import math, torch

def build_dense_from_stages(W_par, rightmost = True):
    """
    Convert the (e, L, 2) stage parameters into a full LxL matrix.
    This is O(L^2) and intended only for correctness checks    
    """    
    e, L, _ = W_par.shape
    identity = torch.eye(L, dtype=W_par.dtype, device=W_par.device)
    stages = range(e) if rightmost else reversed(range(e))

    W_dense = identity

    for i in stages: # construct and multiply each B_i
        stride = 1 << i # 2^i
        B_i = identity.clone()

        for base in range(0, L, 2*stride): # fill in each 2xstride block of B_i for fixed i
            idx      = torch.arange(stride, device=W_par.device)   # [0 … stride-1]
            rows0    = base + idx # j0 rows
            rows1    = base + stride + idx # j1 rows

            a0a1 = W_par[i, rows0] # shape (stride, 2)
            b0b1 = W_par[i, rows1]

            # row-by-row assignments - assignment and not broadcasting 
            B_i[rows0, rows0] = a0a1[:, 0] # top-left  diagonals
            B_i[rows0, rows1] = a0a1[:, 1] # top-right diagonals
            B_i[rows1, rows0] = b0b1[:, 0] # bottom-left diagonals
            B_i[rows1, rows1] = b0b1[:, 1] # bottom-right diagonals
        W_dense = W_dense @ B_i # right mult

    return W_dense

def butterfly_mm_ref(X, W_par, rightmost=True):
    """
    Compute Y = X · W, where W is the product of log_2(L) butterfly factors.

    Inputs
    ----------
    X       : (B, F, L) tensor 
    W_par   : (e, L, 2) tensor   — e = log_2 (L)
    rightmost : bool             — if True, W_par[0] is closest to X
                                   (i.e. multiply stages left to right)

    Returns
    -------
    Y : tensor of shape (B, F, L)
    """

    B, F, L = X.shape
    e = int(math.log2(L))
    assert W_par.shape == (e, L, 2), "W_par has wrong shape"
    assert 1 << e == L,              "L must be a power of two"

    Y = X.clone()
    stage_order = range(e) if rightmost else reversed(range(e))

    for i in stage_order:
        stride = 1 << i # distance between paired indices
        for base in range(0, L, 2*stride):
            j0 = slice(base, base + stride)
            j1 = slice(base + stride, base + 2*stride)

            v0 = Y[:, :, j0].clone() # (B, F, stride)
            v1 = Y[:, :, j1].clone()

            # coefficients broadcast to (B, F, stride)
            a0 = W_par[i, j0, 0].unsqueeze(0).unsqueeze(0)
            a1 = W_par[i, j0, 1].unsqueeze(0).unsqueeze(0)
            b0 = W_par[i, j1, 0].unsqueeze(0).unsqueeze(0)
            b1 = W_par[i, j1, 1].unsqueeze(0).unsqueeze(0)

            Y[:, :, j0] = a0 * v0 + b0 * v1
            Y[:, :, j1] = a1 * v0 + b1 * v1
    return Y


if __name__ == "__main__":
    torch.manual_seed(0)
    B, F, L = 3, 5, 16
    e = int(math.log2(L))

    X      = torch.randn(B, F, L)
    W_par  = torch.randn(e, L, 2)

    Y_ref  = butterfly_mm_ref(X, W_par)
    W_dense = build_dense_from_stages(W_par)
    Y_check = X @ W_dense # (B, F, L) * (L, L) = (B, F, L)

    print("max diff (abs) :", (Y_ref - Y_check).abs().max().item())

    # Small matrix test
    print("\n--- Small matrix test (L=4, B=1, F=1) ---")
    B, F, L = 1, 1, 4
    e = int(math.log2(L))
    X = torch.arange(1, L+1, dtype=torch.float32).reshape(B, F, L)
    W_par = torch.arange(1, e*L*2+1, dtype=torch.float32).reshape(e, L, 2)
    print("X:\n", X)
    print("W_par:\n", W_par)
    W_dense = build_dense_from_stages(W_par, rightmost=True)
    print("W_dense:\n", W_dense)
    Y_ref = butterfly_mm_ref(X, W_par, rightmost=True)
    print("Y_ref:\n", Y_ref)
    Y_check = X @ W_dense
    print("Y_check:\n", Y_check)
    print("max diff (abs) (small test):", (Y_ref - Y_check).abs().max().item())
