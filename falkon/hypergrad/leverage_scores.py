import torch

import falkon


def subs_deff_simple(kernel: falkon.kernels.Kernel,
                     penalty: torch.Tensor,
                     X: torch.Tensor,
                     J: torch.Tensor) -> torch.Tensor:
    n = X.shape[0]
    K_JJ = kernel(J, J)
    K_XJ = kernel(X, J)

    # Inversion
    U, S, _ = torch.svd(K_JJ + penalty * n * torch.eye(K_JJ.shape[0]))
    # Exclude eigen-values close to 0
    thresh = (S[0] * S.shape[0] * torch.finfo(S.dtype).eps).item()
    stable_eig = (S > thresh)
    U_thin = U[:, stable_eig]  # n x m
    S_thin = S[stable_eig]     # m
    S_thin_root_inv = torch.sqrt(S_thin).reciprocal().reshape(-1, 1)  # m x 1
    K_JJ_inv = S_thin_root_inv * U_thin.T   # square root inverse (m x n)

    # Multiply by the large kernel
    E = K_JJ_inv @ K_XJ.T  # x * j

    # Calculate the trace, which is just the squared norm of the columns
    dim_eff = (1 / penalty) - (1 / (penalty * n)) * torch.sum(torch.norm(E, dim=1) ** 2)
    return dim_eff


def full_deff_simple(kernel, penalty, X):
    K = kernel(X, X)
    K_inv = torch.pinverse(K + penalty * X.shape[0] * torch.eye(X.shape[0]))
    tau = torch.diagonal(K @ K_inv)
    return tau.sum()


def full_deff(kernel, penalty, X: torch.Tensor):
    K = kernel(X, X)  # n * n
    n = X.shape[0]

    K_invertible = K + penalty * n * torch.eye(X.shape[0])

    U_DD, S_DD, _ = torch.svd(K_invertible)
    # Exclude eigen-values close to 0
    thresh = (S_DD.max() * S_DD.shape[0] * torch.finfo(S_DD.dtype).eps).item()
    stable_eig = (S_DD > thresh)
    U_thin = U_DD[:, stable_eig]  # n x m
    S_thin = S_DD[stable_eig]     # m
    S_thin_inv = torch.sqrt(S_thin).reciprocal().reshape(-1, 1)  # m x 1

    # diag(S_root_inv) @ U.T
    E = S_thin_inv * U_thin.T   # square root inverse (m x n)
    E = E @ K
    # the diagonal entries of XX'(X'X + lam*S^(-2))^(-1)XX' are just the squared
    # ell-2 norm of the columns of (X'X + lam*S^(-2))^(-1/2)XX'
    tau = (torch.diagonal(K) - torch.square(E).sum(0)) / (penalty * n)
    assert torch.all(tau >= 0)
    d_eff = tau.sum()
    return d_eff
