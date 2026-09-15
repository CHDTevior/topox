"""Spectral coordinates of a skeleton tree, ported from UniMate
(outside_docs/UniMate/unimate/utils/topology_utils.py compute_laplacian_eigenvectors with its defaults
laplacian_norm='sym', eigvec_norm='L2'; UniMate builds the Laplacian through torch_geometric.get_laplacian, which
is not installed here, so the symmetric-normalised Laplacian L = I - D^-1/2 A D^-1/2 is assembled densely).

Per joint the K smallest NON-TRIVIAL eigenvectors of L (the trivial eigenvalue ~0 is skipped), eigenvalues clamped at
0, every column L2-normalised, the frequency axis zero-padded to K when the tree has fewer than K + 1 joints. The
signs of the columns are LAPACK's choice: the consumer (src/models/v2/spec_rope.py SignNetSpectralEncoder) is
sign-invariant by construction, as in UniMate. Inside a degenerate eigenspace (a rig with mirror-symmetric limbs has
repeated eigenvalues) the basis is LAPACK's choice too, and UniMate shares that property; numpy's eigh is deterministic
for identical input bytes on one machine, and the pair loader caches one array per rig.
"""
import numpy as np


def laplacian_eigenvectors(parents, max_freqs: int = 8):
    """parents [J] int (root: -1 or itself) -> (eigvecs [J, max_freqs] float32, eigvals [max_freqs] float32)."""
    parents = np.asarray(parents, dtype=np.int64).ravel()
    n = int(parents.shape[0])
    k = min(n - 1, int(max_freqs))
    A = np.zeros((n, n), dtype=np.float64)
    for j, p in enumerate(parents.tolist()):
        if p >= 0 and p != j:
            A[j, p] = A[p, j] = 1.0
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)), 0.0)
    L = np.eye(n) - d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    eigvals_all, eigvecs_all = np.linalg.eigh(L)
    eigvals = np.maximum(eigvals_all[1:1 + k], 0.0)
    eigvecs = np.real(eigvecs_all[:, 1:1 + k]).copy()
    for c in range(eigvecs.shape[1]):
        col = eigvecs[:, c]
        denom = np.sqrt((col ** 2).sum())
        if denom > 1e-12:
            eigvecs[:, c] = col / denom
    if k < max_freqs:
        eigvecs = np.pad(eigvecs, ((0, 0), (0, max_freqs - k)))
        eigvals = np.pad(eigvals, (0, max_freqs - k))
    return eigvecs.astype(np.float32), eigvals.astype(np.float32)
