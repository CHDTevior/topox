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

H1 (2026-09-24): `heat_kernel_signature` is the alternative coordinate for the same RoPE. The eigenvector coordinate has
two weaknesses that were measured on the corpora: about half of the rigs (25 of 50 sampled PZ rigs,
handoff/20260924_161717_h1_hks_notes.md) have a repeated eigenvalue among the first 8 non-trivial modes, where the basis
(not only the sign) is LAPACK's, so the SignNet is fed a coordinate that a mere relabelling of the joints changes
column-wide (PZ 50%, UniML3D 66% of the rigs; handoff/20260923_224710_ood_research_synthesis.md); and the kept
eigenvectors move by O(1) when a leaf is removed, because the eigenvalue gaps are ~0. The heat-kernel signature is the
diagonal of the spectral kernel exp(-t L) over every non-trivial mode, which depends on the eigenSPACES only, so it is
invariant to sign and basis, stable under small edits of the tree, and needs no SignNet.
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


def hks_scales(num_scales: int = 8):
    """The heat-kernel time ladder: t_i = 0.25 * 2^i for i < num_scales (8 scales: 0.25, 0.5, 1, 2, 4, 8, 16, 32), one
    octave per scale from a quarter step (the joint's own degree structure) to 32 steps (the slowest modes of the tree).
    Sized by the same K that sizes the eigenvector coordinate, so the served array keeps its [J, K] shape."""
    return 0.25 * (2.0 ** np.arange(int(num_scales), dtype=np.float64))


def heat_kernel_signature(parents, scales):
    """parents [J] int (root: -1 or itself) -> trace-normalised HKS [J, len(scales)] float32:
    h~_j(t) = h_j(t) / mean_i h_i(t), with h_j(t) = sum_{k >= 1} exp(-t lambda_k) v_kj^2 over ALL J - 1 non-trivial
    eigenpairs of the same symmetric-normalised Laplacian as `laplacian_eigenvectors` (no truncation), the trivial pair
    (lambda_0 = 0) skipped (`_heat_kernel_signature64`, float64). h is the diagonal of the heat kernel exp(-t L) minus its
    trivial part, so it depends on the eigenspaces only: the sign and the basis LAPACK picks inside a repeated eigenvalue
    drop out (v_kj^2 summed over the eigenspace is the projector's diagonal), and a relabelling of the joints permutes the
    rows; dividing every column by its mean over the rig's joints (the scaled HKS of Bronstein & Kokkinos 2010) keeps both
    properties and removes the per-rig magnitude: raw h lies in (0, 1) and its column means differ from rig to rig
    (measured over 50 training rigs, J 22-96: the t = 32 column's mean spans 0.028-0.054, a 1.9x spread, Spearman -0.51
    against J), a per-rig scale the encoder's first layer cannot absorb (user 2026-09-24: normalise before training).
    Served: every column has mean 1 over the rig, only a joint's standing relative to the rig's other joints remains. No zero padding is needed: the width is the
    scale count whatever J is; a single-joint tree, which has no non-trivial mode (h = 0, mean 0), gets zeros."""
    h = _heat_kernel_signature64(parents, scales)                      # [J, S] raw, every value in (0, 1) for J >= 2
    m = h.mean(axis=0, keepdims=True)                                 # [1, S] per-rig column mean, > 0 for J >= 2
    return np.divide(h, m, out=np.zeros_like(h), where=m > 0).astype(np.float32)


def _heat_kernel_signature64(parents, scales):
    """The float64 computation behind `heat_kernel_signature` (the test suite checks its invariances at 1e-9)."""
    parents = np.asarray(parents, dtype=np.int64).ravel()
    n = int(parents.shape[0])
    A = np.zeros((n, n), dtype=np.float64)
    for j, p in enumerate(parents.tolist()):
        if p >= 0 and p != j:
            A[j, p] = A[p, j] = 1.0
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)), 0.0)
    L = np.eye(n) - d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    eigvals_all, eigvecs_all = np.linalg.eigh(L)
    lam = np.maximum(eigvals_all[1:], 0.0)                     # [J-1] non-trivial eigenvalues, clamped as above
    v2 = np.real(eigvecs_all[:, 1:]) ** 2                     # [J, J-1] squared components: sign and basis drop out
    t = np.asarray(scales, dtype=np.float64).ravel()
    return v2 @ np.exp(-np.outer(lam, t))                     # [J, S]
