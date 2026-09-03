"""
voxel2surf — fairing: implicit variational solve + refinement.

Fairing is the discrete obstacle problem: minimize a thin-plate (bi-Laplacian)
energy ``xᵀ L M⁻¹ L x`` plus a soft data term, subject to per-axis Dirichlet pins
and a box containment, solved per coordinate by primal-dual active set
(``solve_implicit`` / ``_solve_axis_bounded``). Refinement (``subdivide`` midpoint
/ ``decimate_to`` quadric) interleaves with smoothing; ``subdivide`` carries the
provenance mask through (``mask[mid] = mask[a] & mask[b]``).

NOTE: planned — intrinsic Delaunay flips in ``_cotangent_laplacian`` so the
negative-cotangent clamp can be dropped.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def to_triangles(faces: np.ndarray, *, return_map: bool = False):
    """Split quads into two triangles (triangles pass through unchanged).

    With ``return_map=True`` also return ``parent`` (length = n_tris): the source
    face index of each triangle, so per-face provenance (BC patch id, feature
    flags, …) rides onto the triangulation. For quads the two triangles of face
    ``i`` sit at positions ``i`` and ``F + i`` to match the ``vstack`` order.
    """
    faces = np.asarray(faces)
    if faces.shape[1] == 3:
        tris = faces
        parent = np.arange(len(faces))
    else:
        tris = np.vstack([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])
        parent = np.tile(np.arange(len(faces)), 2)
    return (tris, parent) if return_map else tris


def _cotangent_edges(verts, faces) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directed edges with cotangent weights wᵤᵥ = Σ cot(opposite angle), clamped ≥0."""
    v = np.asarray(verts, dtype=float)
    t = to_triangles(faces)
    i, j, k = t[:, 0], t[:, 1], t[:, 2]

    def cot_at(a, b, c):  # cot of the angle at a in triangle (a, b, c)
        u, w = v[b] - v[a], v[c] - v[a]
        return (u * w).sum(1) / np.maximum(np.linalg.norm(np.cross(u, w), axis=1), 1e-12)

    ci, cj, ck = cot_at(i, j, k), cot_at(j, k, i), cot_at(k, i, j)  # opposite edges (j,k),(k,i),(i,j)
    rows = np.concatenate([j, k, k, i, i, j])
    cols = np.concatenate([k, j, i, k, j, i])
    vals = np.concatenate([ci, ci, cj, cj, ck, ck])
    n = len(v)
    cw = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    cw.data = np.maximum(cw.data, 0.0)  # clamp negative (obtuse) weights for stability
    cw.eliminate_zeros()
    co = cw.tocoo()
    return co.row, co.col, co.data


# --- implicit variational solve ----------------------------------------------
def _cotangent_laplacian(verts, faces) -> sp.csr_matrix:
    """Symmetric cotangent Laplacian ``L = D − A`` (PSD; A = clamped cotangent
    weights) — un-normalized and symmetrized so it forms a proper quadratic
    ``xᵀLx`` (the Dirichlet/membrane energy) for the constrained solve."""
    src, dst, w = _cotangent_edges(verts, faces)
    n = len(verts)
    a = sp.coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()  # symmetric (cotan)
    deg = np.asarray(a.sum(1)).ravel()
    return (sp.diags(deg) - a).tocsr()


def _lumped_mass(verts, faces) -> np.ndarray:
    """Barycentric (lumped Voronoi) vertex areas M_ii = ⅓ Σ incident triangle
    areas — the diagonal mass matrix for the thin-plate operator L M⁻¹ L."""
    verts = np.asarray(verts, dtype=float)
    t = to_triangles(faces)
    v0, v1, v2 = verts[t[:, 0]], verts[t[:, 1]], verts[t[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    m = np.zeros(len(verts))
    for k in range(3):
        np.add.at(m, t[:, k], area / 3.0)
    m[m <= 0] = float(m[m > 0].mean()) if (m > 0).any() else 1.0
    return m


def _solve_axis_bounded(Qr, pinned, target, lo, hi, x0, eps, iters):
    """Bound-constrained scalar QP for one coordinate channel by **primal-dual
    active set** (semismooth Newton): minimize ½ xᵀQx + ½ε‖x−x0‖² s.t.
    x[pinned]=target, lo≤x≤hi.

    On the free (un-pinned) variables ``A y = c + μ`` with bound multiplier ``μ``;
    each round predicts the active set from ``μ + c₀(y − bound)`` — which lets a
    variable *leave* a bound, not just join it — fixes active vars to their bound,
    solves the SPD system on the inactive set, and recomputes ``μ``. Converges to
    the true constrained minimizer, so the surface meets the corridor tangentially
    (no crease). A best-feasible-energy globalization guards PDAS cycling."""
    from scipy.sparse.linalg import spsolve

    pinned = np.asarray(pinned, dtype=bool)
    out = np.where(pinned, target, x0).astype(float)
    fidx = np.where(~pinned)[0]
    if len(fidx) == 0:
        return out
    pidx = np.where(pinned)[0]

    A = Qr[fidx][:, fidx].tocsc()
    c = eps * x0[fidx]
    if len(pidx):
        c = c - (Qr[fidx][:, pidx] @ target[pidx])   # coupling to the Dirichlet pins
    loF, hiF = lo[fidx], hi[fidx]
    c0 = float(A.diagonal().mean()) or 1.0

    def energy(z):  # ½ zᵀA z − cᵀz on the feasible (clipped) iterate
        zf = np.clip(z, loF, hiF)
        return 0.5 * float(zf @ (A @ zf)) - float(c @ zf), zf

    y = np.clip(x0[fidx], loF, hiF)
    mu = np.zeros(len(fidx))
    best_e, best_y = energy(y)
    prev = None
    for _ in range(max(1, iters)):
        al = (mu + c0 * (y - loF)) < 0          # lower-active
        au = (mu + c0 * (y - hiF)) > 0          # upper-active
        au &= ~al
        free = ~(al | au)
        if prev is not None and np.array_equal(al, prev[0]) and np.array_equal(au, prev[1]):
            break                                # active set fixed → converged
        prev = (al.copy(), au.copy())
        y[al] = loF[al]
        y[au] = hiF[au]
        if free.any():
            rhs = c[free].copy()
            if al.any():
                rhs -= A[free][:, al] @ loF[al]
            if au.any():
                rhs -= A[free][:, au] @ hiF[au]
            y[free] = spsolve(A[free][:, free].tocsc(), rhs)
        e, yf = energy(y)                         # globalization: keep best feasible
        if e < best_e:
            best_e, best_y = e, yf
        mu = np.zeros(len(fidx))
        act = al | au
        if act.any():
            mu[act] = (A[act] @ y) - c[act]
    out[fidx] = best_y
    return out


def solve_implicit(verts, faces, pinned, target, lo, hi, *,
                   order: str = "membrane", iters: int = 6, reg: float = 1e-4):
    """Minimize the fairing energy per coordinate channel subject to the box pins
    (``pinned``→``target``) and the occupancy containment ``[lo, hi]`` — the
    implicit fair-and-constrain. ``order`` is ``"membrane"`` (Dirichlet, `Q=L`) or
    ``"thin-plate"`` (bi-Laplacian, `Q=L M⁻¹ L`, C¹ at constraints). ``reg`` weights a
    soft pull ``½·reg·mean(diagQ)·‖x−x0‖²`` toward the input positions: the
    **anti-shrink data term** that also convexifies the bi-Laplacian (its null
    space makes the box-only problem ill-posed). Returns the corrected positions."""
    verts = np.asarray(verts, dtype=float)
    n = len(verts)
    L = _cotangent_laplacian(verts, faces)
    if order == "thin-plate":
        minv = sp.diags(1.0 / _lumped_mass(verts, faces))
        Q = (L @ minv @ L).tocsr()  # mass-weighted bi-Laplacian (geometrically correct)
    elif order == "membrane":
        Q = L
    else:
        raise ValueError(f"order must be 'membrane' or 'thin-plate', got {order!r}")
    d = Q.diagonal()
    # ``reg`` (the soft x0 data term) is optional; the 1e-6 floor keeps Q SPD on
    # pin-free components when it is off, so the box containment does the anti-shrink.
    eps = max(reg, 1e-6) * (float(np.mean(d[d > 0])) if (d > 0).any() else 1.0)
    Qr = (Q + eps * sp.identity(n)).tocsr()
    pinned = np.asarray(pinned, dtype=bool)
    target = np.asarray(target, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    x = verts.copy()
    for ax in range(3):
        x[:, ax] = _solve_axis_bounded(
            Qr, pinned[:, ax], target[:, ax], lo[:, ax], hi[:, ax], verts[:, ax], eps, iters)
    return x


# --- refinement (decimate needs pyvista) -------------------------------------
def subdivide(verts, faces, mask=None):
    """One 1→4 midpoint subdivision; returns triangles (input quads triangulated).

    Implemented in numpy (no trimesh) so a per-vertex provenance ``mask`` (V, k)
    can ride through: each new edge-midpoint vertex inherits ``mask[a] & mask[b]``
    from its two parents. For the box-face mask this is exact — a midpoint lies
    on a coordinate plane iff *both* endpoints do, which is precisely the bitwise
    AND. Original vertices keep indices ``0..V-1`` (and their mask rows); unique
    edge ``k`` becomes vertex ``V + k``.

    Returns ``(verts, faces)`` normally, or ``(verts, faces, mask)`` when ``mask``
    is supplied.
    """
    verts = np.asarray(verts, dtype=float)
    tris = to_triangles(faces)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]

    # unique undirected edges; inv maps each (local-edge, tri) → unique edge id.
    e = np.concatenate([
        np.sort(np.stack([a, b], 1), 1),
        np.sort(np.stack([b, c], 1), 1),
        np.sort(np.stack([c, a], 1), 1),
    ], 0)
    ue, inv = np.unique(e, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel().reshape(3, len(tris))  # rows: ab, bc, ca

    V = len(verts)
    mab, mbc, mca = V + inv[0], V + inv[1], V + inv[2]
    mid = 0.5 * (verts[ue[:, 0]] + verts[ue[:, 1]])
    new_verts = np.vstack([verts, mid])
    new_tris = np.vstack([
        np.stack([a, mab, mca], 1),
        np.stack([b, mbc, mab], 1),
        np.stack([c, mca, mbc], 1),
        np.stack([mab, mbc, mca], 1),
    ])
    if mask is None:
        return new_verts, new_tris
    mask = np.asarray(mask, dtype=bool)
    new_mask = np.vstack([mask, mask[ue[:, 0]] & mask[ue[:, 1]]])
    return new_verts, new_tris, new_mask


def decimate_to(verts, faces, target_faces: int):
    """Quadric edge-collapse to ~``target_faces`` (boundary-preserving) via pyvista."""
    import pyvista as pv

    tris = to_triangles(faces)
    pd = pv.PolyData(np.asarray(verts), np.c_[np.full(len(tris), 3), tris].ravel())
    reduction = max(0.0, 1.0 - target_faces / len(tris))
    out = pd.decimate_pro(reduction, preserve_topology=True, boundary_vertex_deletion=False)
    return np.asarray(out.points, dtype=float), out.faces.reshape(-1, 4)[:, 1:]
