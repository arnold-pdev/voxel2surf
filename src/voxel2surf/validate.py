"""
voxel2surf — mesh-quality validation (numpy/scipy only; runs in any env).

Topology / watertightness, signed volume + volume-fraction delta, free-skin
dihedral p95, self-intersection count (vectorized Möller broadphase with the
box-plane BC faces excluded), inter-body overlap, BC-plane residual, and
load-on-surface. ``validate`` assembles the report and gates the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from . import fair
from .label import classify

if TYPE_CHECKING:
    from .pipeline import Logger


# --- mesh statistics (numpy/scipy only — works in any env) -------------------
def _edge_incidence(faces):
    f = np.asarray(faces)
    k = f.shape[1]
    e = np.concatenate(
        [np.sort(np.stack([f[:, i], f[:, (i + 1) % k]], 1), 1) for i in range(k)], 0
    )
    return np.unique(e, axis=0, return_counts=True)


def topo(verts, faces) -> dict:
    eu, cnt = _edge_incidence(faces)
    n = len(verts)
    a = sp.coo_matrix((np.ones(len(eu)), (eu[:, 0], eu[:, 1])), shape=(n, n))
    _, lab = connected_components(a, directed=False)
    used = np.unique(np.asarray(faces))
    return {
        "verts": len(verts),
        "faces": len(faces),
        "bodies": int(len(np.unique(lab[used]))) if len(used) else 0,
        "watertight": bool((cnt == 2).all()),
        "nonmanifold_edges": int((cnt != 2).sum()),
    }


def _signed_volume(verts, faces) -> float:
    t = fair.to_triangles(faces)
    v0, v1, v2 = verts[t[:, 0]], verts[t[:, 1]], verts[t[:, 2]]
    return float(np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v0).sum() / 6.0)


def _free_dihedral_p95(verts, faces, origin, spacing, shape) -> float:
    t = fair.to_triangles(faces)
    if len(t) == 0:
        return 0.0
    v0, v1, v2 = verts[t[:, 0]], verts[t[:, 1]], verts[t[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    cen = verts[t].mean(1)
    tolb = 0.25 * float(np.min(spacing))
    lo = np.asarray(origin, float)
    hi = lo + np.asarray(shape, int) * np.asarray(spacing, float)
    free = ~((np.abs(cen - lo) <= tolb).any(1) | (np.abs(cen - hi) <= tolb).any(1))
    e = np.concatenate([np.sort(np.stack([t[:, i], t[:, (i + 1) % 3]], 1), 1) for i in range(3)], 0)
    tid = np.tile(np.arange(len(t)), 3)
    eu, inv = np.unique(e, axis=0, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_o, tid_o = inv[order], tid[order]
    slot = np.arange(len(inv_o)) - np.searchsorted(inv_o, np.arange(len(eu)))[inv_o]
    et = np.full((len(eu), 2), -1, dtype=int)
    k2 = slot < 2
    et[inv_o[k2], slot[k2]] = tid_o[k2]
    man = (et[:, 0] >= 0) & (et[:, 1] >= 0)
    both_free = man & free[et[:, 0]] & free[et[:, 1]]
    if not both_free.any():
        return 0.0
    dot = np.abs((nrm[et[both_free, 0]] * nrm[et[both_free, 1]]).sum(1)).clip(-1, 1)
    return float(np.percentile(np.degrees(np.arccos(dot)), 95))


def _interval(d, v):
    pos, neg = d > 1e-12, d < -1e-12
    if pos.sum() == 1:
        lone, others = int(np.where(pos)[0][0]), np.where(~pos)[0]
    elif neg.sum() == 1:
        lone, others = int(np.where(neg)[0][0]), np.where(~neg)[0]
    else:
        z = np.where(np.abs(d) <= 1e-12)[0]
        if len(z) == 0:
            return None
        lone, others = int(z[0]), np.array([i for i in range(3) if i != z[0]])
    ts = []
    for o in others:
        den = d[lone] - d[o]
        ts.append(v[lone] if abs(den) < 1e-15 else v[lone] + (v[o] - v[lone]) * d[lone] / den)
    return (min(ts), max(ts))


def _tri_tri(p, q, eps=1e-9):
    """Möller (1997) triangle-triangle intersection test for 3×3 arrays p, q."""
    p0, p1, p2 = p
    q0, q1, q2 = q
    n2 = np.cross(q1 - q0, q2 - q0)
    dp = n2 @ p.T - n2.dot(q0)
    if (dp > eps).all() or (dp < -eps).all():
        return False
    n1 = np.cross(p1 - p0, p2 - p0)
    dq = n1 @ q.T - n1.dot(p0)
    if (dq > eps).all() or (dq < -eps).all():
        return False
    d = np.cross(n1, n2)
    if d @ d < eps:  # coplanar: 2D edge-cross or containment
        ax = int(np.argmax(np.abs(n1)))
        keep = [i for i in range(3) if i != ax]
        a, b = p[:, keep], q[:, keep]

        def o(u, w, r):
            return np.sign((w[0] - u[0]) * (r[1] - u[1]) - (w[1] - u[1]) * (r[0] - u[0]))

        for i in range(3):
            for j in range(3):
                if o(a[i], a[(i + 1) % 3], b[j]) != o(a[i], a[(i + 1) % 3], b[(j + 1) % 3]) and \
                   o(b[j], b[(j + 1) % 3], a[i]) != o(b[j], b[(j + 1) % 3], a[(i + 1) % 3]):
                    return True

        def inside(pt, tri):
            s = [np.sign(np.cross(tri[(k + 1) % 3] - tri[k], pt - tri[k])) for k in range(3)]
            return all(x >= 0 for x in s) or all(x <= 0 for x in s)

        return inside(a[0], b) or inside(b[0], a)
    idx = int(np.argmax(np.abs(d)))
    i1, i2 = _interval(dp, p[:, idx]), _interval(dq, q[:, idx])
    if i1 is None or i2 is None:
        return False
    return max(i1[0], i2[0]) <= min(i1[1], i2[1]) + eps


def _self_intersections(verts, faces, *, origin=None, spacing=None, shape=None) -> int:
    """Count intersecting non-adjacent triangle pairs in the *free* skin.

    Triangles lying on a domain box plane (the BC faces) are excluded: they are
    exact planar tessellations that cannot self-intersect, and — being large and
    coplanar — would otherwise flood the broadphase with pairs that all fall into
    the slow coplanar branch. The realistic failure mode (thin free members
    folding onto themselves) is between free triangles, checked here in full.

    Broadphase uses the *median* triangle diameter (robust to the few large
    decimated faces) and a vectorized AABB + Möller plane early-reject, so only
    genuine plane-crossing pairs reach the Python narrowphase. Numpy/scipy only.
    """
    from scipy.spatial import cKDTree

    t = fair.to_triangles(faces)
    if len(t) < 2:
        return 0
    V = np.asarray(verts, float)
    cen_all = V[t].mean(1)
    if origin is not None and shape is not None:
        lo = np.asarray(origin, float)
        hi = lo + np.asarray(shape, int) * np.asarray(spacing, float)
        on_box = (np.abs(cen_all - lo) <= 1e-6).any(1) | (np.abs(cen_all - hi) <= 1e-6).any(1)
        t = t[~on_box]
    if len(t) < 2:
        return 0

    p = V[t]  # (n, 3, 3)
    cen = p.mean(1)
    rad = np.linalg.norm(p - cen[:, None, :], axis=2).max(1)   # per-triangle radius
    r = 2.0 * float(np.median(rad)) + 1e-12
    pairs = cKDTree(cen).query_pairs(r, output_type="ndarray")
    if len(pairs) == 0:
        return 0
    a, b = pairs[:, 0], pairs[:, 1]
    shared = (t[a][:, :, None] == t[b][:, None, :]).any(axis=(1, 2))  # edge/vertex-adjacent
    a, b = a[~shared], b[~shared]
    if len(a) == 0:
        return 0

    PA, PB = p[a], p[b]
    ov = ((PA.min(1) <= PB.max(1)) & (PB.min(1) <= PA.max(1))).all(1)  # tight AABB overlap
    a, b, PA, PB = a[ov], b[ov], PA[ov], PB[ov]
    if len(a) == 0:
        return 0

    eps = 1e-9  # vectorized Möller plane test: all of A one side of B's plane → no hit
    nB = np.cross(PB[:, 1] - PB[:, 0], PB[:, 2] - PB[:, 0])
    dA = np.einsum("mij,mj->mi", PA - PB[:, :1], nB)
    nA = np.cross(PA[:, 1] - PA[:, 0], PA[:, 2] - PA[:, 0])
    dB = np.einsum("mij,mj->mi", PB - PA[:, :1], nA)
    sep = ((dA > eps).all(1) | (dA < -eps).all(1)) | ((dB > eps).all(1) | (dB < -eps).all(1))
    a, b = a[~sep], b[~sep]
    return int(sum(_tri_tri(p[i], p[j]) for i, j in zip(a, b)))


def _body_overlaps(verts, faces):
    """Inter-body interpenetration (None if trimesh unavailable)."""
    try:
        import trimesh
    except Exception:
        return None
    m = trimesh.Trimesh(np.asarray(verts), fair.to_triangles(faces), process=False)
    try:
        bodies = m.split(only_watertight=False)
    except Exception:
        return None
    n = 0
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            try:
                ai = (
                    bodies[j].contains(bodies[i].vertices).any()
                    if len(bodies[i].vertices)
                    else False
                )
                aj = (
                    bodies[i].contains(bodies[j].vertices).any()
                    if len(bodies[j].vertices)
                    else False
                )
            except Exception:
                return None
            n += int(bool(ai or aj))
    return n


def validate(verts, faces, *, vox, origin, spacing, shape, log: "Logger",
             anchors=None, on_fail: str = "warn", bc_tol: float = 1e-5) -> dict:
    origin = np.asarray(origin, float)
    spacing = np.asarray(spacing, float)
    shape = np.asarray(shape, int)
    rep = topo(verts, faces)

    vol = abs(_signed_volume(verts, faces))
    domain = float(np.prod(shape * spacing))
    rep["vf_voxel"] = float(np.count_nonzero(vox)) / float(np.prod(shape))
    rep["vf_mesh"] = (vol / domain) if rep["watertight"] else float("nan")
    rep["vf_delta"] = rep["vf_mesh"] - rep["vf_voxel"]

    c = classify(verts, origin, spacing, shape, anchors=anchors)
    onp = c.face_idx >= 0
    if onp.any():
        fi = c.face_idx[onp]
        ax = fi // 2
        ext = np.where(fi % 2 == 1, (origin + shape * spacing)[ax], origin[ax])
        vax = verts[onp][np.arange(int(onp.sum())), ax]
        rep["bc_plane_max_residual"] = float(np.abs(vax - ext).max())
    else:
        rep["bc_plane_max_residual"] = 0.0

    rep["load_off_surface"], rep["load_max_dist"] = 0, 0.0
    if anchors is not None and len(anchors):
        from scipy.spatial import cKDTree

        d, _ = cKDTree(verts).query(np.asarray(anchors, float))
        rep["load_max_dist"] = float(d.max())
        rep["load_off_surface"] = int((d > 0.5 * float(spacing.min())).sum())

    rep["free_dihedral_p95_deg"] = _free_dihedral_p95(verts, faces, origin, spacing, shape)
    rep["self_intersections"] = _self_intersections(
        verts, faces, origin=origin, spacing=spacing, shape=shape)
    rep["body_overlaps"] = _body_overlaps(verts, faces)

    failed = []
    if not rep["watertight"]:
        failed.append("not-watertight")
    if rep["self_intersections"] > 0:
        failed.append(f"self_intersections={rep['self_intersections']}")
    if rep["body_overlaps"] not in (0, None):
        failed.append(f"body_overlaps={rep['body_overlaps']}")
    if rep["bc_plane_max_residual"] > bc_tol:
        failed.append(f"bc_plane_residual>{bc_tol:g}")
    if rep["load_off_surface"] > 0:
        failed.append(f"load_off_surface={rep['load_off_surface']}")
    rep["gates_failed"] = failed

    ov = "skipped" if rep["body_overlaps"] is None else rep["body_overlaps"]
    log("  validate:")
    log(f"        watertight={rep['watertight']}  bodies={rep['bodies']}  "
        f"nonmanifold_edges={rep['nonmanifold_edges']}")
    log(f"        vf_mesh={rep['vf_mesh']:.4f}  vf_voxel={rep['vf_voxel']:.4f}  "
        f"vf_delta={rep['vf_delta']:+.4f}")
    log(f"        bc_plane_max_residual={rep['bc_plane_max_residual']:.2e}  "
        f"load_off_surface={rep['load_off_surface']} (max_dist={rep['load_max_dist']:.3f})")
    log(f"        free_dihedral_p95={rep['free_dihedral_p95_deg']:.1f}deg  "
        f"self_intersections={rep['self_intersections']}  body_overlaps={ov}")
    log(f"  gates: {'PASS' if not failed else 'FAIL -> ' + ', '.join(failed)}")
    if failed and on_fail == "raise":
        raise ValueError("validation gates failed: " + ", ".join(failed))
    return rep
