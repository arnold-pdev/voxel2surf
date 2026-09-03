"""
voxel2surf — step 2: labelling for constrained fairing.

The smoothing constraint is read off a per-vertex 6-bit box-face mask
(bit ``2*ax+side``: side 0 = min plane, side 1 = max plane). Because the box
planes are axis-aligned, each face pins one *coordinate*, so the admissible
motion is fixed by the number of *distinct axes* the vertex is pinned in:
  - 0 axes → free      (3 DOF),
  - 1 axis → on_face   (2 DOF, slide in the plane; pin 1 coordinate),
  - 2 axes → on_edge   (1 DOF, slide along the edge line; pin 2 coordinates),
  - 3 axes → on_corner (0 DOF, fixed; pin 3 coordinates).

The mask is *provenance*, not geometry: ``extract.cuberille_mesh`` establishes it
exactly from the integer grid key, and ``fair.subdivide`` carries it through
midpoint refinement via ``mask[mid] = mask[a] & mask[b]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Constraints:
    """Per-vertex constraint class for the fairing solver."""

    fixed: np.ndarray    # (V,) bool
    tangent: np.ndarray  # (V, 3)
    on_face: np.ndarray  # (V, 6) bool
    face_idx: np.ndarray  # (V,) int


def _box_face_mask(verts, origin, shape, spacing, tol) -> np.ndarray:
    """Geometric fallback: (V, 6) box-face membership by coordinate distance."""
    lo, hi = origin, origin + shape * spacing
    at_lo = np.abs(verts - lo) <= tol
    at_hi = np.abs(verts - hi) <= tol
    on_face = np.zeros((len(verts), 6), dtype=bool)
    for ax in range(3):
        on_face[:, 2 * ax] = at_lo[:, ax]
        on_face[:, 2 * ax + 1] = at_hi[:, ax]
    return on_face


def classify(
    verts: np.ndarray,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    shape=None,
    *,
    anchors: np.ndarray | None = None,
    tol: float | None = None,
    face_mask: np.ndarray | None = None,
) -> Constraints:
    """Classify each vertex as free / on_face / on_edge / on_corner."""
    verts = np.asarray(verts, dtype=float)
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    if shape is None:
        raise ValueError("shape (nx, ny, nz) is required to locate the domain box")
    shape = np.asarray(shape, dtype=int)

    if face_mask is not None:
        on_face = np.asarray(face_mask, dtype=bool)
        if on_face.shape != (len(verts), 6):
            raise ValueError(
                f"face_mask must be (V, 6) = ({len(verts)}, 6), got {on_face.shape}")
    else:
        if tol is None:
            tol = 1e-6
        on_face = _box_face_mask(verts, origin, shape, spacing, tol)

    V = len(verts)
    ax_hit = np.zeros((V, 3), dtype=bool)
    for a in range(3):
        ax_hit[:, a] = on_face[:, 2 * a] | on_face[:, 2 * a + 1]
    n_ax = ax_hit.sum(axis=1)

    onface = n_ax == 1
    onedge = n_ax == 2
    fixed = n_ax >= 3

    face_idx = np.full(V, -1, dtype=int)
    iface = np.where(onface)[0]
    if len(iface):
        fa = np.argmax(ax_hit[iface], axis=1)
        at_max = on_face[iface, 2 * fa + 1]
        face_idx[iface] = 2 * fa + at_max.astype(int)

    tangent = np.zeros_like(verts)
    iedge = np.where(onedge)[0]
    if len(iedge):
        free_ax = np.argmin(ax_hit[iedge], axis=1)
        tangent[iedge, free_ax] = 1.0

    if anchors is not None and len(anchors):
        from scipy.spatial import cKDTree

        _, idx = cKDTree(verts).query(np.asarray(anchors, dtype=float))
        anchored = np.unique(np.atleast_1d(idx).astype(int))
        fixed[anchored] = True
        tangent[anchored] = 0.0
        face_idx[anchored] = -1

    return Constraints(
        fixed=fixed, tangent=tangent, on_face=on_face, face_idx=face_idx)
