"""
voxel2surf — occupancy containment for the implicit fairing solver.

Cell-data analogue of Gibson's Constrained Elastic Surface Nets: a cuberille
corner vertex is contained in its **dual cell** — an axis-aligned box of
half-width ½ voxel centred on the vertex. Staying within ½ voxel keeps the
surface from crossing a *neighbouring voxel centre*, so it cannot flip a cell's
solid/void classification → the smoothed surface stays occupancy-consistent with
the binary.

This is a strong per-vertex proxy for the exact per-cell volume-fraction
constraint; it keeps the per-axis box-QP and does **not** leak (containment is
genuinely axis-aligned, so there is no normal direction to approximate). It does
*not* prevent two adjacent vertices from meeting on their shared dual-cell face
(zero-area triangles) — that extrinsic degeneracy is a separate concern.
"""

from __future__ import annotations

import numpy as np


def derive_bounds(verts, spacing, *, half_voxels: float = 0.5):
    """Per-vertex axis-aligned containment box ``[x0 − r, x0 + r]`` with
    ``r = half_voxels · spacing`` — the dual cell centred on each vertex. Returns
    ``(lo, hi)``, each (V, 3)."""
    x0 = np.asarray(verts, dtype=float)
    r = float(half_voxels) * np.asarray(spacing, dtype=float)
    return x0 - r, x0 + r
