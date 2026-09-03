"""
voxel2surf — step 1: cuberille extraction with face-connectivity weld.

One exposed voxel face → one quad (1:1). Vertices are welded only across voxels
that are *face-connected* (6-connectivity), evaluated locally at each grid corner
via its 8-voxel rosette. This makes the surface graph's connectivity identical to
the voxel FEM's face-connectivity: diagonal (edge/vertex) contacts split into
separate manifold sheets; face contacts weld. No iso-surfacing, no tolerances.

Returns world-coordinate vertices, quad faces, and per-face provenance
``(cell, axis, side)`` for the labelling step.
"""

from __future__ import annotations

import numpy as np

# --- rosette octant indexing -------------------------------------------------
# Octant o in 0..7 <-> (di,dj,dk) bits = the 8 voxels touching a grid corner.
_OCT = [(o & 1, (o >> 1) & 1, (o >> 2) & 1) for o in range(8)]
_OCT_EDGES = [
    (a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1
]  # octant face-adjacency = Hamming-1


def _build_comp_table() -> np.ndarray:
    """``COMP[mask, o]`` = face-connected component id of octant ``o`` in the
    rosette ``mask`` (root octant index), or -1 if ``o`` is void. Pure function of
    the 8-bit mask → a 256×8 table built once at import."""
    comp = np.full((256, 8), -1, dtype=np.int8)
    for mask in range(256):
        parent = list(range(8))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in _OCT_EDGES:
            if (mask >> a) & 1 and (mask >> b) & 1:
                parent[find(a)] = find(b)
        for o in range(8):
            if (mask >> o) & 1:
                comp[mask, o] = find(o)
    return comp


_COMP_TABLE = _build_comp_table()


def _rosette_mask_field(solid: np.ndarray) -> np.ndarray:
    """``(nx+1, ny+1, nz+1)`` uint8 field: bit ``o`` at grid corner ``p`` is set
    iff the octant-``o`` voxel ``p − 1 + oct`` is solid. Vectorized over corners."""
    nx, ny, nz = solid.shape
    sp = np.zeros((nx + 2, ny + 2, nz + 2), dtype=bool)
    sp[1:-1, 1:-1, 1:-1] = solid
    field = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.uint8)
    for o, (di, dj, dk) in enumerate(_OCT):
        field |= sp[di : di + nx + 1, dj : dj + ny + 1, dk : dk + nz + 1].astype(np.uint8) << o
    return field


def _exposed_cells(solid: np.ndarray, axis: int, side: int) -> np.ndarray:
    """Solid cells whose ``(axis, side)`` face is exposed (side: 0=min, 1=max)."""
    nbr = np.zeros_like(solid)
    dst = [slice(None)] * 3
    src = [slice(None)] * 3
    if side == 1:
        dst[axis], src[axis] = slice(0, -1), slice(1, None)
    else:
        dst[axis], src[axis] = slice(1, None), slice(0, -1)
    nbr[tuple(dst)] = solid[tuple(src)]
    return solid & ~nbr


def _face_corner_specs(axis: int, side: int) -> list[tuple[tuple[int, int, int], int]]:
    """Per-(axis, side) face: four ``(corner offset, owner octant)`` pairs.

    ``offset`` is the grid-corner offset from the cell base (each in {0,1});
    the owner octant has bit = ``1 − offset`` (the low corner sits in octant di=1).
    Corner order is CCW about the outward normal (axis-1 parity flip applied).
    """
    b, d = (ax for ax in range(3) if ax != axis)
    oa = 0 if side == 0 else 1
    order = [(0, 0), (1, 0), (1, 1), (0, 1)]  # CCW in the (b, d) plane
    if (1 if axis in (0, 2) else -1) * (1 if side == 1 else -1) < 0:
        order = order[::-1]
    specs = []
    for ob, od in order:
        off = [0, 0, 0]
        off[axis], off[b], off[d] = oa, ob, od
        octant = (1 - off[0]) | ((1 - off[1]) << 1) | ((1 - off[2]) << 2)
        specs.append((tuple(off), octant))
    return specs


def cuberille_mesh(
    vox: np.ndarray,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    density_cutoff: float = 0.5,
):
    """
    Build the face-connectivity-welded cuberille surface.

    Returns ``(verts, faces, prov)``:
      verts : (V, 3) float world coordinates,
      faces : (F, 4) int quads indexing ``verts`` (1 per exposed voxel face),
      prov  : dict with ``cell`` (F, 3), ``axis`` (F,), ``side`` (F,) per face,
              and ``vmask`` (V, 6) bool — per-vertex box-face membership
              (bit ``2*ax+side``: side 0 = min plane, side 1 = max plane),
              read *exactly* off the integer grid key (no tolerance). This is
              the base provenance mask; ``fair.subdivide`` propagates it via
              ``mask[mid] = mask[a] & mask[b]`` and ``classify`` turns it into
              the per-vertex motion constraint.
    """
    solid = np.asarray(vox) > density_cutoff
    if solid.ndim != 3:
        raise ValueError(f"cuberille_mesh expects a 3D voxel grid, got {solid.shape}")
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    nx, ny, nz = solid.shape
    field = _rosette_mask_field(solid)

    key_blocks: list[np.ndarray] = []  # each (M, 4 corners, 4 = gridx,gridy,gridz,comp)
    cell_blocks: list[np.ndarray] = []
    axis_blocks: list[np.ndarray] = []
    side_blocks: list[np.ndarray] = []

    for axis in range(3):
        for side in (0, 1):
            cells = np.argwhere(_exposed_cells(solid, axis, side))  # (M, 3)
            if len(cells) == 0:
                continue
            keys = np.empty((len(cells), 4, 4), dtype=np.int64)
            for ci, (off, octant) in enumerate(_face_corner_specs(axis, side)):
                pts = cells + np.asarray(off, dtype=np.int64)        # grid corners (M, 3)
                masks = field[pts[:, 0], pts[:, 1], pts[:, 2]]       # rosette mask (M,)
                keys[:, ci, :3] = pts
                keys[:, ci, 3] = _COMP_TABLE[masks, octant]          # owner's component
            key_blocks.append(keys)
            cell_blocks.append(cells)
            axis_blocks.append(np.full(len(cells), axis, dtype=np.int64))
            side_blocks.append(np.full(len(cells), side, dtype=np.int64))

    if not key_blocks:
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 4), dtype=np.int64),
            {"cell": np.zeros((0, 3), int), "axis": np.zeros(0, int),
             "side": np.zeros(0, int), "vmask": np.zeros((0, 6), bool)},
        )

    keys = np.concatenate(key_blocks, axis=0)        # (F, 4, 4)
    n_faces = len(keys)
    ukeys, inv = np.unique(keys.reshape(n_faces * 4, 4), axis=0, return_inverse=True)
    faces = inv.ravel().reshape(n_faces, 4)
    gk = ukeys[:, :3]                                # integer grid coords of each weld vertex
    verts = origin + gk * spacing                    # weld key → world vertex

    # exact box-face membership from the integer key: a grid corner sits on the
    # min plane of axis ``ax`` iff its coord is 0, on the max plane iff it equals
    # the grid extent (solid.shape[ax]). No float tolerance enters.
    vmask = np.zeros((len(gk), 6), dtype=bool)
    for ax in range(3):
        vmask[:, 2 * ax] = gk[:, ax] == 0
        vmask[:, 2 * ax + 1] = gk[:, ax] == solid.shape[ax]

    prov = {
        "cell": np.concatenate(cell_blocks, axis=0),
        "axis": np.concatenate(axis_blocks, axis=0),
        "side": np.concatenate(side_blocks, axis=0),
        "vmask": vmask,
    }
    return verts, faces, prov


# --- sanity demo -------------------------------------------------------------
if __name__ == "__main__":
    from collections import Counter

    def report(name: str, V: np.ndarray, F: np.ndarray) -> None:
        edges: Counter = Counter()
        par = list(range(len(V)))

        def find(x: int) -> int:
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for q in F:
            for a, b in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
                edges[(min(a, b), max(a, b))] += 1
                par[find(a)] = find(b)
        mx = max(edges.values()) if edges else 0
        bodies = len({find(int(v)) for q in F for v in q})
        ok = "MANIFOLD" if mx <= 2 else "NON-MANIFOLD"
        print(f"{name:18} verts={len(V):3d} faces={len(F):3d} bodies={bodies} "
              f"max_faces/edge={mx} {ok}")

    s = np.zeros((3, 3, 3), int); s[1, 1, 1] = 1
    report("single voxel", *cuberille_mesh(s)[:2])

    s = np.zeros((4, 4, 4), int); s[1, 1, 1] = 1; s[2, 2, 1] = 1   # edge-diagonal
    report("edge-diag pair", *cuberille_mesh(s)[:2])

    s = np.zeros((4, 4, 3), int); s[1, 1, 1] = s[2, 1, 1] = s[2, 2, 1] = 1
    report("L-tromino", *cuberille_mesh(s)[:2])
