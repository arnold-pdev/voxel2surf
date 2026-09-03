"""Load 3D occupancy arrays and write binary STL."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fair import to_triangles

_NPZ_KEYS = ("binaries", "occupancy", "vox", "rho", "density", "volume")


def load_volume(
    path: str | Path,
    *,
    key: str | None = None,
    member: int | None = None,
) -> np.ndarray:
    """Load a 3D volume from ``.npy`` or ``.npz``.

    For ``.npz``, ``key`` selects the array (default: first well-known name,
    else the first file). A 4D array of shape ``(members, nx, ny, nz)`` is
    sliced with ``member`` (required in that case).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        z = np.load(path)
        if key is None:
            for candidate in _NPZ_KEYS:
                if candidate in z.files:
                    key = candidate
                    break
            else:
                key = z.files[0]
        if key not in z.files:
            raise KeyError(f"{path} has no array {key!r}; files={list(z.files)}")
        arr = z[key]
    else:
        raise ValueError(f"unsupported input {path} (use .npy or .npz)")

    arr = np.asarray(arr)
    if arr.ndim == 4:
        if member is None:
            raise ValueError(
                f"{path} is 4D {arr.shape}; pass --member to pick a slice"
            )
        arr = arr[int(member)]
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {arr.shape} from {path}")
    return arr


def demo_volume(n: int = 24) -> np.ndarray:
    """Unit sphere of radius ``0.32 n`` on an ``n³`` grid (for ``--demo``)."""
    c = 0.5 * (n - 1)
    i = np.arange(n, dtype=float)
    x, y, z = np.meshgrid(i, i, i, indexing="ij")
    r2 = (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2
    return (r2 <= (0.32 * n) ** 2).astype(np.uint8)


def write_stl(path: str | Path, verts: np.ndarray, faces: np.ndarray) -> Path:
    """Write a binary STL (geometry only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tris = to_triangles(np.asarray(faces))
    v = np.asarray(verts, dtype=np.float64)
    if len(tris) == 0:
        header = b"voxel2surf empty".ljust(80, b"\0")
        with path.open("wb") as fh:
            fh.write(header)
            fh.write(np.uint32(0).tobytes())
        return path
    v0, v1, v2 = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    dt = np.dtype(
        [
            ("n", "<f4", 3),
            ("v0", "<f4", 3),
            ("v1", "<f4", 3),
            ("v2", "<f4", 3),
            ("attr", "<u2"),
        ]
    )
    rec = np.empty(len(tris), dtype=dt)
    rec["n"] = nrm
    rec["v0"] = v0
    rec["v1"] = v1
    rec["v2"] = v2
    rec["attr"] = 0
    header = b"voxel2surf binary STL".ljust(80, b"\0")
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(np.uint32(len(tris)).tobytes())
        rec.tofile(fh)
    return path
