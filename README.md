# voxel2surf

Binary voxel occupancy → a **smooth, watertight STL**.

Spin-off of the surface-extraction stage from [TOSA](https://github.com/arnold-pdev/TOSA): cuberille meshing plus constrained thin-plate fairing. No FEA, no NITO I/O, no BC tagging — just occupancy in, STL out.

## Install

```bash
pip install -e .
```

Python 3.10+, `numpy`, `scipy`. Optional `pip install -e ".[decimate]"` if you want `--target-faces` (pyvista quadric collapse).

## CLI

```bash
# 3D .npy (any numeric array; values > 0.5 are solid)
voxel2surf occupancy.npy -o part.stl

# .npz — auto-picks binaries / occupancy / vox / rho if present
voxel2surf ensemble.npz --key binaries --member 0 -o part.stl --scale 2

# smoke test (24³ sphere)
voxel2surf --demo -o demo.stl
```

`--scale` is uniform voxel spacing in output units (TOSA EDS surfaces used `--scale 2`). `--sub-levels 1` (default) does one midpoint subdivision after the native-resolution fair.

## Python

```python
import numpy as np
from voxel2surf import Options, mesh_surface, write_stl

vox = np.load("occupancy.npy")          # (nx, ny, nz)
verts, faces, report = mesh_surface(
    vox,
    spacing=2.0,
    opts=Options(sub_levels=1, density_cutoff=0.5),
)
write_stl("part.stl", verts, faces)
print(report["watertight"], report["gates_failed"])
```

`mesh_surface` returns triangle or quad faces (quads after extraction, triangles after subdivision). `write_stl` triangulates as needed.

## Method (short)

1. **Extract** — one quad per exposed voxel face (cuberille). Vertices weld only across **6-connected** (face-adjacent) solid, so diagonal contacts stay as separate sheets.
2. **Label** — box-face vertices may only slide in that face / along that edge; corners are pinned. Labels come from integer grid keys, not float tolerances.
3. **Fair** — minimize thin-plate (bi-Laplacian) energy plus a soft pull to the cuberille positions, subject to those pins and a ±½-voxel occupancy corridor. Solved per coordinate as a bound-constrained QP (primal-dual active set). This is an obstacle problem, not Laplacian smoothing: it does not shrink thin members.
4. **Refine** — optional midpoint subdivision (and optional decimation), with the same fairing at each level.
5. **Validate** — watertightness, self-intersection, volume-fraction delta.

Defaults match the TOSA production path (`thin-plate`, `data-weight=1e-2`, `corridor-voxels=0.5`, one subdivision).

## Layout

```
src/voxel2surf/
  extract.py    cuberille + 6-connected weld
  label.py      box-face / edge / corner pins
  corridor.py   ±½-voxel containment
  fair.py       implicit thin-plate solve, subdivide
  validate.py   watertight / self-intersection gates
  pipeline.py   mesh_surface()
  io.py         npy/npz in, binary STL out
  cli.py        voxel2surf ...
```
