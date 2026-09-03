"""
voxel2surf CLI — binary occupancy (.npy / .npz) → STL.

    voxel2surf occupancy.npy -o part.stl
    voxel2surf ensemble.npz --key binaries --member 0 -o part.stl --scale 2
    voxel2surf --demo -o demo.stl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .io import demo_volume, load_volume, write_stl
from .pipeline import Logger, Options, mesh_surface


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="voxel2surf: binary voxels → smooth watertight STL "
        "(cuberille + constrained variational fairing)",
    )
    p.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="3D occupancy .npy, or .npz (see --key / --member)",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="mesh a built-in sphere instead of reading a file",
    )
    p.add_argument("-o", "--output", type=Path, required=True, help="output .stl")
    p.add_argument("--key", type=str, default=None, help="array name inside .npz")
    p.add_argument("--member", type=int, default=None, help="slice a 4D .npz array")
    p.add_argument(
        "--cutoff",
        type=float,
        default=0.5,
        help="solid if value > cutoff (default 0.5)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="uniform voxel spacing in output units (default 1)",
    )
    p.add_argument("--origin", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))

    g = p.add_argument_group("mesh")
    g.add_argument("--sub-levels", type=int, default=1, help="midpoint subdivisions (each ×4 faces)")
    g.add_argument("--target-faces", type=int, default=None, help="decimate to this triangle count (needs pyvista)")
    g.add_argument("--max-sub-levels", type=int, default=2)

    gi = p.add_argument_group("fairing")
    gi.add_argument("--smooth-order", choices=("membrane", "thin-plate"), default="thin-plate")
    gi.add_argument("--data-weight", type=float, default=1e-2, help="soft pull to cuberille positions (anti-shrink)")
    gi.add_argument("--corridor-voxels", type=float, default=0.5, help="dual-cell containment half-width")
    gi.add_argument("--implicit-iters", type=int, default=12, help="primal-dual active-set rounds")
    g.add_argument("--on-fail", choices=("warn", "raise"), default="warn")

    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = _parse_args(argv)
    if a.demo == bool(a.input):
        print("specify exactly one of INPUT or --demo", file=sys.stderr)
        return 2

    log = Logger(echo=not a.quiet, path=a.log_file)
    if a.demo:
        vox = demo_volume()
        log("voxel2surf  --demo  (24³ sphere)")
    else:
        vox = load_volume(a.input, key=a.key, member=a.member)
        log(f"voxel2surf  {a.input}")

    origin = np.zeros(3, float) if a.origin is None else np.asarray(a.origin, float)
    spacing = np.full(3, float(a.scale), dtype=float)
    opts = Options(
        density_cutoff=a.cutoff,
        target_faces=a.target_faces,
        sub_levels=a.sub_levels,
        max_sub_levels=a.max_sub_levels,
        smooth_order=a.smooth_order,
        corridor_voxels=a.corridor_voxels,
        implicit_iters=a.implicit_iters,
        data_weight=a.data_weight,
        on_fail=a.on_fail,
    )
    verts, faces, report = mesh_surface(
        vox, origin=origin, spacing=spacing, opts=opts, log=log,
    )
    write_stl(a.output, verts, faces)
    log(f"  wrote {a.output}")
    log.close()
    return 1 if report.get("gates_failed") else 0


if __name__ == "__main__":
    sys.exit(main())
