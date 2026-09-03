"""
voxel2surf — meshing pipeline orchestrator.

cuberille+weld extraction → provenance labelling → constrained variational
fairing (implicit obstacle problem) → optional refine → validate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import corridor, fair
from .extract import cuberille_mesh
from .label import classify
from .validate import topo, validate


@dataclass
class Options:
    density_cutoff: float = 0.5
    target_faces: int | None = None
    sub_levels: int = 1
    max_sub_levels: int = 2
    smooth_order: str = "thin-plate"
    data_weight: float = 1e-2
    corridor_voxels: float = 0.5
    implicit_iters: int = 12
    on_fail: str = "warn"  # "warn" | "raise"


class Logger:
    """Echo to stdout and/or append to a log file."""

    def __init__(self, echo: bool = True, path: str | Path | None = None):
        self.echo = echo
        self._fh = open(path, "w") if path else None

    def __call__(self, msg: str = "") -> None:
        if self.echo:
            print(msg)
        if self._fh:
            self._fh.write(msg + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()


def mesh_surface(
    vox,
    *,
    origin=None,
    spacing=None,
    opts: Options | None = None,
    log: Logger | None = None,
):
    """Full pipeline. Returns ``(verts, faces, report)``.

    ``vox`` is a 3D occupancy / density array. Cells with value
    ``> opts.density_cutoff`` are solid. ``origin`` / ``spacing`` map voxel
    indices to world coordinates (defaults: origin 0, unit voxels).
    """
    opts = opts or Options()
    close_log = log is None
    log = log or Logger(echo=False)
    vox = np.asarray(vox)
    if vox.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {vox.shape}")
    origin = np.zeros(3, float) if origin is None else np.asarray(origin, float)
    spacing = np.ones(3, float) if spacing is None else np.asarray(spacing, float)
    if np.shape(spacing) == ():
        spacing = np.full(3, float(spacing))
    shape = np.asarray(vox.shape, int)

    t0 = time.perf_counter()
    log(f"  shape={tuple(int(x) for x in shape)}  solid_voxels={int(np.count_nonzero(vox > opts.density_cutoff)):,}")

    ts = time.perf_counter()
    verts, faces, prov = cuberille_mesh(
        vox, origin, spacing, density_cutoff=opts.density_cutoff
    )
    vmask = prov["vmask"]
    st = topo(verts, faces)
    log(
        f"  extract [cuberille+weld] — verts={st['verts']:,} faces={st['faces']:,} "
        f"bodies={st['bodies']} watertight={st['watertight']}  ({time.perf_counter() - ts:.2f}s)"
    )
    if len(faces) == 0:
        report = {
            "verts": 0,
            "faces": 0,
            "bodies": 0,
            "watertight": True,
            "gates_failed": ["empty-mesh"],
        }
        log("  empty mesh (no solid voxels / no exposed faces)")
        if close_log:
            log.close()
        return verts, faces, report

    def _box_pins(c, verts):
        v = len(verts)
        hi_box = origin + shape * spacing
        pinned = np.zeros((v, 3), dtype=bool)
        target = np.zeros((v, 3), dtype=float)
        for a in range(3):
            lo_m, hi_m = c.on_face[:, 2 * a], c.on_face[:, 2 * a + 1]
            pinned[:, a] = lo_m | hi_m
            target[lo_m, a] = origin[a]
            target[hi_m, a] = hi_box[a]
        return pinned, target

    def _fair(verts, faces, tag, mask=None):
        ts = time.perf_counter()
        c = classify(verts, origin, spacing, shape, face_mask=mask)
        pinned, target = _box_pins(c, verts)
        lo, hi = corridor.derive_bounds(verts, spacing, half_voxels=opts.corridor_voxels)
        verts = fair.solve_implicit(
            verts, faces, pinned, target, lo, hi,
            order=opts.smooth_order, iters=opts.implicit_iters, reg=opts.data_weight,
        )
        mode = f"{opts.smooth_order} corr={opts.corridor_voxels} data={opts.data_weight:g}"
        n_edge = int((c.tangent ** 2).sum(1).astype(bool).sum())
        log(
            f"  fair [{tag}] — verts={len(verts):,} faces={len(faces):,} "
            f"on_face={int((c.face_idx >= 0).sum()):,} on_edge={n_edge:,} "
            f"corner={int(c.fixed.sum()):,} [{mode}]  ({time.perf_counter() - ts:.2f}s)"
        )
        return verts

    verts = _fair(verts, faces, "native", mask=vmask)

    sub = 0

    def _refine(verts, faces, vmask):
        nonlocal sub
        ts = time.perf_counter()
        verts, faces, vmask = fair.subdivide(verts, faces, vmask)
        sub += 1
        log(
            f"  subdivide x1 -> verts={len(verts):,} faces={len(faces):,}  "
            f"({time.perf_counter() - ts:.2f}s)"
        )
        verts = _fair(verts, faces, f"sub{sub}", mask=vmask)
        return verts, faces, vmask

    for _ in range(opts.sub_levels):
        verts, faces, vmask = _refine(verts, faces, vmask)

    while (
        opts.target_faces is not None
        and len(fair.to_triangles(faces)) < opts.target_faces
        and sub < opts.max_sub_levels
    ):
        verts, faces, vmask = _refine(verts, faces, vmask)

    if opts.target_faces is not None and len(fair.to_triangles(faces)) > opts.target_faces:
        ts = time.perf_counter()
        verts, faces = fair.decimate_to(verts, faces, opts.target_faces)
        log(f"  decimate -> faces={len(faces):,}  ({time.perf_counter() - ts:.2f}s)")
        verts = _fair(verts, faces, "polish")

    report = validate(
        verts, faces, vox=vox > opts.density_cutoff, origin=origin, spacing=spacing,
        shape=shape, log=log, on_fail=opts.on_fail,
    )
    report["vertices"] = int(len(verts))
    report["faces"] = int(len(faces))
    log(
        f"  done — {len(verts):,} verts, {len(faces):,} faces  "
        f"({time.perf_counter() - t0:.2f}s total)"
    )
    if close_log:
        log.close()
    return verts, faces, report
