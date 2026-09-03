import unittest

import numpy as np

from voxel2surf.extract import cuberille_mesh
from voxel2surf.io import demo_volume, write_stl
from voxel2surf.pipeline import Logger, Options, mesh_surface
from voxel2surf.validate import topo


class ExtractTests(unittest.TestCase):
    def test_single_voxel_six_quads(self):
        s = np.zeros((3, 3, 3), int)
        s[1, 1, 1] = 1
        verts, faces, _ = cuberille_mesh(s)
        self.assertEqual(len(faces), 6)
        self.assertTrue(topo(verts, faces)["watertight"])


class PipelineTests(unittest.TestCase):
    def test_demo_sphere_watertight(self):
        vox = demo_volume(12)
        verts, faces, report = mesh_surface(
            vox,
            opts=Options(sub_levels=0, implicit_iters=4),
            log=Logger(echo=False),
        )
        self.assertGreater(len(faces), 0)
        self.assertTrue(report["watertight"])
        self.assertFalse(report.get("gates_failed"))

    def test_write_stl_roundtrip_header(self):
        import tempfile
        from pathlib import Path

        s = np.zeros((3, 3, 3), int)
        s[1, 1, 1] = 1
        verts, faces, _ = cuberille_mesh(s)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.stl"
            write_stl(path, verts, faces)
            raw = path.read_bytes()
            ntri = int.from_bytes(raw[80:84], "little")
            self.assertEqual(ntri, 12)  # 6 quads → 12 tris
            self.assertEqual(len(raw), 84 + ntri * 50)


if __name__ == "__main__":
    unittest.main()
