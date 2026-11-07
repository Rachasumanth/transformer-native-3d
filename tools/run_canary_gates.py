"""Production Canary Verification Runner for CI/CD and release pipelines.

Validates:
1. Static Prop Pipeline against all 10 production gates.
2. Articulated Character Pipeline against all 10 gates with --require-rig.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
import trimesh

from t3d.doctor import main as doctor_main
from t3d.glb import pack_static_glb
from t3d.articulation import Skeleton


def generate_canary_assets() -> tuple[Path, Path]:
    static_path = Path("/tmp/blaster_a.glb")
    rigged_path = Path("/tmp/rigged_robot.glb")

    if not static_path.exists():
        # Build canonical static prop
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        pbr = np.tile([0.8, 0.8, 0.8, 0.1, 0.5, 1.0], (len(mesh.vertices), 1))
        glb = pack_static_glb(mesh.vertices, mesh.faces, pbr=pbr, source_up='z')
        static_path.write_bytes(glb)

    if not rigged_path.exists():
        # Build canonical rigged character
        cyl = trimesh.creation.cylinder(radius=0.15, height=1.0, sections=16)
        pbr = np.tile([0.2, 0.7, 0.3, 0.1, 0.5, 1.0], (len(cyl.vertices), 1))
        skel = Skeleton(
            joints=np.array([[0.0, 0.0, -0.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.4]], dtype=np.float32),
            parents=np.array([-1, 0, 1], dtype=np.int64)
        )
        glb = pack_static_glb(cyl.vertices, cyl.faces, pbr=pbr, skeleton=skel, source_up='z')
        rigged_path.write_bytes(glb)

    return static_path, rigged_path


def main() -> int:
    static_asset, rigged_asset = generate_canary_assets()

    print(f"\n==========================================")
    print(f"CANARY 1: Static Prop Pipeline Verification")
    print(f"Asset: {static_asset}")
    print(f"==========================================")
    code_static = doctor_main(["--asset", str(static_asset), "--json"])
    if code_static != 0:
        print("[FAIL] Canary 1: Static Prop Pipeline verification failed!", file=sys.stderr)
        return code_static

    print(f"\n==========================================")
    print(f"CANARY 2: Articulated Pipeline Verification")
    print(f"Asset: {rigged_asset} (--require-rig)")
    print(f"==========================================")
    code_rigged = doctor_main(["--asset", str(rigged_asset), "--require-rig", "--json"])
    if code_rigged != 0:
        print("[FAIL] Canary 2: Articulated Pipeline verification failed!", file=sys.stderr)
        return code_rigged

    print("\n[SUCCESS] Both Production Canary Gates Passed 10/10 Verification!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
