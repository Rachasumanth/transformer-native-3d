"""Unit tests for QEM Decimation, Normal Baker, and 10 Production Asset Verification Gates."""
from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from t3d.articulation import Skeleton
from t3d.asset_validator import validate_production_asset
from t3d.decimation import bake_tangent_normal_map, decimate_mesh
from t3d.doctor import main as doctor_main, run_doctor
from t3d.glb import pack_glb
from t3d.sampling import salient_edges


def _create_unwrapped_cube_glb(
    *,
    tex_size: int = 256,
    border_gutter_px: int = 8,
    inter_gutter_px: int = 8,
    skin_weights: np.ndarray | None = None,
    skeleton: Skeleton | None = None,
) -> bytes:
    """Helper to create a game-ready cube GLB with contiguous UV charts and guaranteed gutter."""
    from t3d.decimation import generate_uv_charts
    v = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ], dtype=np.float32)
    f = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)

    v_arr, f_arr, uv_arr, src_idx = generate_uv_charts(
        v, f, texture_size=tex_size, margin=inter_gutter_px, gutter=inter_gutter_px
    )

    mesh = trimesh.Trimesh(v_arr, f_arr, process=False)
    mesh.fix_normals()
    pbr = np.full((tex_size, tex_size, 6), [0.8, 0.8, 0.8, 0.2, 0.5, 1.0], dtype=np.float32)

    render_weights = None
    if skin_weights is not None:
        if len(skin_weights) == len(v):
            render_weights = skin_weights[src_idx]
        elif len(skin_weights) == len(v_arr):
            render_weights = skin_weights
        else:
            # Map by repeating / tile
            render_weights = np.repeat(skin_weights[:1], len(v_arr), axis=0)

    return pack_glb(
        mesh.vertices,
        mesh.faces,
        uv_arr,
        mesh.vertex_normals,
        pbr,
        source_up="y",
        skin_weights=render_weights,
        skeleton=skeleton,
    )


# =========================================================================
# Priority 2: QEM Decimation & Tangent Normal Map Baker Tests
# =========================================================================

def test_decimate_mesh_preserves_manifoldness_and_target_count():
    """Verify decimation reduces faces to target count while preserving watertight 2-manifold topology."""
    box = trimesh.creation.box().subdivide().subdivide().subdivide()
    orig_faces = len(box.faces)
    assert orig_faces == 768

    v_dec, f_dec = decimate_mesh(box.vertices, box.faces, target_faces=60, aggressive=False)
    assert len(f_dec) <= int(60 * 1.1)
    assert len(f_dec) < orig_faces

    # Verify watertight and 2-manifold topology
    m_dec = trimesh.Trimesh(v_dec, f_dec, process=False)
    assert m_dec.is_watertight
    assert m_dec.is_volume

    # Verify sharp feature edges (> 30 deg) along cube creases are preserved
    creases = salient_edges(v_dec, f_dec, threshold=np.pi / 6)
    assert len(creases) >= 12


def test_decimate_mesh_no_op_when_under_target():
    """Verify decimation is a clean no-op if the mesh is already below target count."""
    box = trimesh.creation.box()
    v_dec, f_dec = decimate_mesh(box.vertices, box.faces, target_faces=5000)
    assert len(f_dec) == len(box.faces)
    assert len(v_dec) == len(box.vertices)


def test_bake_tangent_normal_map_neutral_projection():
    """Verify normal baking between identical meshes produces neutral normal map (0.5, 0.5, 1.0)."""
    low = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    theta = np.arctan2(low.vertices[:, 1], low.vertices[:, 0])
    phi = np.arcsin(np.clip(low.vertices[:, 2] / np.linalg.norm(low.vertices, axis=1), -1, 1))
    uvs = np.column_stack([(theta + np.pi) / (2 * np.pi), (phi + np.pi / 2) / np.pi])
    uvs = np.clip(uvs, 0.05, 0.95)

    nmap = bake_tangent_normal_map(
        low.vertices, low.faces, low.vertices, low.faces, uvs, texture_size=128
    )
    assert nmap.shape == (128, 128, 3)
    assert nmap.dtype == np.float32

    # In neutral normal map, Z channel (normal axis) is ~1.0
    assert np.all(nmap[..., 2] >= 0.9)

    # Test as_uint8=True
    nmap_u8 = bake_tangent_normal_map(
        low.vertices, low.faces, low.vertices, low.faces, uvs, texture_size=128, as_uint8=True
    )
    assert nmap_u8.dtype == np.uint8
    assert nmap_u8.shape == (128, 128, 3)
    assert np.all(nmap_u8[..., 2] >= 225)


def test_bake_tangent_normal_map_captures_surface_displacement():
    """Verify normal baking projects detail from deformed high-poly mesh onto low-poly mesh."""
    high = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    high_v = high.vertices.copy()
    # Add high-frequency displacement bump
    high_v += 0.1 * np.sin(4 * high_v[:, 0:1]) * high_v

    low = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    theta = np.arctan2(low.vertices[:, 1], low.vertices[:, 0])
    phi = np.arcsin(np.clip(low.vertices[:, 2] / np.linalg.norm(low.vertices, axis=1), -1, 1))
    uvs = np.column_stack([(theta + np.pi) / (2 * np.pi), (phi + np.pi / 2) / np.pi])
    uvs = np.clip(uvs, 0.05, 0.95)

    nmap = bake_tangent_normal_map(high_v, high.faces, low.vertices, low.faces, uvs, texture_size=128)
    assert np.any(np.abs(nmap[..., 0] - 0.5) > 0.05)


# =========================================================================
# Priority 3: 10 Production Asset Verification Gates Tests
# =========================================================================

def test_validate_production_asset_clean_valid_asset():
    """Verify that a compliant game-ready asset passes all 10 production gates."""
    glb_bytes = _create_unwrapped_cube_glb()
    report = validate_production_asset(glb_bytes, max_tris=15000)

    assert report["ok"] is True
    assert report["passed"] is True
    assert len(report["gates"]) == 10
    for gate_name, gate in report["gates"].items():
        assert gate["ok"] is True, f"Gate failed unexpectedly: {gate_name}: {gate['detail']}"
    assert report["summary"] == "All 10 production gates PASSED"


def test_validate_production_asset_valid_skinned_asset():
    """Verify that a compliant skinned asset passes bone containment and skinning gates."""
    v_count = 24
    skel = Skeleton(
        joints=np.array([[0.0, -0.3, 0.0], [0.0, 0.3, 0.0]], dtype=np.float32),
        parents=np.array([-1, 0], dtype=np.int32),
    )
    weights = np.full((v_count, 2), 0.5, dtype=np.float32)

    glb_bytes = _create_unwrapped_cube_glb(skin_weights=weights, skeleton=skel)
    report = validate_production_asset(glb_bytes, max_tris=15000)

    assert report["ok"] is True
    assert report["gates"]["bone_containment"]["ok"] is True
    assert report["gates"]["skinning_compliance"]["ok"] is True


def test_validate_production_asset_oversized_triangle_count():
    """Gate 4 Blocker: Fail assets exceeding max_tris limit."""
    glb_bytes = _create_unwrapped_cube_glb()
    # Cube has 12 triangles; setting max_tris=10 will trigger Gate 4 blocker
    report = validate_production_asset(glb_bytes, max_tris=10)

    assert report["ok"] is False
    assert report["gates"]["triangle_count"]["ok"] is False
    assert "limit: <= 10" in report["gates"]["triangle_count"]["detail"]


def test_validate_production_asset_non_manifold():
    """Gate 1 Blocker: Fail assets containing non-manifold geometry."""
    # Create cube with a 3-sheet non-manifold edge
    m = trimesh.creation.box()
    nm_faces = np.vstack([m.faces, [0, 1, 2]])  # Duplicate face creates non-manifold edge
    nm_uvs = np.zeros((len(m.vertices), 2), dtype=np.float32)
    pbr = np.full((64, 64, 6), 0.5, dtype=np.float32)

    nm_glb = pack_glb(m.vertices, nm_faces, nm_uvs, m.vertex_normals, pbr, source_up="y")
    report = validate_production_asset(nm_glb)

    assert report["ok"] is False
    assert report["gates"]["non_manifold"]["ok"] is False
    assert "non-manifold" in report["gates"]["non_manifold"]["detail"]


def test_validate_production_asset_unnormalized_skinning():
    """Gate 9 Blocker: Fail skinned assets where bone weights sum != 1.0."""
    v_count = 24
    skel = Skeleton(
        joints=np.array([[0.0, -0.3, 0.0], [0.0, 0.3, 0.0]], dtype=np.float32),
        parents=np.array([-1, 0], dtype=np.int32),
    )
    weights = np.full((v_count, 2), 0.5, dtype=np.float32)
    glb_bytes = _create_unwrapped_cube_glb(skin_weights=weights, skeleton=skel)

    # Corrupt WEIGHTS_0 in BIN chunk by scaling by 0.5
    from t3d.asset_validator import _parse_glb_chunks
    doc, bin_data = _parse_glb_chunks(glb_bytes)
    w_acc_id = doc["meshes"][0]["primitives"][0]["attributes"]["WEIGHTS_0"]
    w_acc = doc["accessors"][w_acc_id]
    bv = doc["bufferViews"][w_acc["bufferView"]]
    offset = bv["byteOffset"] + w_acc.get("byteOffset", 0)
    byte_len = w_acc["count"] * 4 * 4
    w_raw = np.frombuffer(bin_data[offset : offset + byte_len], dtype=np.float32).copy()
    w_raw *= 0.5  # sum is now 0.5, violating sum(w) == 1.0 +- 1e-5
    bin_corrupt = bytearray(bin_data)
    bin_corrupt[offset : offset + byte_len] = w_raw.tobytes()

    json_bytes = json.dumps(doc).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_corrupt += b"\x00" * ((4 - len(bin_corrupt) % 4) % 4)
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_corrupt)
    corrupt_glb = (
        struct.pack("<4sII", b"glTF", 2, total_len)
        + struct.pack("<I4s", len(json_bytes), b"JSON")
        + json_bytes
        + struct.pack("<I4s", len(bin_corrupt), b"BIN\x00")
        + bytes(bin_corrupt)
    )

    report = validate_production_asset(corrupt_glb)
    assert report["ok"] is False
    assert report["gates"]["skinning_compliance"]["ok"] is False
    assert "sum(w) err" in report["gates"]["skinning_compliance"]["detail"]


def test_validate_production_asset_unskinned_vertices():
    """Gate 9 Blocker: Fail skinned assets with unskinned vertices (sum == 0)."""
    v_count = 24
    skel = Skeleton(
        joints=np.array([[0.0, -0.3, 0.0], [0.0, 0.3, 0.0]], dtype=np.float32),
        parents=np.array([-1, 0], dtype=np.int32),
    )
    weights = np.full((v_count, 2), 0.5, dtype=np.float32)
    glb_bytes = _create_unwrapped_cube_glb(skin_weights=weights, skeleton=skel)

    from t3d.asset_validator import _parse_glb_chunks
    doc, bin_data = _parse_glb_chunks(glb_bytes)
    w_acc_id = doc["meshes"][0]["primitives"][0]["attributes"]["WEIGHTS_0"]
    w_acc = doc["accessors"][w_acc_id]
    bv = doc["bufferViews"][w_acc["bufferView"]]
    offset = bv["byteOffset"] + w_acc.get("byteOffset", 0)
    byte_len = w_acc["count"] * 4 * 4
    w_raw = np.frombuffer(bin_data[offset : offset + byte_len], dtype=np.float32).copy()
    # Zero out vertex 0
    w_raw[:4] = 0.0
    bin_corrupt = bytearray(bin_data)
    bin_corrupt[offset : offset + byte_len] = w_raw.tobytes()

    json_bytes = json.dumps(doc).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_corrupt += b"\x00" * ((4 - len(bin_corrupt) % 4) % 4)
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_corrupt)
    corrupt_glb = (
        struct.pack("<4sII", b"glTF", 2, total_len)
        + struct.pack("<I4s", len(json_bytes), b"JSON")
        + json_bytes
        + struct.pack("<I4s", len(bin_corrupt), b"BIN\x00")
        + bytes(bin_corrupt)
    )

    report = validate_production_asset(corrupt_glb)
    assert report["ok"] is False
    assert report["gates"]["skinning_compliance"]["ok"] is False
    assert "unskinned verts=1" in report["gates"]["skinning_compliance"]["detail"]


def test_validate_production_asset_bone_outside_aabb():
    """Gate 8 Blocker: Fail skinned assets with bone joints outside mesh AABB."""
    v_count = 24
    # Bounding box is [-0.5, 0.5]^3; bone joint at Y=5.0 is far outside AABB
    skel = Skeleton(
        joints=np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float32),
        parents=np.array([-1, 0], dtype=np.int32),
    )
    weights = np.full((v_count, 2), 0.5, dtype=np.float32)
    glb_bytes = _create_unwrapped_cube_glb(skin_weights=weights, skeleton=skel)

    report = validate_production_asset(glb_bytes)
    assert report["ok"] is False
    assert report["gates"]["bone_containment"]["ok"] is False
    assert "bones outside mesh AABB" in report["gates"]["bone_containment"]["detail"]


def test_validate_production_asset_uv_overlap():
    """Gate 6 Blocker: Fail assets with overlapping UV islands."""
    # Create two disjoint quads (2 separate islands) placed at the exact same overlapping UV square [0.2, 0.8]^2
    v = np.array([
        # Quad 1
        [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
        # Quad 2
        [2, -1, 0], [4, -1, 0], [4, 1, 0], [2, 1, 0],
    ], dtype=np.float32)
    f = np.array([
        [0, 1, 2], [2, 3, 0],
        [4, 5, 6], [6, 7, 4],
    ], dtype=np.int32)
    uvs = np.array([
        [0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8],
        [0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8],
    ], dtype=np.float32)
    n = np.array([[0, 0, 1]] * 8, dtype=np.float32)
    pbr = np.full((128, 128, 6), 0.5, dtype=np.float32)
    overlap_glb = pack_glb(v, f, uvs, n, pbr, source_up="y")

    report = validate_production_asset(overlap_glb)
    assert report["ok"] is False
    assert report["gates"]["uv_overlap"]["ok"] is False
    assert "overlapping pixels" in report["gates"]["uv_overlap"]["detail"]


def test_validate_production_asset_corrupted_gltf():
    """Gate 10 Blocker: Fail assets with truncated or corrupted binary glTF chunks."""
    corrupt_bytes = b"not a valid gltf binary file at all"
    report = validate_production_asset(corrupt_bytes)
    assert report["ok"] is False
    assert report["gates"]["gltf_compliance"]["ok"] is False


# =========================================================================
# Doctor CLI Integration Tests
# =========================================================================

def test_doctor_cli_asset_option_pass_and_fail():
    """Verify t3d doctor --asset runs production validation and returns exit code 0 on PASS, 1 on FAIL."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Valid asset -> Exit Code 0
        valid_glb = tmp_path / "valid.glb"
        valid_glb.write_bytes(_create_unwrapped_cube_glb())
        code_pass = doctor_main(["--asset", str(valid_glb)])
        assert code_pass == 0

        # JSON mode
        report_pass = run_doctor(asset=valid_glb)
        assert report_pass["ok"] is True
        assert len(report_pass["checks"]) == 10

        # 2. Corrupt / failing asset -> Exit Code 1
        fail_glb = tmp_path / "fail.glb"
        fail_glb.write_bytes(b"corrupted binary data")
        code_fail = doctor_main(["--asset", str(fail_glb)])
        assert code_fail == 1

        # 3. Non-existent file -> Exit Code 1
        code_missing = doctor_main(["--asset", str(tmp_path / "missing.glb")])
        assert code_missing == 1

        # 4. Require-rig flag on static asset -> Exit Code 1 (fails Gates 8 & 9)
        code_rig_fail = doctor_main(["--asset", str(valid_glb), "--require-rig"])
        assert code_rig_fail == 1

        report_rig_fail = run_doctor(asset=valid_glb, require_rig=True)
        assert report_rig_fail["ok"] is False
        assert report_rig_fail["asset_report"]["gates"]["bone_containment"]["ok"] is False
        assert report_rig_fail["asset_report"]["gates"]["skinning_compliance"]["ok"] is False


def test_skin_inverse_bind_matrices_validation():
    """Verify Gate 10 enforces non-singular, non-NaN inverseBindMatrices matching joint count."""
    skel = Skeleton(
        joints=np.array([[0.0, -0.3, 0.0], [0.0, 0.3, 0.0]], dtype=np.float32),
        parents=np.array([-1, 0], dtype=np.int32),
    )
    weights = np.full((8, 2), 0.5, dtype=np.float32)
    glb_bytes = _create_unwrapped_cube_glb(skin_weights=weights, skeleton=skel)

    # Compliant skinned asset passes Gate 10 and passes when require_rig=True
    report = validate_production_asset(glb_bytes, require_rig=True)
    assert report["ok"] is True
    assert report["gates"]["gltf_compliance"]["ok"] is True
    assert report["gates"]["bone_containment"]["ok"] is True
    assert report["gates"]["skinning_compliance"]["ok"] is True
