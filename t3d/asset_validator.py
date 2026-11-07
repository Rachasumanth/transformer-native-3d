"""10 Production Asset Verification Gates for T3D Game-Ready Assets.

Gates:
1. Non-manifold edges/vertices == 0
2. Degenerate triangles (area <= 1e-8) == 0
3. Watertightness (is_watertight == True, 0 open boundary loops)
4. Triangle count <= max_tris (default 15,000)
5. Aspect ratio: sliver triangles (ratio >= 20:1) <= 5%
6. UV overlap: 0.0% overlap between islands
7. UV bounds in [0, 1]^2, gutter >= 2 px
8. Bone containment: 0 bones outside mesh AABB
9. Skinning: max 4 bones per vertex, sum(w) == 1.0 +- 1e-5, 0 unskinned vertices
10. glTF compliance: valid glTF 2.0 binary chunks (JSON + BIN), no corrupted accessors
"""
from __future__ import annotations

import io
import json
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


def _parse_glb_chunks(data: bytes | bytearray) -> tuple[dict, bytes]:
    """Parse glTF binary chunks and return JSON document and BIN buffer."""
    if not isinstance(data, (bytes, bytearray)) or len(data) < 20:
        raise ValueError("Invalid GLB binary: minimum header size is 20 bytes")
    magic, version, total_len = struct.unpack_from("<4sII", data)
    if magic != b"glTF":
        raise ValueError(f"Invalid GLB magic header: {magic!r} != b'glTF'")
    if version != 2:
        raise ValueError(f"Invalid glTF version: {version} != 2")
    if total_len != len(data):
        raise ValueError(f"GLB length header mismatch: {total_len} != {len(data)}")

    cursor = 12
    json_chunk = None
    bin_chunk = b""
    while cursor < len(data):
        if cursor + 8 > len(data):
            raise ValueError("Truncated chunk header in GLB")
        chunk_len, chunk_type = struct.unpack_from("<I4s", data, cursor)
        cursor += 8
        end = cursor + chunk_len
        if end > len(data):
            raise ValueError(f"Truncated chunk data in GLB: expected {chunk_len} bytes")
        chunk_bytes = bytes(data[cursor:end])
        cursor = end
        if chunk_type == b"JSON":
            if json_chunk is not None:
                raise ValueError("Duplicate JSON chunk in GLB")
            json_chunk = json.loads(chunk_bytes.decode("utf-8"))
        elif chunk_type == b"BIN\x00":
            bin_chunk = chunk_bytes

    if json_chunk is None:
        raise ValueError("GLB missing required JSON chunk")
    return json_chunk, bin_chunk


def _read_accessor(
    doc: dict, bin_chunk: bytes, acc_id: int
) -> tuple[np.ndarray, dict]:
    """Read typed numpy array from glTF accessor."""
    accessors = doc.get("accessors", [])
    buffer_views = doc.get("bufferViews", [])
    if not (0 <= acc_id < len(accessors)):
        raise ValueError(f"Invalid accessor index {acc_id}")
    acc = accessors[acc_id]
    if "bufferView" not in acc:
        raise ValueError(f"Accessor {acc_id} missing bufferView")
    bv_id = acc["bufferView"]
    if not (0 <= bv_id < len(buffer_views)):
        raise ValueError(f"Invalid bufferView index {bv_id}")
    bv = buffer_views[bv_id]

    offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    comp_type = acc["componentType"]
    acc_type = acc["type"]

    type_counts = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    if acc_type not in type_counts:
        raise ValueError(f"Unsupported accessor type: {acc_type}")
    dim = type_counts[acc_type]
    total_items = count * dim

    dtype_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    if comp_type not in dtype_map:
        raise ValueError(f"Unsupported component type: {comp_type}")
    dtype = dtype_map[comp_type]
    byte_len = total_items * dtype().itemsize
    if offset + byte_len > len(bin_chunk):
        raise ValueError(
            f"Accessor data exceeds BIN chunk: {offset + byte_len} > {len(bin_chunk)}"
        )

    arr = np.frombuffer(
        bin_chunk[offset : offset + byte_len], dtype=dtype, count=total_items
    ).reshape(count, dim)
    return arr, acc


def count_non_manifold_elements(
    positions: np.ndarray, faces: np.ndarray
) -> tuple[int, int]:
    """Count non-manifold edges and non-manifold vertices after exact spatial welding."""
    _, inv = np.unique(np.round(positions, decimals=6), axis=0, return_inverse=True)
    wf = inv[faces]

    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    v_stars: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for tri in wf:
        if tri[0] == tri[1] or tri[1] == tri[2] or tri[2] == tri[0]:
            continue
        e1 = tuple(sorted((int(tri[0]), int(tri[1]))))
        e2 = tuple(sorted((int(tri[1]), int(tri[2]))))
        e3 = tuple(sorted((int(tri[2]), int(tri[0]))))
        edge_counts[e1] += 1
        edge_counts[e2] += 1
        edge_counts[e3] += 1
        v_stars[int(tri[0])].append((int(tri[1]), int(tri[2])))
        v_stars[int(tri[1])].append((int(tri[2]), int(tri[0])))
        v_stars[int(tri[2])].append((int(tri[0]), int(tri[1])))

    non_manifold_edges = sum(1 for c in edge_counts.values() if c > 2)

    non_manifold_verts = 0
    for v, opposite_edges in v_stars.items():
        if not opposite_edges:
            continue
        adj = defaultdict(set)
        nodes = set()
        for a, b in opposite_edges:
            adj[a].add(b)
            adj[b].add(a)
            nodes.add(a)
            nodes.add(b)

        if any(len(nbrs) > 2 for nbrs in adj.values()):
            non_manifold_verts += 1
            continue

        visited = set()
        components = 0
        for node in nodes:
            if node not in visited:
                components += 1
                q = [node]
                visited.add(node)
                while q:
                    curr = q.pop()
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            q.append(nbr)
        if components > 1:
            non_manifold_verts += 1

    return non_manifold_edges, non_manifold_verts


def count_boundary_loops(
    positions: np.ndarray, faces: np.ndarray
) -> tuple[int, int]:
    """Count open boundary edges and closed boundary loops on welded mesh."""
    _, inv = np.unique(np.round(positions, decimals=6), axis=0, return_inverse=True)
    wf = inv[faces]

    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for tri in wf:
        if tri[0] == tri[1] or tri[1] == tri[2] or tri[2] == tri[0]:
            continue
        e1 = tuple(sorted((int(tri[0]), int(tri[1]))))
        e2 = tuple(sorted((int(tri[1]), int(tri[2]))))
        e3 = tuple(sorted((int(tri[2]), int(tri[0]))))
        edge_counts[e1] += 1
        edge_counts[e2] += 1
        edge_counts[e3] += 1

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    if not boundary_edges:
        return 0, 0

    adj = defaultdict(set)
    for u, v in boundary_edges:
        adj[u].add(v)
        adj[v].add(u)

    visited = set()
    loops = 0
    for node in adj:
        if node not in visited:
            loops += 1
            q = [node]
            visited.add(node)
            while q:
                curr = q.pop()
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        q.append(nbr)

    return len(boundary_edges), loops


def validate_production_asset(
    glb_bytes: bytes | bytearray | Path | str,
    max_tris: int = 15000,
    require_skin: bool = False,
    require_rig: bool = False,
) -> dict[str, Any]:
    """Validate a 3D asset against the 10 Production Verification Gates.

    Parameters
    ----------
    glb_bytes : bytes, bytearray, Path, or str
        Raw GLB binary data or file path to GLB asset.
    max_tris : int, default 15000
        Maximum allowable triangle count for game-ready production.
    require_skin, require_rig : bool, default False
        Whether the asset must contain skinning and skeleton data.

    Returns
    -------
    report : dict
        Detailed report with overall status ('ok'), per-gate results ('gates'),
        and asset metrics.
    """
    require_skin = bool(require_skin or require_rig)
    if isinstance(glb_bytes, (str, Path)):
        glb_bytes = Path(glb_bytes).read_bytes()

    gates: dict[str, dict[str, Any]] = {}

    def add_gate(key: str, name: str, ok: bool, detail: str, metrics: dict | None = None):
        gates[key] = {
            "name": name,
            "ok": bool(ok),
            "detail": detail,
            "metrics": metrics or {},
        }

    # -------------------------------------------------------------
    # Gate 10: glTF Compliance (Parsed first to enable extraction)
    # -------------------------------------------------------------
    try:
        doc, bin_chunk = _parse_glb_chunks(glb_bytes)
        if doc.get("asset", {}).get("version") != "2.0":
            raise ValueError("glTF version must be '2.0'")

        meshes = doc.get("meshes", [])
        if not meshes:
            raise ValueError("glTF contains no meshes")
        prims = meshes[0].get("primitives", [])
        if not prims:
            raise ValueError("glTF mesh contains no primitives")
        prim = prims[0]
        attrs = prim.get("attributes", {})
        if "POSITION" not in attrs:
            raise ValueError("Primitive missing required POSITION attribute")

        positions, pos_acc = _read_accessor(doc, bin_chunk, attrs["POSITION"])
        if not np.isfinite(positions).all() or len(positions) == 0:
            raise ValueError("POSITION accessor has invalid or empty values")

        if "indices" in prim:
            indices_arr, _ = _read_accessor(doc, bin_chunk, prim["indices"])
            indices = indices_arr.flatten().astype(np.int64)
            if len(indices) % 3 != 0:
                raise ValueError("Indices count must be a multiple of 3")
            faces = indices.reshape(-1, 3)
            if (faces < 0).any() or (faces >= len(positions)).any():
                raise ValueError("Indices reference out-of-range vertex indices")
        else:
            if len(positions) % 3 != 0:
                raise ValueError("Unindexed vertex count must be a multiple of 3")
            faces = np.arange(len(positions), dtype=np.int64).reshape(-1, 3)

        # Validate skin inverseBindMatrices when skin node is present
        skins = doc.get("skins", [])
        if skins:
            for s_idx, skin in enumerate(skins):
                joints = skin.get("joints", [])
                ibm_id = skin.get("inverseBindMatrices")
                if ibm_id is not None:
                    ibm_mats, _ = _read_accessor(doc, bin_chunk, ibm_id)
                    if len(ibm_mats) != len(joints):
                        raise ValueError(f"Skin {s_idx} inverseBindMatrices count ({len(ibm_mats)}) != joints count ({len(joints)})")
                    if not np.isfinite(ibm_mats).all() or np.isnan(ibm_mats).any():
                        raise ValueError(f"Skin {s_idx} inverseBindMatrices contains non-finite or NaN values")
                    for midx, mat in enumerate(ibm_mats):
                        m4 = mat.reshape(4, 4)
                        det = float(np.linalg.det(m4))
                        if abs(det) < 1e-12 or not np.isfinite(det):
                            raise ValueError(f"Skin {s_idx} inverseBindMatrix {midx} is singular or invalid (det={det})")

        add_gate(
            "gltf_compliance",
            "Gate 10: glTF 2.0 Compliance",
            True,
            f"Valid glTF 2.0 binary chunks (JSON + {len(bin_chunk)} bytes BIN), verified accessors",
            {"version": "2.0", "bin_size": len(bin_chunk)},
        )
        gltf_ok = True
    except Exception as exc:
        add_gate(
            "gltf_compliance",
            "Gate 10: glTF 2.0 Compliance",
            False,
            f"glTF binary format error: {exc}",
            {"error": str(exc)},
        )
        gltf_ok = False
        # Fill remaining gates with failure due to corrupt glTF
        for i, (k, n) in enumerate([
            ("non_manifold", "Gate 1: Manifold Topology"),
            ("degenerate_triangles", "Gate 2: Degenerate Triangles"),
            ("watertightness", "Gate 3: Watertightness"),
            ("triangle_count", "Gate 4: Triangle Count"),
            ("aspect_ratio", "Gate 5: Aspect Ratio"),
            ("uv_overlap", "Gate 6: UV Island Overlap"),
            ("uv_bounds_gutter", "Gate 7: UV Bounds & Gutter"),
            ("bone_containment", "Gate 8: Bone Containment"),
            ("skinning_compliance", "Gate 9: Skinning Compliance"),
        ], start=1):
            add_gate(k, n, False, "Skipped: glTF compliance failed")
        return {
            "ok": False,
            "passed": False,
            "gates": gates,
            "checks": list(gates.values()),
            "summary": "Asset failed glTF compliance",
        }

    # -------------------------------------------------------------
    # Gate 4: Triangle Count
    # -------------------------------------------------------------
    tri_count = len(faces)
    tri_ok = tri_count <= max_tris
    add_gate(
        "triangle_count",
        "Gate 4: Triangle Count",
        tri_ok,
        f"{tri_count:,} triangles (limit: <= {max_tris:,})",
        {"triangles": tri_count, "max_tris": max_tris},
    )

    # -------------------------------------------------------------
    # Gate 2: Degenerate Triangles (area <= 1e-8)
    # -------------------------------------------------------------
    tris_pos = positions[faces]
    cross = np.cross(
        tris_pos[:, 1] - tris_pos[:, 0], tris_pos[:, 2] - tris_pos[:, 0]
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    degenerate_count = int(np.count_nonzero(areas <= 1e-8))
    degen_ok = degenerate_count == 0
    add_gate(
        "degenerate_triangles",
        "Gate 2: Degenerate Triangles",
        degen_ok,
        f"{degenerate_count} degenerate triangles (area <= 1e-8)",
        {"degenerate_count": degenerate_count, "min_area": float(areas.min()) if len(areas) else 0.0},
    )

    # -------------------------------------------------------------
    # Gate 5: Aspect Ratio (sliver triangles ratio >= 20:1 <= 5%)
    # -------------------------------------------------------------
    e0 = np.linalg.norm(tris_pos[:, 1] - tris_pos[:, 0], axis=1)
    e1 = np.linalg.norm(tris_pos[:, 2] - tris_pos[:, 1], axis=1)
    e2 = np.linalg.norm(tris_pos[:, 0] - tris_pos[:, 2], axis=1)
    l_max = np.maximum(np.maximum(e0, e1), e2)
    h_min = np.where(l_max > 1e-12, 2.0 * areas / np.maximum(l_max, 1e-12), 0.0)
    aspect_ratios = np.where(h_min > 1e-12, l_max / h_min, np.inf)
    sliver_count = int(np.count_nonzero(aspect_ratios >= 20.0))
    sliver_pct = (sliver_count / max(1, tri_count)) * 100.0
    max_aspect = float(np.max(aspect_ratios)) if len(aspect_ratios) else 1.0
    aspect_ok = (sliver_pct <= 3.5) and (max_aspect <= 100.0)
    add_gate(
        "aspect_ratio",
        "Gate 5: Aspect Ratio",
        aspect_ok,
        f"{sliver_count} sliver triangles ({sliver_pct:.2f}% <= 3.5%, max aspect={max_aspect:.1f} <= 100.0)",
        {"sliver_count": sliver_count, "sliver_pct": sliver_pct, "max_aspect": max_aspect},
    )

    # -------------------------------------------------------------
    # Gate 1: Non-manifold Geometry (edges == 0, vertices == 0)
    # -------------------------------------------------------------
    nm_edges, nm_verts = count_non_manifold_elements(positions, faces)
    nm_ok = (nm_edges == 0) and (nm_verts == 0)
    add_gate(
        "non_manifold",
        "Gate 1: Manifold Topology",
        nm_ok,
        f"{nm_edges} non-manifold edges, {nm_verts} non-manifold vertices",
        {"non_manifold_edges": nm_edges, "non_manifold_vertices": nm_verts},
    )

    # -------------------------------------------------------------
    # Gate 3: Watertightness (is_watertight == True, 0 open loops)
    # -------------------------------------------------------------
    b_edges, b_loops = count_boundary_loops(positions, faces)
    watertight = (b_edges == 0) and (nm_edges == 0) and (tri_count > 0)
    wt_ok = watertight and (b_loops == 0)
    add_gate(
        "watertightness",
        "Gate 3: Watertightness",
        wt_ok,
        f"is_watertight={watertight}, {b_loops} open boundary loops ({b_edges} boundary edges)",
        {"is_watertight": watertight, "boundary_edges": b_edges, "open_loops": b_loops},
    )

    # -------------------------------------------------------------
    # Gate 6 & Gate 7: UV Island Overlap & UV Bounds/Gutter
    # -------------------------------------------------------------
    if "TEXCOORD_0" not in attrs:
        add_gate("uv_overlap", "Gate 6: UV Island Overlap", False, "No TEXCOORD_0 attribute found in mesh")
        add_gate("uv_bounds_gutter", "Gate 7: UV Bounds & Gutter", False, "No TEXCOORD_0 attribute found in mesh")
    else:
        uvs, _ = _read_accessor(doc, bin_chunk, attrs["TEXCOORD_0"])
        # Check UV bounds in [0, 1]^2
        bounds_ok = bool(np.all(uvs >= -1e-4) and np.all(uvs <= 1.0 + 1e-4))

        # Resolution for gutter and overlap check
        texture_size = 1024
        images = doc.get("images", [])
        if images:
            img_meta = images[0]
            if "bufferView" in img_meta:
                bv = doc["bufferViews"][img_meta["bufferView"]]
                offset = bv.get("byteOffset", 0)
                img_data = bin_chunk[offset : offset + bv["byteLength"]]
                try:
                    pil_img = Image.open(io.BytesIO(img_data))
                    texture_size = max(1024, max(pil_img.size))
                except Exception:
                    pass

        # Identify UV islands
        # Two faces are in the same UV island if they share an edge in UV space
        rows = np.repeat(np.arange(len(faces)), 3)
        cols = faces.flatten()
        face_vert = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.int32), (rows, cols)),
            shape=(len(faces), len(positions)),
        )
        face_adj = (face_vert @ face_vert.T) >= 2
        n_islands, island_labels = sp.csgraph.connected_components(face_adj, directed=False)

        uv_tris = uvs[faces]
        uv_px = uv_tris * texture_size

        # High-performance forward & backward rasterization to detect any inter-island overlap
        im_fwd = Image.new("I", (texture_size, texture_size), 0)
        draw_fwd = ImageDraw.Draw(im_fwd)
        for fi in range(len(faces)):
            isl_id = int(island_labels[fi]) + 1
            poly = [tuple(p) for p in uv_px[fi]]
            draw_fwd.polygon(poly, fill=isl_id)

        im_bwd = Image.new("I", (texture_size, texture_size), 0)
        draw_bwd = ImageDraw.Draw(im_bwd)
        for fi in reversed(range(len(faces))):
            isl_id = int(island_labels[fi]) + 1
            poly = [tuple(p) for p in uv_px[fi]]
            draw_bwd.polygon(poly, fill=isl_id)

        arr_fwd = np.array(im_fwd)
        arr_bwd = np.array(im_bwd)

        # Overlapping texels between distinct islands
        overlap_mask = (arr_fwd != arr_bwd) & (arr_fwd > 0) & (arr_bwd > 0)
        overlap_pixels = int(np.count_nonzero(overlap_mask))

        # Topological vertices: unique positions rounded to 5 decimals
        _, topo_inv = np.unique(np.round(positions, decimals=5), axis=0, return_inverse=True)
        topo_vert_count = len(np.unique(topo_inv))
        render_topo_ratio = len(positions) / max(1, topo_vert_count)
        max_allowed_islands = max(64, int(np.ceil(0.15 * len(faces))))

        islands_ok = n_islands <= max_allowed_islands
        ratio_ok = render_topo_ratio <= 1.8 + 1e-4
        overlap_ok = (overlap_pixels == 0) and islands_ok and ratio_ok

        detail_overlap = (
            f"{n_islands} islands (limit <= {max_allowed_islands}: {islands_ok}), "
            f"render/topo ratio={render_topo_ratio:.2f}x (limit <= 1.80x: {ratio_ok}), "
            f"{'0.0% overlap between islands' if overlap_pixels == 0 else f'{overlap_pixels} overlapping pixels between islands'}"
        )

        add_gate(
            "uv_overlap",
            "Gate 6: UV Island Overlap & Fragmentation",
            overlap_ok,
            detail_overlap,
            {
                "islands": int(n_islands),
                "max_allowed_islands": max_allowed_islands,
                "render_vertices": len(positions),
                "topo_vertices": topo_vert_count,
                "render_topo_ratio": render_topo_ratio,
                "overlapping_pixels": overlap_pixels,
            },
        )

        # Gutter check: border margin >= 2 px
        border_gutter_violation = bool(
            np.any(arr_fwd[:2, :] > 0)
            or np.any(arr_fwd[-2:, :] > 0)
            or np.any(arr_fwd[:, :2] > 0)
            or np.any(arr_fwd[:, -2:] > 0)
        )

        # Inter-island gutter >= 2 px
        inter_gutter_violation = False
        if n_islands > 1 and np.any(arr_fwd > 0):
            dist, (iy, ix) = distance_transform_edt(arr_fwd == 0, return_indices=True)
            nearest = arr_fwd[iy, ix]
            # Horizontal transitions
            diff_h = (nearest[:, :-1] != nearest[:, 1:]) & (nearest[:, :-1] > 0) & (nearest[:, 1:] > 0)
            if np.any(diff_h):
                gap_h = np.min(dist[:, :-1][diff_h] + dist[:, 1:][diff_h])
                if gap_h < 2.0:
                    inter_gutter_violation = True
            # Vertical transitions
            if not inter_gutter_violation:
                diff_v = (nearest[:-1, :] != nearest[1:, :]) & (nearest[:-1, :] > 0) & (nearest[1:, :] > 0)
                if np.any(diff_v):
                    gap_v = np.min(dist[:-1, :][diff_v] + dist[1:, :][diff_v])
                    if gap_v < 2.0:
                        inter_gutter_violation = True

        gutter_ok = bounds_ok and (not border_gutter_violation) and (not inter_gutter_violation)
        detail_gutter = (
            f"Bounds in [0, 1]^2 ({bounds_ok}), "
            f"border gutter >= 2 px ({not border_gutter_violation}), "
            f"inter-island gutter >= 2 px ({not inter_gutter_violation})"
        )
        add_gate(
            "uv_bounds_gutter",
            "Gate 7: UV Bounds & Gutter",
            gutter_ok,
            detail_gutter,
            {
                "bounds_ok": bounds_ok,
                "border_gutter_ok": not border_gutter_violation,
                "inter_island_gutter_ok": not inter_gutter_violation,
                "texture_size": texture_size,
            },
        )

    # -------------------------------------------------------------
    # Gate 8: Bone Containment
    # -------------------------------------------------------------
    mesh_min = positions.min(axis=0)
    mesh_max = positions.max(axis=0)

    skins = doc.get("skins", [])
    if skins:
        skin = skins[0]
        joints = skin.get("joints", [])
        ibm_id = skin.get("inverseBindMatrices")
        joint_positions = []
        if ibm_id is not None:
            ibm_mats, _ = _read_accessor(doc, bin_chunk, ibm_id)
            for mat in ibm_mats:
                # glTF 2.0 matrices are column-major; transpose to standard row-major
                m4 = mat.reshape(4, 4).T
                try:
                    world_m = np.linalg.inv(m4)
                    joint_positions.append(world_m[:3, 3])
                except Exception:
                    pass

        if not joint_positions and "nodes" in doc:
            for j_node_idx in joints:
                node = doc["nodes"][j_node_idx]
                joint_positions.append(node.get("translation", [0.0, 0.0, 0.0]))

        if joint_positions:
            j_arr = np.array(joint_positions, dtype=np.float64)
            outside = (j_arr < mesh_min - 1e-4) | (j_arr > mesh_max + 1e-4)
            violations = int(np.count_nonzero(np.any(outside, axis=1)))
            bone_ok = violations == 0
            add_gate(
                "bone_containment",
                "Gate 8: Bone Containment",
                bone_ok,
                f"{violations} bones outside mesh AABB [{mesh_min.round(3)} to {mesh_max.round(3)}]",
                {"bone_count": len(j_arr), "violations": violations},
            )
        else:
            add_gate(
                "bone_containment",
                "Gate 8: Bone Containment",
                True,
                "Skin present with 0 resolvable joint transforms",
                {"bone_count": 0, "violations": 0},
            )
    else:
        if require_skin:
            add_gate(
                "bone_containment",
                "Gate 8: Bone Containment",
                False,
                "require_rig=True but asset has no skeleton/joints",
                {"bone_count": 0, "violations": 0},
            )
        else:
            add_gate(
                "bone_containment",
                "Gate 8: Bone Containment",
                True,
                "Static asset: 0 bones outside mesh AABB",
                {"bone_count": 0, "violations": 0},
            )

    # -------------------------------------------------------------
    # Gate 9: Skinning Compliance
    # -------------------------------------------------------------
    is_skinned = bool(skins or ("WEIGHTS_0" in attrs))
    if is_skinned:
        if "WEIGHTS_0" not in attrs:
            add_gate(
                "skinning_compliance",
                "Gate 9: Skinning Compliance",
                False,
                "Skinned asset missing WEIGHTS_0 attribute",
            )
        else:
            w0, _ = _read_accessor(doc, bin_chunk, attrs["WEIGHTS_0"])
            weights = w0
            if "WEIGHTS_1" in attrs:
                w1, _ = _read_accessor(doc, bin_chunk, attrs["WEIGHTS_1"])
                weights = np.hstack([weights, w1])

            # Max 4 bones per vertex
            influences = np.count_nonzero(weights > 1e-6, axis=1)
            max_influences = int(influences.max()) if len(influences) else 0
            max_bones_ok = max_influences <= 4

            # Sum of weights == 1.0 +- 1e-5
            row_sums = weights.sum(axis=1)
            max_sum_err = float(np.max(np.abs(row_sums - 1.0)))
            normalized_ok = max_sum_err <= 1e-5

            # 0 unskinned vertices
            unskinned_count = int(np.count_nonzero(row_sums <= 1e-5))
            unskinned_ok = unskinned_count == 0

            skin_ok = max_bones_ok and normalized_ok and unskinned_ok
            detail_skin = (
                f"max influences={max_influences} (<=4: {max_bones_ok}), "
                f"sum(w) err={max_sum_err:.1e} (<=1e-5: {normalized_ok}), "
                f"unskinned verts={unskinned_count}"
            )
            add_gate(
                "skinning_compliance",
                "Gate 9: Skinning Compliance",
                skin_ok,
                detail_skin,
                {
                    "max_influences": max_influences,
                    "max_sum_error": max_sum_err,
                    "unskinned_vertices": unskinned_count,
                },
            )
    else:
        if require_skin:
            add_gate(
                "skinning_compliance",
                "Gate 9: Skinning Compliance",
                False,
                "require_skin=True but asset has no skinning",
            )
        else:
            add_gate(
                "skinning_compliance",
                "Gate 9: Skinning Compliance",
                True,
                "Static asset: 0 unskinned vertices (skinning not required)",
                {"is_skinned": False},
            )

    all_passed = all(g["ok"] for g in gates.values())
    failed_names = [g["name"] for g in gates.values() if not g["ok"]]
    summary = (
        "All 10 production gates PASSED"
        if all_passed
        else f"Failed {len(failed_names)} gate(s): {', '.join(failed_names)}"
    )

    return {
        "ok": all_passed,
        "passed": all_passed,
        "gates": gates,
        "checks": list(gates.values()),
        "summary": summary,
        "asset_metrics": {
            "vertices": len(positions),
            "triangles": tri_count,
            "is_watertight": watertight,
            "non_manifold_edges": nm_edges,
            "non_manifold_vertices": nm_verts,
            "sliver_percentage": sliver_pct,
            "is_skinned": is_skinned,
        },
    }
