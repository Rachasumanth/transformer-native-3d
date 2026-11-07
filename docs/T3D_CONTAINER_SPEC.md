# T3D Container Format Specification (T3DV v2)

**Document Version:** 2.0.0  
**Specification Status:** Production Standard  
**MIME Type:** `model/x-t3d`  
**File Extension:** `.t3d` (provisional compatibility with `.vxz`)

---

## 1. Overview & Architectural Goals

The **Transformer-Native 3D Container (`.t3d`)** is a high-performance, deterministic binary container designed for streaming 3D sparse surface voxels, dense PBR attributes, volumetric articulation latents, and discrete topological streams directly into GPU memory for Transformer-based 3D foundation models.

Key architectural properties:
1. **Zero-Copy Streaming**: Fixed 128-byte header followed by 64-byte aligned contiguous tensor payload blocks.
2. **Space-Filling Curve Serialization**: Bijective 64-bit Hilbert and Morton (Z-order) serialization for optimal spatial locality in 3D attention mechanisms without runtime shuffling overhead.
3. **Decoupled Multi-Track Design**: Supports both Track A (Field-Free O-Voxel + PBR + Articulation) and Track B (BPT topological patch sequence).
4. **Data Integrity & Cryptographic Sealing**: Production profiles (3 and 4) terminate with a mandatory 32-byte SHA-256 trailer computed across the entire header and payload.
5. **Deterministic Version Migration**: Explicit forward and backward version policies with automated migration from legacy v1 / unsealed containers into sealed v2 containers.

---

## 2. Binary Header Layout (128 Bytes)

All integers and floating-point values are stored in **little-endian** byte order (`<`).

```
+-----------------------------------------------------------------------------------+
| 0x00..0x03: Magic "T3DV" (4 bytes ASCII)                                         |
+-----------------------------------------------------------------------------------+
| 0x04..0x07: Schema Version (uint32, value = 2)                                    |
+-----------------------------------------------------------------------------------+
| 0x08..0x1F: Canonical Bounding Box AABB (float32[2][3], min_xyz, max_xyz)         |
+-----------------------------------------------------------------------------------+
| 0x20..0x2B: Grid Resolution (uint32[3], e.g. [32, 32, 32] or [1024, 1024, 1024]) |
+-----------------------------------------------------------------------------------+
| 0x2C..0x37: Voxel Dimensions (float32[3], voxel_size = (aabb[1]-aabb[0])/grid)   |
+-----------------------------------------------------------------------------------+
| 0x38..0x47: Element Counts (uint32[4]: [L_active, L_latents, N_bpt, N_joints])   |
+-----------------------------------------------------------------------------------+
| 0x48..0x7F: Attribute & Stream Layout (uint16[28], 56 bytes)                      |
+-----------------------------------------------------------------------------------+
```

### Detailed Field Breakdown

| Byte Offset | Field Name | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `0x00..0x03` | `magic` | `char[4]` | ASCII identifier `b"T3DV"`. Rejects any other magic. |
| `0x04..0x07` | `version` | `uint32` | Container version. Current production version is `2`. |
| `0x08..0x13` | `aabb_min` | `float32[3]` | World-space minimum corner $[x_{min}, y_{min}, z_{min}]$. Must be strictly less than `aabb_max`. |
| `0x14..0x1F` | `aabb_max` | `float32[3]` | World-space maximum corner $[x_{max}, y_{max}, z_{max}]$. Canonical center is origin $[0,0,0]$. |
| `0x20..0x2B` | `grid_size` | `uint32[3]` | Resolution along $(x, y, z)$. Values must be positive and $\le 65536$. |
| `0x2C..0x37` | `voxel_size`| `float32[3]` | Physical dimensions per voxel cell. Must match `(aabb_max - aabb_min) / grid_size` within $2\times 10^{-7}$ relative tolerance. |
| `0x38..0x3B` | `L_active` | `uint32` | Number of active surface voxels $L$. Must be $0$ for latent profiles. |
| `0x3C..0x3F` | `L_latents`| `uint32` | Number of downsampled latent tokens. Must be $0$ for raw profiles. |
| `0x40..0x43` | `N_bpt` | `uint32` | Number of Track B topological tokens (range $[0, 5120)$). $0$ if absent. |
| `0x44..0x47` | `N_joints` | `uint32` | Number of skeletal joints in articulated hierarchy. $0$ if absent. |
| `0x48..0x7F` | `attr_layout` | `uint16[28]` | Layout table specifying channel slices, profiles, and curve order (see below). |

---

## 3. Attribute Layout Table (`attr_layout`, uint16[28])

The 56-byte layout table defines the structure of packed float16 attributes:

### Raw Profiles (Profile 1 & Profile 3)

Entries 0 through 17 define 9 consecutive `[start, end]` channel slice pairs:

| Slice Index | Attribute Name | Channels | Type | Description |
| :--- | :--- | :--- | :--- | :--- |
| `0..1` | `dual_vertices` | 3 | `float16` | Normalized local offset $(\delta x, \delta y, \delta z) \in [0.0, 1.0]^3$ inside voxel cell. **Mandatory**. |
| `2..3` | `intersected_flag` | 3 | `float16` | Binary edge crossing indicators on primal edges $(+x, +y, +z)$. Values must be exactly 0.0 or 1.0. **Mandatory**. |
| `4..5` | `split_weight` | 1 | `float16` | Quad triangulation diagonal weight. Optional (`65535, 65535` if absent). |
| `6..7` | `base_color` | 3 | `float16` | Linear RGB base color $[0.0, 1.0]^3$. Must be present if any PBR is present. |
| `8..9` | `metallic` | 1 | `float16` | PBR metallic ratio $[0.0, 1.0]$. |
| `10..11` | `roughness` | 1 | `float16` | PBR roughness $[0.0, 1.0]$. |
| `12..13` | `alpha` | 1 | `float16` | Opacity / alpha value $[0.0, 1.0]$ ($1.0 = \text{opaque}$). |
| `14..15` | `skeleton_field` | 6 | `float16` | 6D displacement vector pointing to nearest joint and its parent. Optional. |
| `16..17` | `skin_embedding` | 4 | `float16` | 4-channel vertex skinning latent produced by SkinAE. Optional. |

Entries 18 through 27:
- `layout[18]`: Total attribute width $C = \sum \text{channels}$ (must be in range $[6, 23]$).
- `layout[19]`: Profile identifier:
  - `1`: Legacy unsealed raw container (no trailer).
  - `3`: Production sealed raw container (with 32-byte SHA-256 trailer).
- `layout[20]`: Space-filling curve order:
  - `0`: Morton / Z-order (`z`).
  - `1`: Transposed Morton / Z-order (`tz`).
  - `2`: Hilbert curve (`h`).
  - `3`: Transposed Hilbert curve (`th`).
- `layout[21]`: Memory alignment requirement (must be `64`).
- `layout[22..27]`: Reserved (must be `0`).

---

## 4. Payload Blocks & Alignment

All payload blocks begin on a 64-byte aligned byte offset: `offset = (cursor + 63) // 64 * 64`. Any alignment gaps must be filled with zero bytes (`b"\x00"`).

### Raw Profile 3 Sequence:
1. **Coordinates Block**: `uint16[L, 3]` discrete grid coordinates $(x, y, z) \in [0, \text{grid}-1]^3$. Must be strictly sorted according to the declared space-filling curve.
2. **Attributes Block**: `float16[L, C]` normalized floating-point attributes. Non-finite values (`NaN`, `Inf`) are rejected.
3. **BPT Tokens Block** (optional): `uint16[N_bpt]` discrete topological sequence tokens $\in [0, 5120)$.
4. **Integrity Trailer**: Exactly 32 bytes containing the SHA-256 cryptographic digest of the complete 128-byte header and all aligned payload blocks.

---

## 5. Version Migration Policy

To guarantee forward and backward interoperability:

1. **Reading**:
   - `loads(data)` strictly requires `version == 2` and `profile in (1, 3)`.
   - `loads(data, allow_version_1=True)` permits legacy `version == 1` containers.
2. **Writing**:
   - `dumps(...)` exclusively emits `version == 2`, `profile == 3` (sealed).
3. **Migration Adapter (`migrate_container`)**:
   - `migrate_container(data: bytes) -> bytes` validates the input and idempotently upgrades any v1 or unsealed container to a production-grade v2 sealed container:
     - Detects magic `T3DV` and version `1` or `2`.
     - Validates payload geometry, bounds, and alignment.
     - Upgrades header version to `2` and profile to `3` (or `4` for latents).
     - Appends the verified 32-byte SHA-256 integrity trailer.
     - Rejects malformed or corrupted inputs with `ValueError`.
