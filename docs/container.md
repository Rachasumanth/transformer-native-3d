# Experimental T3DV v2 raw profiles 1 and 3

These local profiles implement the blueprint's 128-byte header. They are
provisional, not a claim of compatibility with upstream `.vxz`. All scalars are
little endian. Profile 1 is the original raw layout; new writes use profile 3,
which adds a SHA-256 integrity trailer.

| Offset | Content |
| --- | --- |
| 0 | `T3DV`, uint32 version 2 |
| 8 | float32 AABB, min xyz then max xyz |
| 32 | uint32 grid xyz |
| 44 | float32 voxel size xyz, derived from stored AABB/grid |
| 56 | uint32 counts: active voxels, latent tokens, BPT tokens, joints |
| 72 | uint16[28] attribute layout |

Layout entries 0–17 are nine start/end channel pairs: dual xyz, edge flags xyz,
split weight, RGB, metallic, roughness, alpha, skeleton displacement (6), skin
embedding (4). Missing pairs use 65535/65535. Present slices must be consecutive
and non-overlapping. Geometry is mandatory; PBR slices must be present together.
Entries 18–21 contain channel count C, profile identifier 1 or 3, curve
identifier, and alignment 64.
Curve identifiers are 0=Z, 1=TZ, 2=H, 3=TH. Transposition exchanges x and y;
Hilbert orientation follows `hilbertcurve` 2.x with p=depth and n=3.
Entries 22–27 are reserved zero. Latent and joint counts must currently be zero.
Optional rig fields are intermediate voxel attributes, without a joint hierarchy.

Payloads start at the next 64-byte boundary, with zero padding:

1. uint16[L,3] coordinates, strictly sorted by the declared curve. Morton uses
   x/y/z lanes 0/1/2. Depth is max(1, bit_length(max(grid_size)-1)).
2. float16[L,C] raw attributes, flags represented by exact 0 or 1.
3. uint16[N_bpt] supplied tokens, each in [0,5120).

There is no final padding before the profile-3 trailer. Empty blocks still begin
at the aligned cursor. BPT token storage does not establish tokenizer grammar
correctness. Float16 attributes are lossy; encoding rejects overflow. AABB uses
float32. Reader checks geometry, attribute layout, flags, padding, exact file
length, coordinate order, and token range. Profile 3 appends 32 bytes containing
SHA-256 of the complete header and payload. This detects accidental corruption,
not malicious edits. Profile 1 remains readable without an integrity check.
Input is loaded in memory, not streamed or memory mapped. `save` validates the
encoded payload, writes a sibling temporary file, and atomically replaces the
destination.

Raw profiles 1 and 3 reject nonzero latent/joint counts rather than
misinterpreting them. Profiles 2 and 4 below store latents and hierarchy;
streaming remains pending for all profiles.

## Experimental latent profiles 2 and 4

`t3d.latent_container` provides `LatentAsset`, `dumps_latents/loads_latents` and
`save_latents/load_latents`. New writes use profile 4, which adds an integrity
trailer to legacy profile 2. Raw profiles 1/3 still use `t3d.container`; its
reader rejects latent profiles explicitly. All profiles retain the same 128-byte
header. For profiles 2/4, counts are `[0, latent_tokens, BPT_tokens, joints]`.

Layout uint16 entries are:

| Entry | Meaning |
| --- | --- |
| 0 | Spatial compression factor 16 |
| 1–3 | Shape, material, rig channel dimensions; zero marks an absent optional stream |
| 4–18 | Reserved zero |
| 19 | Profile identifier 2 (legacy) or 4 (checksummed) |
| 20 | Curve ID 0=Z, 1=TZ, 2=H, 3=TH |
| 21 | Alignment 64 |
| 22–27 | Reserved zero |

Aligned blocks, including alignment before empty blocks, are:

1. uint16[L,3] latent coordinates in `ceil(grid_size/16)`, sorted by declared curve.
2. float16[L,D_shape] shape latents (required).
3. float16[L,D_material] material latents (optional).
4. float16[L,D_rig] rig latents (optional).
5. Three 32-byte SHA256 checkpoint identities in shape/material/rig order; absent
   streams use zero bytes. Present streams require nonzero identities.
6. float32[J,3] world-space joints, optionally empty.
7. int32[J] parents with -1 roots; validated as an acyclic forest.
8. uint16[N_bpt] optional tokens.

Profile 2 has no integrity trailer. Profile 4 appends a 32-byte SHA-256 digest
over the complete header and payload and verifies it before parsing. It detects
accidental corruption, not malicious edits. All latent streams share coordinate
support in both profiles. Separate skeleton
voxel support, inverse bind matrices, animations, and joint confidence remain
future work. Joint hierarchy storage does not imply trained rig reconstruction.

`expected_models={'shape': checkpoint_sha256, 'material': checkpoint_sha256}`
checks decoder identity at load time. The exact checkpoint must be retained:
its model configuration defines network architecture. Opaque checkpoint identity
is not an embedded decoder. Legacy profile 2 has no checksum; profile 4 verifies
its integrity trailer. `save_latents` validates and atomically replaces the
destination. Float16 latents and float32 joints are lossy; overflow is rejected.
AABB/grid and canonical coordinate checks remain mandatory.
