# CPU BPT reference conventions

`t3d.bpt.encode(vertices, faces)` accepts normalized vertices in [-0.5,0.5].
Quantization uses floor((v+0.5)*128), capped at 127 at the upper boundary.
Dequantization returns bin centers, with coordinate error at most 1/256.
Coincident quantized vertices merge. Collapsed, collinear, and duplicate
quantized triangles are rejected explicitly rather than dropped.

Quantized vertices sort by z, y, x. Face cycles rotate to the smallest vertex
index without reversing winding, then sort lexicographically. The first active
face chooses its highest active-degree vertex as center, breaking ties by the
canonical vertex index. Directed incident edges extend a fan forward/backward;
remaining branches start separate patches. Every face is consumed exactly once.

A coordinate q factors into block q//16 (8 bins per axis) and offset q%16.
Both linear indices use x-major order: block x*64+y*8+z and offset x*256+y*16+z.
The 5120-token local vocabulary is:

| Range | Meaning |
| --- | --- |
| 0–511 | Center block, also starts a new patch |
| 512–1023 | Peripheral block plus 512 |
| 1024–5119 | Local offset plus 1024 |

Every center emits a block and offset. Peripheral vertices inherit the previous
block unless it changes; then they emit a peripheral block first. Each patch
contains a center and at least two peripheral vertices. Consecutive peripheral
pairs form oriented triangles with the center. A closed fan repeats its first
peripheral vertex at the end. Stream length terminates the last patch; there are
no extra separator or BOS/EOS tokens. `decode` validates block/offset grammar and
returns merged vertices and triangle indices. The container stores these tokens
but does not yet label the tokenizer profile: provenance must be kept with the
training dataset. Checkpoint byte/token compatibility with upstream BPT is not
claimed; these are explicit local conventions derived from the blueprint.

The `tools/benchmark_bpt_grid.py` command recreates an 8192-face planar fixture.
It preserves all oriented quantized triangles with 15,888 tokens (21.55% of raw
nine-coordinate tokens, or 78.45% reduction) and 2,080 BPT patches. The patch
count does not meet the blueprint's <60 fragmentation target;
this patch count is BPT-specific. The blueprint's comparison of 54.7 EdgeRunner
and 199.5 AMT subsequences comes from EdgeRunner's modified EdgeBreaker tokenizer,
not BPT. Therefore BPT patch count and EdgeBreaker subsequence count are not
interchangeable; the plan's <60 gate needs a separately specified measurement and
an EdgeBreaker baseline before it can be evaluated fairly. The BPT paper defines
patch aggregation as all unvisited faces incident to a selected highest-degree
center, with faces ordered by their lowest vertex; the local encoder follows that
center-selection and ordering convention but uses its own explicit oriented-fan
grammar. See the [BPT paper](https://arxiv.org/abs/2411.07025) and
[EdgeRunner paper](https://arxiv.org/abs/2409.18114) for their distinct methods.

`t3d.bpt.quantization_diagnostics` reports source degeneracies, quantized
vertex collisions, collapsed faces, collinear faces and coordinate errors
without modifying the mesh. `tools/audit_bpt_corpus.py` applies that gate to six
static Khronos GLBs and three rigged Khronos pilot meshes at their shared
node-default pose. The original three static and three rigged pilot meshes are
all rejected at BPT's 128-bin setting: Avocado has
one collinearized triangle; WaterBottle 957 collapsed/28 collinear; Lantern
4,556/86; CesiumMan 324/18; Fox 14/0; and RiggedSimple 0/6. All six have zero
exact zero-area faces in the source; Lantern has six additional source triangles
below the audit's 1e-8 doubled-area threshold. A diagnostic resolution sweep
finds first degeneracy-free settings of 256 bins for Avocado and Fox, 512 for
CesiumMan and RiggedSimple, and 2048 for WaterBottle; Lantern remains degenerate
through 8192. These higher settings are diagnostics only because the current
encoder's fixed block/offset vocabulary implements 128 bins. The pilot is not a
representative artist-mesh corpus; this establishes a preprocessing/data
eligibility gap, not that BPT cannot encode clean meshes. No failed faces are
silently deleted. Full counts and source hashes are in
`artifacts/bpt/corpus-audit.json`; no inference/training or mesh-quality claim
follows from this tokenizer check.

The same preflight checked all 24 proper axis-aligned cube rotations (including
the plan's quarter-turn augmentation family). None of the original six meshes
became eligible at 128 bins under any tested orientation. Orientation
augmentation alone therefore does not repair this pilot's quantization losses.

The static screen also includes four CC0 Khronos sample assets:
[Barramundi Fish](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BarramundiFish),
[Boom Box](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BoomBox),
and [Sheen Chair](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/SheenChair),
plus the high-resolution [Scattering Skull](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/ScatteringSkull).
Their GLBs and license/readme provenance are retained in
`artifacts/bpt-candidates/`. The material-extension-only Sheen Chair path uses a
geometry-only static loader; it still rejects required non-material extensions,
skins, animation, non-triangle primitives, and unknown primitive extensions.
The [CC-BY 4.0 Khronos Box sample](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Box)
is included as a simple eligible source control. It has 12 triangles, encodes
to 36 tokens and three BPT patches, preserves all oriented quantized faces, and
is eligible under all 24 tested rotations. It is a simple box, not a realistic
artist mesh. The three Khronos low/mid-face CC0 assets are rejected at 128 bins: Barramundi
Fish 638 collapsed / 11 collinear faces, Boom Box 1,200 / 23, and Sheen Chair
20,634 / 228. Scattering Skull has 188,871 faces and fails with 99,659
collapsed / 393 collinear faces. Added a 33,902-face [CC0 Poly Haven Lantern 01](https://polyhaven.com/a/Lantern_01)
source using its 1k glTF geometry and textures, retaining files and SHA256
provenance in `artifacts/bpt-candidates/PolyHaven_Lantern_01/`. It fails with
17,681 collapsed / 180 collinear faces. Four more CC0 Poly Haven models are
retained with source manifests: Coffee Table 01 (10,404 faces; 4,434 collapsed /
109 collinear), Wooden Chair 01 (19,992; 13,661 / 454), Food Lime 01 (8,100;
116 / 35), and Metal Toolbox (14,228; 10,475 / 36). All sixteen screened assets
have zero exact zero-area source faces. In the untouched-source audit, only the
simple Box encodes; the other fifteen fail, and none of their 24 tested
orientations is eligible. These sixteen licensed assets remain a small
eligibility screen, not a representative corpus.

An explicit cleanup experiment drops only faces that become collapsed or
collinear at 128 bins (and duplicate quantized faces), then verifies the
retained oriented face set through BPT round-trip. The 682-face Avocado loses
one face (0.15%); the 8,100-face Food Lime loses 151 (1.86%). Both cleaned meshes
pass all 24 cube rotations. At 1,024 area-weighted samples and 0.01 tolerance,
Avocado has mean bidirectional surface distance 0.00170 and F-score 1.0; Food
Lime has 0.00128 and 1.0. Food Lime's quantized topology has 95 boundary edges
and one nonmanifold edge, so the geometric score alone does not establish
topological quality. Source meshes remain unchanged. The reproducible report
and cleaned examples are in `artifacts/bpt/quantized-cleanup-audit.json`,
`tools/benchmark_bpt_quantized_cleanup.py`, and
`artifacts/bpt/cleaned_*_all_24_rotations.npz`.

An explicit quadric-error simplification experiment is recorded in
`artifacts/bpt/decimation-audit.json` and reproduced by
`tools/benchmark_bpt_decimation.py` (install the optional research extra with
`pip install -e '.[research]'`). The simplifier did not meet requested face
counts on these Khronos assets and did not clear the 128-bin degeneracy gate;
it is not a validated repair path. For a clean tokenizer control, the separate
synthetic five-subdivision icosphere fixture is reproduced by
`tools/benchmark_bpt_icosphere.py`. It encodes 20,480 faces in 35,996 tokens
(80.47% fewer than nine raw coordinate tokens per face), passes the 128-bin
gate, and preserves every oriented quantized face and closed edge incidence
through round-trip. It is synthetic evidence only, not curated training data
or evidence of real-asset quality.

`tools/benchmark_bpt_locality.py` adds the plan's exploratory CR/PTME/AVD audit
for the eligible Box, planar-grid, icosphere, and explicitly filtered Avocado
and Food Lime inputs. Under its explicit
local definitions (CR = tokens / 9F; PTME = empirical token entropy in bits,
normalized by raw coordinate-token count following the metric family in
[FreeMesh](https://arxiv.org/abs/2505.13573); AVD@t = mean distance from each
emitted vertex occurrence to all previous t emitted vertex occurrences), the
icosphere has CR 0.195, PTME 2.293, and AVD@5/25/50 of 0.214/0.336/0.381. The
planar grid has CR 0.215, PTME 1.514, and AVD@5/25/50 of 0.040/0.097/0.164.
Cleaned Avocado has CR 0.253, PTME 2.181 and AVD@5/25/50 of 0.207/0.351/0.394;
cleaned Food Lime has CR 0.218, PTME 2.424 and AVD@5/25/50 of
0.124/0.288/0.392. The two synthetic fixtures and two explicitly filtered
sources meet the plan's numeric CR and PTME thresholds under this local
calculation. The Box control has CR 0.333 and PTME 1.306. These are exploratory measurements,
not estimates over artist assets or directly comparable reproductions of the
paper's tokenizer. The fixture set mixes synthetic and real meshes and is too
small for corpus-level PTME claims. The BPT paper describes AVD@t as distance to
the previous t vertices but does not give its exact averaging formula in the
paper text; this local all-prior-vertices average is therefore an explicit
operational interpretation, not a reproduction. Per-mesh definitions, center
patch counts and limitations are recorded in `artifacts/bpt/locality-audit.json`.

`tools/benchmark_edgerunner_fragmentation.py` independently counts `B` starts
from EdgeRunner's published half-edge traversal pseudocode, then compares those
counts with BPT patch starts on the same local BPT-round-tripped topology. The
CC-BY Box has 2 EdgeRunner-style starts / 3 BPT patches; the open 8,192-face
grid has 65 / 2,080; the closed 20,480-face icosphere has 27 / 3,551; and
cleaned Avocado has 16 / 142. Food Lime is not counted because its decoded
topology has a nonmanifold edge. For boundary handling, missing neighboring
faces are treated as already visited when choosing a traversal direction; the
paper initializes boundary vertices as visited but does not spell out this
missing-face case. The counter also uses the local fixture face order and does
not emit vertex tokens. These counts are an auditable local paper-derived
metric, not a reproduction of EdgeRunner's reported 54.7 mean or AMT's 199.5
mean on the authors' dataset. See
`artifacts/bpt/edgerunner-fragmentation-audit.json`.
