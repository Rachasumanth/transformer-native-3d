# Device-resident dual-grid extraction

`t3d.extraction_torch.voxels_to_mesh` accepts same-device int64 coordinates,
floating local dual vertices, boolean edge flags, grid dimensions and an AABB,
with optional floating split weights. It returns world vertices and int64 faces
on that device, retaining input row correspondence for material and rig fields.

Marked primal edges query four transverse neighbors. Keys use a signed-int64
linear grid index with explicit coordinate-bound checks to prevent boundary
aliasing. Sorting plus `searchsorted` resolves sparse neighbors. Missing quads
are skipped. Weighted splitting normalizes weights before products; otherwise
the minimum interior angle across the two triangles selects the diagonal.
Degenerate/tied alternatives select diagonal 1–3, matching the CPU reference.
Boolean flags do not encode winding, so orientation remains provisional.

This is a PyTorch reference GPU implementation. The plan's custom CUDA hash
table, sub-100 ms performance gate, CuMesh cleanup/decimation, GPU UV baking and
production GLB packaging remain unfinished. Scalar validation predicates may
synchronize the device. The routine materializes marked-edge queries and both
candidate triangulations; memory scalability still needs measurement.

Parity checks compare CPU and device outputs on randomized full/sparse grids,
shuffled input order, all-axis boundary edges, weighted and angle splitting,
degenerate geometry and empty input. CUDA execution evidence belongs in
PROGRESS.md after source-matched Kaggle results have been inspected.

`python -m tools.benchmark_extraction --source-sha256 BUNDLE_SHA256` runs the
Kaggle latency/memory probe at 512³, 1024³ and 1536³ by default. Each fixture is
one axis-aligned sheet with R² active voxels and 2(R−1)² triangles. It checks exact
face counts and index bounds plus total triangle area before timing. Both supplied
weights and minimum-angle triangulation are measured after a warmup, with three
synchronized repetitions per case. JSON records each wall-clock duration, CUDA
event duration and peak PyTorch allocated memory (including resident inputs),
plus incremental allocation above the warmed input baseline. This excludes
allocator-reserved memory and non-PyTorch GPU allocations. Wall timing includes
validation synchronizations, key sorting, lookup and output generation; input
transfer, VAE decode and all later export stages are excluded. Synthetic planar
support is not a production asset distribution or an end-to-end export benchmark.

Kaggle version 14 measured the following on one Tesla T4 with Torch 2.11.0+cu128
and CUDA 12.8. Values are medians of three full-call wall times after warmup.

| Grid | Active voxels | Triangles | Weighted split (ms) | Minimum angle (ms) |
| --- | ---: | ---: | ---: | ---: |
| 512³ | 262,144 | 522,242 | 5.77 | 10.15 |
| 1024³ | 1,048,576 | 2,093,058 | 19.82 | 32.00 |
| 1536³ | 2,359,296 | 4,712,450 | 36.27 | 69.99 |

Peak PyTorch allocated memory at 1536³ was 981,286,912 bytes for weighted
splitting and 2,303,284,224 bytes for minimum-angle splitting, including inputs.
All six count/bounds/area checks passed. The source-matched raw observations are
in `artifacts/kaggle-v14/extraction-benchmark.json`. These fixture-specific times
meet 100 ms but do not prove the research gate over production asset workloads.
They also show the angle branch's greater temporary memory use; it remains a
candidate for fused kernels or bounded-memory processing if larger support is
required. No custom hash kernel is currently used.
