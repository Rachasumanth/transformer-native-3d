# Source-surface query hierarchy

`t3d.proximity.TriangleBVH` builds a CPU triangle AABB hierarchy and reuses the
reference closest-triangle solver within leaves. Query bounds prune nodes by
minimum possible squared distance with conservative floating-point traversal
slack. Exact leaf distances decide updates; equal distances select the lowest
source face ID regardless of traversal order. Reuse one hierarchy for repeated
queries. Optional diagnostics report evaluated versus exhaustive triangle/query
pairs. `closest_surface(..., method='bvh')` constructs one for a single call;
default `auto` currently retains exhaustive projection pending broader evaluation.

Two tests cover reference parity on separated randomized triangles, spatial
pruning, equidistant source-order ties, empty queries and invalid query rejection.
`python -m tools.benchmark_proximity SOURCE.glb ...` compares real source meshes
on 256 deterministically perturbed vertex queries after canonical normalization.
The current observations are in `artifacts/proximity-benchmark.json`.

An initial version recomputed leaf distances with a different NumPy reduction,
which changed some source face IDs despite almost identical projected points.
Returning the original leaf solver's squared distances fixes this on both real
fixtures: IDs, projected positions and distances now match exhaustive traversal
exactly. Tests/fixtures do not establish all material-seam cases or production
speed. Single observed timings including build were roughly 0.118 s versus
0.119 s on Avocado and 0.657 s versus 0.774 s on WaterBottle. The hierarchy
visited 23,768/174,592 and 68,506/1,154,560 candidate pairs respectively. Python
traversal and per-leaf validation still limit speed; no <10-second production
preprocessing claim follows from these small query batches.
