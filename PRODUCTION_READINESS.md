# t3d production delivery objective

The goal is a production-ready t3d system described in plan.md, not test-count
increases, infrastructure completion, or more isolated research pilots.
The VM coordinates Gemini 3.8 Flash High through AGY; ALL GPU experiments run
on Kaggle via its APIs. Preserve existing work and verified checkpoints.

## First action and release scope
Audit the actual implementation and latest downloaded experiment outputs against
every step of plan.md. Maintain a release gap table in this file with requirement,
current evidence, remaining blocker, next action and acceptance evidence. Never
mark an item complete because code exists or a synthetic pilot passes. Preserve
the blueprint's complete scope; staged releases must clearly state limitations.

## Required release gates
1. Versioned container/schema: documented interoperable spec, bounds/size checks,
   malformed input rejection, compatibility/migration policy, deterministic
   round trips for geometry, all six PBR channels, optional articulation and BPT.
2. Data and models: licensed provenance, validated deduplicated source-disjoint
   splits; reproducible preprocessing and resumable Kaggle training; verified
   checkpoint/config/data identities. Scale beyond pilots based on measured
   learning curves and available Kaggle resources, never merely more steps.
3. Integrated generation: real prompt/image to occupancy to geometry to material
   to t3d to usable GLB; separately validate advertised rigging/animation and
   optional topology generation. Match stage coordinate systems, grids and latent
   dimensions. Test complete learned inference, not ground-truth shortcuts.
4. Quality: frozen held-out assets/prompts, geometric fidelity/topology, PBR
   render fidelity, prompt adherence, diversity, rig deformation, failure rate
   and comparisons with existing checkpoints/baselines. Establish explicit
   justified acceptance thresholds before selecting on results; label proposed
   thresholds until justified. Prompt sensitivity alone is not usable generation.
5. Runtime: real asset end-to-end latency, peak memory and throughput on named
   Kaggle GPU hardware and supported resolutions. High-resolution claims require
   measured complete paths, not synthetic extraction timings. Handle empty,
   invalid and over-budget outputs with useful errors.
6. Delivery: installable package and reproducible dependency environment, public
   CLI/API for validated flows, checkpoint distribution/integrity, usage examples,
   operational recovery, licensing, release notes and a fresh-environment smoke
   test on held-out real assets. Provide a release evidence manifest.

## Iteration selection
Choose the highest-impact unresolved blocker on the critical path to integrated
usable generation. Prioritize debugging and evaluating the current learned
stages, missing material/rig stages and real export over cosmetic cleanup or
adding tests for their own sake. Tests support delivered behavior.

Each turn must update the gap table and PROGRESS.md with artifacts, commands,
Kaggle kernel/version/status, measured outcome, failed assumptions and next
critical action. Poll/retrieve active experiments before duplicate submissions.
Pending jobs are pending, not successful experiments. Inspect any interrupted
turn's files and jobs before retrying. Do not discard earlier progress.

## Completion
Only report production-ready when all advertised release gates have reproducible
acceptance evidence. Otherwise state remaining blockers candidly and continue
useful work. If Kaggle quotas, compute or missing inputs prevent progress, persist
an exact resumable checkpoint and blocker. Do not invent results or silently
reduce the blueprint to a smaller research demo.
