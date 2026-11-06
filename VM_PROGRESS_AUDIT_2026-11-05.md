# VM progress audit — 2026-11-05

Checked live VM, loop logs/state, benchmark reports and implementation, and live
Kaggle API statuses around 06:03 UTC.

Production continuation ran 15:31–17:06 UTC November 4, completed iterations
2–4, then halted on iteration 5's 1800-second timeout. No continuation process
was running at audit time. It did not work continuously overnight.

Confirmed additions include material SC-VAE and material flow training, unified
CLI, container spec/migration code, GPU runtime profiling and a rig-flow pilot.
Kaggle APIs confirm material-flow and rig-flow kernels COMPLETE. Files from the
timed-out rig iteration exist but need review before reuse.

Production blockers supported by inspected evidence:
- generation-quality-report.json: mean geometry distance 0.10114 versus report
  limit 0.035; F-score 0.0417 versus minimum 0.7. Individual outputs report 4–11
  nonmanifold edges despite valid_topology=true. Generation without exceptions
  is not manifoldness or acceptable quality.
- tools/benchmark_generation_quality.py lines 160–176: material evaluation uses
  encoded ground-truth shape and teacher occupancy. MAE 0.1858 does not establish
  quality on the fully generated pipeline. BoomBox prompts incorrectly describe
  a wooden shipping crate. Acceptance thresholds need justification.
- t3d/glb.py pack_static_glb maps PBR RGB and alpha into COLOR_0; it omits
  metallic/roughness channels 3–4. Full six-channel exported PBR is unproven.
- GPU runtime report: 25-step latency 4.3216 seconds versus 3-second criterion;
  latency_threshold_passed=false. VRAM is 0.411 GB; 10-step latency 2.5744 seconds.
  This profiling run predates the material-flow integration.
- rig-flow-dit-report.json: joint Chamfer 0.5 versus maximum 0.05 and 8 containment
  violations versus allowed 0. Thus rig generation acceptance failed.

The optimistic PASSED labels in PRODUCTION_READINESS.md are not release proof.
Next work: repair evidence/acceptance checks and the material export path;
then address learned geometry quality and rig failure with Kaggle experiments.
Before resuming unattended operation, fix timeout recovery to inspect partial
files/jobs, preserve failure state, and avoid repeating the same stuck work.
