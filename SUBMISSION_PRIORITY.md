# Immediate user priority
The user must submit this repository soon. Preserve full production intent, but
prepare an honest reviewable submission now; do not label the prototype production
ready. Read VM_PROGRESS_AUDIT_2026-11-05.md before trusting previous gate claims.

First turn: save an inventory of code/checkpoints and prepare SUBMISSION_STATUS.md
with exact supported commands, setup, current measured results, limitations and
remaining blockers. Correct misleading gate labels without deleting historical
evidence. Inspect git state; create a local repository if absent, with appropriate
ignores for secrets, environments and large artifacts. Do not publish or push.
Build a source-only submission archive and checksum, excluding credentials,
profiles, private logs, environments, datasets and model binaries. Include a
checkpoint manifest with hashes and reproducible retrieval instructions instead.
Validate package build and available smoke tests, recording failures honestly.

Next priorities: fix PBR export dropping metallic/roughness, invalid topology
acceptance, mislabeled prompts and ground-truth-assisted benchmark claims. Then
improve learned geometry and rig quality with Kaggle GPU experiments. Do not
replace the production objective with more tests, fabricated milestones or
arbitrary relaxed thresholds. Each turn must leave durable work and exact next
steps. Use the existing Kaggle account; honor quotas and cooldowns.

## Confirmed deadline
User confirmed submission within two days on 2026-11-05. Produce a source archive and honest status in the FIRST turn, then improve it. Preserve a usable submission throughout. Reserve the final hours for reproducibility and documentation; do not start speculative long training near the deadline.

## Prototype verified and parallel refinement authorized
On 2026-11-05 the real three-stage checkpoint CLI ran successfully for the duck
prompt, producing /tmp/t3d-prototype-check.glb with 8725 vertices and 1718 faces.
A working research prototype exists. Focus on refining this toward production;
do not rebuild research demos or spend repeated turns on packaging alone.
The user explicitly encourages parallel independent work across authorized
Gemini and Kaggle accounts with available capacity. Do not assume the cached
kaggle-status registry is live GPU quota proof; verify before scheduling.

Gemini profile 2 is fixing PBR export in /home/ubuntu/t3d-parallel/export.
Gemini profile 3 is fixing benchmark correctness in /home/ubuntu/t3d-parallel/evaluation.
They work in isolated copies. Before touching those areas, check their systemd
service status and REPORT.md files. Review completed changes with git diff and
new-file listings; integrate only verified changes and rerun relevant checks.
Do not overwrite parallel edits or call a worker complete just because it exited.
The primary worker should focus on learned geometry quality diagnosis, Kaggle
experiment design and submission evidence while these workers are active.
Independent Kaggle experiments may use available authorized accounts concurrently;
track account/kernel/version and prevent duplicate jobs. Respect each account's
service limits. Do not globally switch credentials during concurrent work.
