# VM continuation workflow

The AWS account2 t3d VM coordinates this project with AGY CLI using
`gemini-3.8-flash-high`. All training and GPU experiments run on Kaggle GPUs
through the installed Kaggle API profiles, as in previous sessions. VM CPU
PyTorch is for tests and lightweight validation only.

Read `note.md`, the latest `PROGRESS.md` entries, and relevant experiment reports
before choosing work. Continue from the existing Kaggle kernels, datasets and
verified checkpoints. Check active kernel status before submitting duplicate
work. Preserve train/validation separation and record kernel versions, metrics,
artifact hashes and failures in PROGRESS.md. Retrieve and verify outputs before
claiming an experiment succeeded. Respect provider quotas and cooldowns.

Kaggle experiment submission, polling and output retrieval are authorized parts
of this continuation workflow. Do not switch training to the VM. Keep each loop
iteration bounded and persist its result before proceeding.
