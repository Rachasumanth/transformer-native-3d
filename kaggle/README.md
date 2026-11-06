# Kaggle GPU execution

Private notebook: https://www.kaggle.com/code/sumu143/transformer-native-3d-gpu-bootstrap

Kaggle notebook execution is the intended GPU backend for this project. Local
checks use CPU; trainable PyTorch/CUDA work should run in a GPU notebook.

Regenerate the upload bundle after source changes:

```sh
.venv/bin/python -m tools.build_kaggle_notebook
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4 --timeout 1200
kaggle kernels status sumu143/transformer-native-3d-gpu-bootstrap
kaggle kernels output sumu143/transformer-native-3d-gpu-bootstrap -p artifacts/kaggle-bootstrap
```

The builder includes only t3d Python modules, tests, pyproject.toml, and the
roundtrip demo. It embeds a zip with a SHA256 digest, excluding credentials,
local environments, research documents, and unrelated files. Authentication uses
the local Kaggle CLI credential store; no API token is embedded in the notebook.
Metadata requests a private notebook, internet for dependencies, and T4 GPUs.

The notebook asserts CUDA availability rather than silently falling back to CPU,
runs all bundled tests, creates the reference mesh/container artifacts, and runs
30 synthetic MLP optimization steps on CUDA. Outputs include gpu-report.json,
unit-tests.log and a synthetic-probe.pt checkpoint. This verifies execution and
gradients; it is not SC-VAE training or evidence of reconstruction quality.
GPU allocation remains subject to the account's Kaggle availability/quota.

The notebook now attaches the private `sumu143/transformer-native-3d-pilot`
dataset: prepared CC0 Avocado (train) and WaterBottle (validation) plus their
upstream notices. `tools/train_corpus_probe.py` runs ten stochastic VAE updates
per mode, checkpoint reload and held-out evaluation, writing `corpus-probe.json`
and `shape-pilot.pt` / `material-pilot.pt`. This tiny integration pilot does not
replace the blueprint's curated training corpus or establish model quality.
