# Prepared-asset VAE training

The training command consumes a version-1 JSON manifest beside prepared `.t3d`
files. Each record contains `asset_id`, relative `path`, lowercase content
`sha256`, `split` (`train`, `validation`, `test`), `source`, and `license` strings.
These record supplied provenance; they do not perform license discovery or VLM
quality assessment. Asset paths must stay beneath the manifest directory.
Duplicate IDs, paths, or content hashes across the entire manifest are rejected.
Every load verifies the content hash. Optional `source_sha256` identifies the
original input bytes and `source_group` identifies a related asset family. If
either identity occurs in more than one split, the manifest is rejected;
multiple derived assets may share an identity within one split. These fields
are curator-supplied and optional, so missing or incorrect identities and
semantic duplicates remain undetected. Corpus curation still needs semantic
deduplication and reliable source-family labeling.

Example record (replace the hash with the actual file digest):

```json
{"version":1,"assets":[
  {"asset_id":"source-id","path":"prepared/asset.t3d","sha256":"<64 lowercase hex digits>",
   "split":"train","source":"original dataset and asset identifier","license":"original license"}
]}
```

The command requires nonempty train and held-out validation splits:

```sh
python -m t3d.training --manifest /kaggle/input/prepared-corpus/manifest.json \
  --mode shape --checkpoint /kaggle/working/shape.pt --steps 10000 --batch-size 2
# Continue to 20000 total steps with the identical manifest:
python -m t3d.training --manifest /kaggle/input/prepared-corpus/manifest.json \
  --mode shape --checkpoint /kaggle/working/shape.pt --steps 20000 --batch-size 2 --resume
```

Run the material instance separately with `--mode material`; all selected assets
must have PBR attributes. Batching adds batch indices, preserving geometry and
features. Shape split weights default to zero when absent, as an explicit
preprocessing policy; stored shape input has seven channels. Current batching
allows varying spatial grids, so curate a consistent grid resolution per corpus
for meaningful training.

Training uses stochastic posterior sampling, AdamW, tensor reconstruction,
occupancy BCE and latent KL. Validation uses posterior means and reports an
asset-weighted mean. Iteration order defaults to manifest order. Set
`--shuffle-seed 42` on a new run for deterministic per-epoch shuffling;
checkpoints store the seed and batch size, and resume rejects explicit
mismatches. Existing checkpoints without batch-size metadata cannot verify
the original batch size. Augmentation is not yet applied. Checkpoints are
replaced atomically after each update and
store model configuration/state, optimizer, CPU/CUDA RNG, epoch, cursor, steps,
and manifest fingerprint. Resume rejects a changed manifest. Use only trusted
project checkpoints; loading includes optimizer state. Exact replay assumes the
same software/hardware topology, batch size, input content and deterministic
operations. Device migration is not an exact-replay guarantee.

The CLI is a baseline training workflow, not the blueprint's completed training
protocol. Dataset acquisition/curation, rendering losses, mixed precision,
learning-rate scheduling, distributed execution, augmentation and high-throughput
sparse kernels remain pending. No curated real-asset training corpus has been
attached to the Kaggle notebook yet. Test fixtures are explicitly synthetic.

The private Kaggle pilot dataset now supplies two real CC0 Khronos integration
assets at 16³. Version 7 verified ten stochastic updates per model with Avocado
train and WaterBottle held out, including checkpoint reload. Results and model
states are under `artifacts/kaggle-v7`. The full curated training corpus and
production training protocol remain pending; this pilot is not a substitute.

An experimental float32 CUDA/Triton convolution can be selected for a new run
with `--convolution-backend triton`. The default remains `reference`. A resumed
checkpoint restores its recorded backend; if a backend is explicitly supplied
at resume, it must match. The Triton implementation currently builds neighbor
maps and computes backward gradients in PyTorch, supports float32 only, and has
been checked only on limited T4 probes. It is not yet a validated mixed-precision
or production-scale replacement.

Optional `--occupancy-level-weights W0 W1 W2 W3` changes the relative
contribution of decoder occupancy losses from coarsest to finest. Weights must
be nonnegative with a positive sum and are normalized, so equal weights match
the default mean. The weights are saved in checkpoints and must match when
explicitly supplied on resume. This changes the training objective and requires
separate evaluation of free-decoded support; it is not a validated accuracy fix.
