# Track B autoregressive model

`t3d.bpt_ar.BPTAutoregressiveTransformer` implements a causal Transformer with
cross-attention to a point-cloud condition. An XYZ MLP and learned query tokens
pool an unordered point cloud into a fixed number of condition tokens. Each
decoder block applies causal self-attention, cross-attention, and a feedforward
layer. The input and output embeddings use the local BPT vocabulary and three
reserved IDs: BOS 5120, EOS 5121, PAD 5122.

`teacher_forcing_batch` constructs complete shifted input/target streams with
right padding. Loss ignores PAD targets; BOS predicts the first geometry token,
and the final geometry token predicts EOS. Active tokens never attend to later
padding because self-attention is causal. Conditioning points must be finite
and have shape `(batch, points, 3)`.

The current baseline uses learned absolute token positions and full attention
within its configured context. It rejects longer inputs. It has no sliding
cache across context windows, Hourglass shortening, grammar-constrained sampling or DPO stage yet.
Corpus training and useful held-out generation remain open; the local held-out probe below fails.

`generate` runs one asset at a time using per-layer KV caches and a point
condition encoded once. It supports greedy decoding or temperature sampling,
excludes BOS/PAD predictions, and returns geometry tokens plus an explicit stop
reason: `eos`, `token_limit` or `context_limit`. Only `eos` means the model
terminated its stream. Tokens must still pass `t3d.bpt.decode` before mesh use.
Generation requires evaluation mode and cannot exceed the learned position
range. Cached logits are checked against full causal forward logits.

The dedicated private Kaggle notebook is
https://www.kaggle.com/code/sumu143/transformer-native-3d-bpt-ar-probe.
Regenerate and submit it using:

```sh
.venv/bin/python -m tools.build_bpt_ar_notebook
kaggle kernels push -p kaggle-bpt-ar --accelerator NvidiaTeslaT4 --timeout 1200
```

The notebook contains only model code, focused tests, the cleaned CC0 Avocado
fixture and its preserved source notice. It asserts CUDA availability, checks
causality, point permutation invariance, padding loss and conditioning gradients,
then performs 100 optimization steps using 4,096 area-sampled conditioning
points and the complete Avocado token stream. It writes a source hash, fixture
hash, loss history, training accuracy and checkpoint. This is a one-asset
optimization probe; it provides no held-out or generated mesh quality claim.

Version 1 completed on a Tesla T4 (PyTorch 2.11.0+cu130 / CUDA 13.0). All
three focused tests passed. After 100 updates, loss decreased from 8.70884 to
0.210439, with approximately 100% teacher-forced training token accuracy. The
source-matched report, logs and checkpoint are in `artifacts/kaggle-bpt-ar-v1`.
This measures memorization under teacher forcing; free-running sampling and
held-out accuracy still require separate measurements.

Version 2 completed on the same T4 runtime. All five tests passed, including
cached/full causal-logit parity. After checkpoint reload, greedy generation
reproduced all 1,550 Avocado tokens exactly and terminated with EOS, yielding
363 vertices and 681 faces. Held-out Box teacher-forced accuracy was 0% (loss
10.2779); Food Lime accuracy was 0.05127% (loss 10.1292), measured across its
entire stream with the documented context-window resets. Both held-out point
clouds produced the exact Avocado training tokens and mesh. The independent
comparison is `artifacts/kaggle-bpt-ar-v2/independent-generation-audit.json`.
This demonstrates that the single-asset model memorizes the training sequence
and fails to follow held-out geometry conditions. The full goal requires
multi-asset training and useful held-out generation.

Sampling now accepts `grammar_constrained=True`. This optional mask prevents
malformed fan patches and zero-area quantized triangles and enables EOS only
after a complete patch. It does not establish manifold topology or shape quality.
A context/token limit can still leave an incomplete patch. CPU grammar tests
pass; model integration awaits CUDA verification. The multiasset v1 baseline
failed decoding on all four generated streams, each reaching its context limit.
