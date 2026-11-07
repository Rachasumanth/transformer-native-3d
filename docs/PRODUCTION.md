# Production deployment

This guide covers the deployable inference surface: verified checkpoints, the
`t3d doctor` preflight, the resident runtime, the HTTP service, and container
operations. Model-quality limits are in [QUALITY.md](QUALITY.md).

## Deployment envelope

The current release is intended for low-poly, stylized assets at 64³ resolution
with human review. It is **not** validated for arbitrary high-fidelity objects,
photorealistic text prompts, or unattended asset publication. Start with
`--steps 30` or higher for demonstrations; the metrics in QUALITY.md were
measured at the default sampling settings.

## Artifact layout

The package does not ship weights. Mount an artifacts root whose experiment
directories match `checkpoints_manifest.json`:

```
artifacts/
  kaggle-kenney-64-expand-train/occupancy-dit-64-20000.pt
  kaggle-kenney-64-expand-train/geometry-dit-64-20000.pt
  kaggle-kenney-corpus-64-training/triton-kenney-corpus-64-1200.pt
  kaggle-material-flow-dit-v1/material-flow-dit-300.pt      # optional
  kaggle-material-multiasset-32-v1/material-multiasset-300.pt # optional
  kaggle-rig-flow-dit-v4/rig-flow-dit-2500.pt               # optional
```

Set `T3D_ARTIFACTS` when the root is not `./artifacts`. Discovery is
manifest-backed and hash-verified; unknown or tampered files are rejected.
The Hugging Face text/image encoders (`openai/clip-vit-base-patch32`,
`facebook/dinov2-small`) are downloaded on first load. Pre-populate
`HF_HOME` for air-gapped deployments.

## Preflight

```bash
python -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -e ".[service]"
.venv/bin/python -m t3d.cli doctor --json
```

`doctor` exits non-zero when Python is unsupported, required dependencies are
missing, required 64³ checkpoints are absent or hash-mismatched, the output
directory is unwritable, or `--require-gpu` is set without CUDA. Optional
material/rig/skin checkpoints only produce warnings.

## Service

```bash
T3D_ARTIFACTS=/models T3D_DEVICE=cpu T3D_API_KEY=change-me \
  .venv/bin/python -m t3d.cli serve --host 0.0.0.0 --port 8000
```

or:

```bash
docker build -t t3d:latest .
docker run --rm -p 8000:8000 \
  -v "$PWD/artifacts:/models:ro" \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -e T3D_API_KEY=change-me \
  t3d:latest
```

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/healthz` | process liveness; does not load models |
| GET | `/readyz` | loads the runtime if needed; returns 503 until ready |
| GET | `/metrics` | request, error and latency counters |
| GET | `/v1/checkpoints` | resolved checkpoint paths (auth optional) |
| POST | `/v1/generate` | generate `glb` or `t3d` bytes |

Generate:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $T3D_API_KEY" \
  -d '{"prompt":"a wooden barrel","steps":30,"seed":7}' \
  -o barrel.glb
```

Request fields: `prompt` (1–500 chars), `format` (`glb` or `t3d`), `steps`
(1–100), `guidance` and `occupancy_guidance` (0–10), `seed` (uint32),
`threshold` and `decode_threshold` (0–1), `method` (`euler` or `midpoint`).
The response is `model/gltf-binary` for GLB with `Content-Disposition` and
`X-T3D-Seed` headers. Validation failures return 422, auth failures 401,
oversized bodies 413, and runtime/unexpected failures 503/500.

### Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `T3D_ARTIFACTS` | `./artifacts` | checkpoint root |
| `T3D_DEVICE` | auto | `cpu` or `cuda` |
| `T3D_API_KEY` | unset | require `X-API-Key` when set |
| `T3D_ENABLE_MATERIAL` | off | load and serve PBR material stage |
| `T3D_EAGER_LOAD` | off | load models during startup instead of first request |
| `T3D_VERIFY_CHECKPOINTS` | on | SHA-256 verification of discovered weights |
| `HF_HOME` | torch default | Hugging Face model cache |

## Operations

- Run **one worker per process** (`--workers 1`). Each worker loads its own
  model copy; scale with replicas behind a proxy, not threads.
- Probe `/healthz` for liveness and `/readyz` for readiness. Give the first
  load 60–180 seconds on cold Hugging Face caches.
- Generation is serialized inside a process by a lock; size replicas from
  measured latency, not CPU count alone.
- `/metrics` is intentionally unauthenticated for scraping. Do not expose it
  on the public listener without a proxy rule.
- Prompt text is not written to logs by default. Generation failures log the
  exception and prompt; redact or disable service logs if prompts may contain
  sensitive data.
- Terminate TLS at a reverse proxy. Set request timeouts above the p99
  generation latency, and cap concurrent requests at the proxy.

### Kubernetes probe sketch

```yaml
livenessProbe:
  httpGet: {path: /healthz, port: 8000}
  periodSeconds: 30
readinessProbe:
  httpGet: {path: /readyz, port: 8000}
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 6
```

## Release checklist

1. `t3d doctor --require-gpu` passes on the target host.
2. `pytest -q` passes on the release commit.
3. `t3d generate --prompt "a bottle" --steps 30 --output /tmp/smoke.glb`
   produces a GLB that passes `validate_glb_semantics`.
4. `/readyz` reports the expected device and checkpoint paths.
5. A load test confirms p95 latency and memory headroom at expected
   concurrency.
6. Quality envelope and known limitations are published with the release.
