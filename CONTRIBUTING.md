# Contributing to Transformer-Native 3D (T3D)

Thank you for your interest in contributing to T3D!

## Development Setup

1. Clone repository and create Python 3.11/3.12 environment:
```bash
make install
```
2. Verify system preflight:
```bash
make doctor
```
3. Run test suite:
```bash
make test
```

## Code Quality Standards

- All new features must include pytest unit tests under `tests/`.
- Maintain 100% 2-manifold mesh generation requirements (0 non-manifold edges).
- Checkpoints must be registered with exact SHA-256 digests in `checkpoints_manifest.json`.
- Adhere to PEP 8 standards with type hints where applicable.

## Pull Request Checklist

1. Preflight check: `t3d doctor` passes.
2. Full test suite passes: `pytest -q`.
3. Quality benchmark runs without regression:
```bash
python -m tools.benchmark_generation_quality --output artifacts/test-quality-report.json
```
4. Update relevant documentation in `docs/` and `README.md`.
