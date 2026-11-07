"""Shared grid and quantization constants for FORGE-3D corpora and models."""

GRID = 64
FINE = 128
SDF_CLIP_VOXELS = 4.0
SDF_QUANT_MAX = 127.0
N_SURFACE = 15000
N_QUERY = 30000
N_INPUT_POINTS = 2048
MAX_FACES = 300000
SANITIZE_FACES = 60000
BIT_BYTES = (GRID ** 3 + 7) // 8


def pack_bits(grid):
    import numpy as np
    return np.packbits(np.asarray(grid, dtype=bool).reshape(-1))


def unpack_bits(packed, grid=GRID):
    import numpy as np
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))
    return bits[:grid ** 3].reshape(grid, grid, grid).astype(np.float32)


def quantize_sdf(sdf_world, fine=FINE):
    import numpy as np
    voxels = np.asarray(sdf_world, dtype=np.float64) * fine
    scaled = voxels / SDF_CLIP_VOXELS * SDF_QUANT_MAX
    return np.clip(np.rint(scaled), -SDF_QUANT_MAX, SDF_QUANT_MAX).astype(np.int8)


def dequantize_sdf(sdf_i8, fine=FINE):
    import numpy as np
    return np.asarray(sdf_i8, dtype=np.float32) / SDF_QUANT_MAX * SDF_CLIP_VOXELS / fine


def quantize_normals(normals):
    import numpy as np
    n = np.asarray(normals, dtype=np.float64)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    return np.clip(np.rint(n * SDF_QUANT_MAX), -SDF_QUANT_MAX, SDF_QUANT_MAX).astype(np.int8)


def dequantize_normals(normals_i8):
    import numpy as np
    n = np.asarray(normals_i8, dtype=np.float32) / SDF_QUANT_MAX
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
