"""CPU atlas-gap inpainting independent of CUDA rasterization."""
import numpy as np


def pad_pbr(image,valid):
    """Fill uncovered texels per channel with Telea; preserve covered values.

    Image is linear RGB/metal/rough/alpha in [0,1]. Input arrays are unchanged.
    At least one covered texel is required; an all-covered atlas needs no OpenCV.
    """
    source=np.asarray(image)
    valid=np.asarray(valid)
    if source.ndim!=3 or source.shape[-1]!=6 or min(source.shape[:2])<1:
        raise ValueError('image must be nonempty (H,W,6)')
    if not np.issubdtype(source.dtype,np.floating) or not np.isfinite(source).all() or ((source<0)|(source>1)).any():
        raise ValueError('image must contain finite floats in [0,1]')
    if valid.shape!=source.shape[:2] or valid.dtype!=np.bool_:
        raise ValueError('valid mask must be boolean (H,W)')
    if not valid.any():raise ValueError('atlas has no covered texels')
    output=source.astype(np.float32,copy=True)
    if valid.all():return output
    import cv2
    holes=(~valid).astype(np.uint8)
    for channel in range(6):
        # Telea's float correction is not scale invariant: [0,1] channels can
        # drift badly even for constant fields. Use the plan's byte-map domain.
        byte=np.rint(output[...,channel]*255).astype(np.uint8)
        filled=cv2.inpaint(byte,holes,3,cv2.INPAINT_TELEA).astype(np.float32)/255
        # Numerical extrapolation must not invent material extrema. This also
        # preserves constant material channels exactly across atlas padding.
        known=source[...,channel][valid]
        output[...,channel]=np.clip(filled,known.min(),known.max())
    output=np.clip(output,0,1)
    output[valid]=source[valid]
    return output
