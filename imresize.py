"""MATLAB-compatible bicubic imresize (with antialiasing when downscaling).

SR papers generate LR images with MATLAB's imresize; OpenCV's INTER_CUBIC does not
antialias, which shifts bicubic PSNR by ~0.3-1 dB. Using this keeps numbers
comparable to the literature (e.g. Set5 x2 bicubic = 33.66 dB).
"""
import numpy as np


def _cubic(x):
    a = np.abs(x)
    a2, a3 = a ** 2, a ** 3
    return ((1.5 * a3 - 2.5 * a2 + 1) * (a <= 1)
            + (-0.5 * a3 + 2.5 * a2 - 4 * a + 2) * ((a > 1) & (a <= 2)))


def _weights_indices(in_len, out_len, scale, kernel_width=4.0, antialias=True):
    if scale < 1 and antialias:
        kernel_width = kernel_width / scale
    x = np.arange(1, out_len + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    p = int(np.ceil(kernel_width)) + 2
    idx = left[:, None] + np.arange(p)[None, :]          # 1-based
    dist = u[:, None] - idx
    if scale < 1 and antialias:
        w = scale * _cubic(dist * scale)
    else:
        w = _cubic(dist)
    w = w / w.sum(axis=1, keepdims=True)
    idx = idx.astype(np.int64) - 1                         # 0-based
    idx = np.where(idx < 0, -idx - 1, idx)                 # symmetric padding
    idx = np.where(idx >= in_len, 2 * in_len - idx - 1, idx)
    keep = np.any(w != 0, axis=0)                          # drop all-zero taps
    return w[:, keep], idx[:, keep]


def imresize(img, scale):
    """img: HxW or HxWxC, uint8 or float. scale: e.g. 0.5 or 2.
    Returns the same dtype (uint8 is rounded and clipped like MATLAB)."""
    in_dtype = img.dtype
    x = img.astype(np.float64)
    squeeze = x.ndim == 2
    if squeeze:
        x = x[:, :, None]
    h, w, _ = x.shape
    oh, ow = int(np.ceil(h * scale)), int(np.ceil(w * scale))
    wh, ih = _weights_indices(h, oh, scale)
    x = np.einsum('op,opwc->owc', wh, x[ih])
    ww, iw = _weights_indices(w, ow, scale)
    x = np.einsum('op,hopc->hoc', ww, x[:, iw])
    if squeeze:
        x = x[:, :, 0]
    if in_dtype == np.uint8:
        return np.clip(np.round(x), 0, 255).astype(np.uint8)
    return x.astype(in_dtype)
