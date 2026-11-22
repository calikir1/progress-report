"""PSNR / SSIM on the Y channel with a `scale`-pixel border crop, as in SR papers."""
import numpy as np
from skimage.metrics import structural_similarity


def rgb2y(img):
    """RGB uint8 (or float 0-255) -> Y in [16, 235], BT.601 as in MATLAB rgb2ycbcr."""
    x = img.astype(np.float64) / 255.0
    return 16.0 + 65.481 * x[..., 0] + 128.553 * x[..., 1] + 24.966 * x[..., 2]


def _prep(sr, hr, crop):
    assert sr.shape == hr.shape, (sr.shape, hr.shape)
    a, b = rgb2y(sr), rgb2y(hr)
    if crop > 0:
        a, b = a[crop:-crop, crop:-crop], b[crop:-crop, crop:-crop]
    return a, b


def psnr_y(sr, hr, crop):
    a, b = _prep(sr, hr, crop)
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim_y(sr, hr, crop):
    a, b = _prep(sr, hr, crop)
    return structural_similarity(a, b, data_range=255, gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False)
