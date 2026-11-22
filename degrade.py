"""LR generation: optional blur (on HR) -> bicubic down -> noise -> JPEG.

All images are RGB uint8 HxWx3. Every function returns the parameters it used so
the degradation estimator can be trained against them.
"""
import cv2
import numpy as np

from imresize import imresize

# Training ranges (proposal): blur sigma on HR pixels, noise sigma on 0-255, JPEG quality
BLUR_RANGE = (0.2, 2.0)
NOISE_RANGE = (0.0, 10.0)
JPEG_RANGE = (40, 95)
BLUR_KSIZE = 21

# Clean setting is encoded as blur 0, noise 0, quality 100
CLEAN_PARAMS = (0.0, 0.0, 100)


def gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (BLUR_KSIZE, BLUR_KSIZE), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT)


def add_noise(img, sigma, rng):
    if sigma <= 0:
        return img
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


def jpeg(img, quality):
    if quality >= 100:
        return img
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    assert ok
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def degrade(hr, scale, blur_sigma, noise_sigma, jpeg_q, rng):
    """Deterministic given its arguments and rng state. Returns LR uint8."""
    x = gaussian_blur(hr, blur_sigma)
    x = imresize(x, 1.0 / scale)
    x = add_noise(x, noise_sigma, rng)
    x = jpeg(x, jpeg_q)
    return x


def sample_params(rng):
    return (float(rng.uniform(*BLUR_RANGE)),
            float(rng.uniform(*NOISE_RANGE)),
            int(rng.integers(JPEG_RANGE[0], JPEG_RANGE[1] + 1)))


def normalize_params(blur_sigma, noise_sigma, jpeg_q):
    """Map to roughly [0, 1] targets for the estimator's auxiliary loss."""
    return np.array([blur_sigma / BLUR_RANGE[1],
                     noise_sigma / NOISE_RANGE[1],
                     (100 - jpeg_q) / (100 - JPEG_RANGE[0])], dtype=np.float32)


def denormalize_params(d):
    d = np.asarray(d, dtype=np.float64)
    return (d[..., 0] * BLUR_RANGE[1],
            d[..., 1] * NOISE_RANGE[1],
            100 - d[..., 2] * (100 - JPEG_RANGE[0]))
