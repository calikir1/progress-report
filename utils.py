import hashlib
import os
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def list_images(d):
    """Images in d, or in d/HR if that exists (EDSR benchmark layout)."""
    d = Path(d)
    if (d / 'HR').is_dir():
        d = d / 'HR'
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def modcrop(img, scale):
    h, w = img.shape[:2]
    return img[:h - h % scale, :w - w % scale]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_of_dir(d):
    h = hashlib.sha256()
    for p in sorted(Path(d).rglob('*')):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def stable_int(s):
    """Deterministic int from a string (Python's hash() is salted per process)."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
