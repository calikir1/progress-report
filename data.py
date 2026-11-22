"""Training patches sampled on the fly from DIV2K HR images stored as .npy (memory-mapped)."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import degrade as D

BLUR_MARGIN = D.BLUR_KSIZE // 2 + 1


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0


def to_uint8(t):
    """(3,H,W) float tensor in [0,1] -> HxWx3 uint8."""
    a = t.detach().clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return a.transpose(1, 2, 0)


class PatchDataset(Dataset):
    """Each item: a random HR crop -> (augment) -> degrade -> (lr, hr, d).

    One epoch = `patches_per_epoch` samples. Sampling is deterministic given
    (seed, epoch, index); call set_epoch() before each epoch.
    """

    def __init__(self, npy_dir, scale=2, lr_patch=48, degraded=False,
                 patches_per_epoch=16000, seed=0, n_images=None):
        self.files = sorted(Path(npy_dir).glob('*.npy'))
        if n_images:
            self.files = self.files[:n_images]
        assert self.files, f'no .npy files in {npy_dir} (run prepare_data.py)'
        self.scale, self.lr_patch, self.degraded = scale, lr_patch, degraded
        self.hr_patch = lr_patch * scale
        self.n, self.seed, self.epoch = patches_per_epoch, seed, 0
        self._arrays = None

    def set_epoch(self, e):
        self.epoch = e

    def _arr(self, i):
        if self._arrays is None:             # opened lazily inside each worker
            self._arrays = [np.load(f, mmap_mode='r') for f in self.files]
        return self._arrays[i]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        img = self._arr(int(rng.integers(len(self.files))))
        h, w = img.shape[:2]
        m = BLUR_MARGIN if self.degraded else 0
        P = self.hr_patch + 2 * m
        y, x = int(rng.integers(0, h - P + 1)), int(rng.integers(0, w - P + 1))
        hr = np.array(img[y:y + P, x:x + P])

        # augmentation on HR before degradation == identical flips/rotations on LR and HR
        if rng.random() < 0.5:
            hr = hr[:, ::-1]
        if rng.random() < 0.5:
            hr = hr[::-1, :]
        hr = np.ascontiguousarray(np.rot90(hr, int(rng.integers(4))))

        if self.degraded:
            sb, sn, q = D.sample_params(rng)
            blurred = D.gaussian_blur(hr, sb)                    # blur with real context
            hr, blurred = hr[m:-m, m:-m], blurred[m:-m, m:-m]
            lr = D.imresize(blurred, 1.0 / self.scale)
            lr = D.jpeg(D.add_noise(lr, sn, rng), q)
        else:
            sb, sn, q = D.CLEAN_PARAMS
            lr = D.imresize(hr, 1.0 / self.scale)
        d = D.normalize_params(sb, sn, q)
        return to_tensor(lr), to_tensor(hr), torch.from_numpy(d)
