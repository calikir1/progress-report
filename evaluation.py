"""Shared evaluation over the fixed test sets written by make_testsets.py.

Layout:
  testsets/clean/<Set>/HR/*.png        modcropped HR
  testsets/clean/<Set>/LR/*.png        MATLAB-bicubic LR
  testsets/degraded/<Set>/<cond>/*.png degraded LR (HR shared with clean/)
  testsets/degraded_params.csv
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from degrade import denormalize_params
from imresize import imresize
from metrics import psnr_y, ssim_y
from utils import read_rgb


def load_params(root):
    rows = {}
    with open(Path(root) / 'degraded_params.csv') as f:
        for r in csv.DictReader(f):
            rows[(r['set'], r['condition'], r['image'])] = r
    return rows


def iter_cases(root, include_clean=True, include_degraded=True):
    """Yields (set, condition, image_name, lr_path, hr_path)."""
    root = Path(root)
    for set_dir in sorted((root / 'clean').iterdir()):
        hr_dir = set_dir / 'HR'
        if include_clean:
            for hr in sorted(hr_dir.glob('*.png')):
                yield set_dir.name, 'clean', hr.name, set_dir / 'LR' / hr.name, hr
        if include_degraded and (root / 'degraded' / set_dir.name).is_dir():
            for cond_dir in sorted((root / 'degraded' / set_dir.name).iterdir()):
                for lr in sorted(cond_dir.glob('*.png')):
                    yield set_dir.name, cond_dir.name, lr.name, lr, hr_dir / lr.name


def evaluate(sr_fn, root, scale, include_clean=True, include_degraded=True, per_image_csv=None):
    """sr_fn(lr_uint8) -> (sr_uint8, d_hat or None). Returns summary rows."""
    params = load_params(root) if include_degraded else {}
    acc = defaultdict(list)
    per_image = []
    for set_name, cond, name, lr_p, hr_p in iter_cases(root, include_clean, include_degraded):
        lr, hr = read_rgb(lr_p), read_rgb(hr_p)
        sr, d_hat = sr_fn(lr)
        p, s = psnr_y(sr, hr, scale), ssim_y(sr, hr, scale)
        row = dict(set=set_name, condition=cond, image=name, psnr=p, ssim=s)
        if d_hat is not None and cond != 'clean':
            pb, pn, pq = denormalize_params(d_hat)
            t = params[(set_name, cond, name)]
            row.update(blur_err=abs(pb - float(t['sigma_blur'])),
                       noise_err=abs(pn - float(t['sigma_noise'])),
                       jpeg_err=abs(pq - float(t['jpeg_q'])))
        per_image.append(row)
        group = 'clean' if cond == 'clean' else 'degraded'
        acc[(set_name, cond)].append(row)
        acc[(set_name, f'{group}_all')].append(row)
    summary = []
    for (set_name, cond), rows in sorted(acc.items()):
        out = dict(set=set_name, condition=cond, n=len(rows),
                   psnr=np.mean([r['psnr'] for r in rows]),
                   ssim=np.mean([r['ssim'] for r in rows]))
        if 'blur_err' in rows[0]:
            for k in ('blur_err', 'noise_err', 'jpeg_err'):
                out[k + '_mae'] = np.mean([r[k] for r in rows])
        summary.append(out)
    if per_image_csv:
        write_csv(per_image_csv, per_image)
    return summary


def bicubic_fn(scale):
    return lambda lr: (imresize(lr, scale), None)


def model_fn(model, device):
    import torch
    from data import to_tensor, to_uint8

    @torch.no_grad()
    def fn(lr):
        x = to_tensor(lr).unsqueeze(0).to(device)
        out = model(x)
        d_hat = None
        if isinstance(out, tuple):
            out, d = out
            d_hat = d[0].float().cpu().numpy()
        return to_uint8(out[0].float()), d_hat
    return fn


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})
