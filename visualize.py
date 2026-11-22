"""Side-by-side crops: LR | bicubic | models... | HR, PSNR/SSIM of the whole image underneath.

The crop is chosen by a fixed rule (highest gradient energy in the HR image), not by hand.

python visualize.py --set Set14 --image comic.png --condition clean \
    --methods bicubic runs/clean/best.pt runs/degraded/best.pt
python visualize.py --set B100 --image 253027.png --condition b1.0_n5_q75 --methods ...
python visualize.py --phone_dir data/phone --methods bicubic runs/clean/best.pt runs/degraded/best.pt
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluation import bicubic_fn, model_fn
from imresize import imresize
from metrics import psnr_y, ssim_y
from utils import list_images, read_rgb

ap = argparse.ArgumentParser()
ap.add_argument('--methods', nargs='+', required=True)
ap.add_argument('--testsets', default='testsets')
ap.add_argument('--set')
ap.add_argument('--image')
ap.add_argument('--condition', default='clean')
ap.add_argument('--phone_dir', help='real photos: each is downscaled x2 and super-resolved (no ground truth)')
ap.add_argument('--crop', type=int, default=96, help='crop size in HR pixels')
ap.add_argument('--scale', type=int, default=2)
ap.add_argument('--out', default='results/figures')
args = ap.parse_args()


def load_fns():
    fns = {}
    for m in args.methods:
        if m == 'bicubic':
            fns['Bicubic'] = bicubic_fn(args.scale)
            continue
        import torch
        from models import build_model
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ck = torch.load(m, map_location=dev, weights_only=False)
        model = build_model(ck['cfg']).to(dev).eval()
        model.load_state_dict(ck['model'])
        fns[Path(m).parent.name] = model_fn(model, dev)
    return fns


def pick_crop(img, c):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    e = cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2
    e = cv2.boxFilter(e, -1, (c, c), normalize=True)
    h, w = g.shape
    e = e[c // 2:h - c // 2, c // 2:w - c // 2]
    y, x = np.unravel_index(np.argmax(e), e.shape)
    return int(y), int(x)


def figure(lr, ref, outs, title, path, ref_is_gt=True):
    c = min(args.crop, ref.shape[0], ref.shape[1])
    y, x = pick_crop(ref, c)
    lr_up = cv2.resize(lr, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)
    panels = [('LR (nearest)', lr_up, '')]
    for name, sr in outs.items():
        sub = f'{psnr_y(sr, ref, args.scale):.2f} dB / {ssim_y(sr, ref, args.scale):.4f}' if ref_is_gt else ''
        panels.append((name, sr, sub))
    panels.append(('HR' if ref_is_gt else 'Original photo', ref, ''))
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(2.6 * (len(panels) + 1), 3.0))
    full = ref.copy()
    cv2.rectangle(full, (x, y), (x + c, y + c), (255, 0, 0), max(2, ref.shape[0] // 150))
    axes[0].imshow(full); axes[0].set_title(title, fontsize=8)
    for ax, (name, img, sub) in zip(axes[1:], panels):
        ax.imshow(img[y:y + c, x:x + c], interpolation='nearest')
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(sub, fontsize=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print('saved', path)


fns = load_fns()
if args.phone_dir:
    for p in list_images(args.phone_dir):
        photo = read_rgb(p)
        h, w = photo.shape[:2]
        s = min(1.0, 1200 / max(h, w))                      # keep inference memory small
        if s < 1:
            photo = imresize(photo, s)
        photo = photo[:photo.shape[0] // args.scale * args.scale, :photo.shape[1] // args.scale * args.scale]
        lr = imresize(photo, 1.0 / args.scale)
        outs = {n: f(lr)[0] for n, f in fns.items()}
        figure(lr, photo, outs, p.name, f'{args.out}/phone_{p.stem}.png', ref_is_gt=False)
else:
    root = Path(args.testsets)
    hr = read_rgb(root / 'clean' / args.set / 'HR' / args.image)
    lr_path = (root / 'clean' / args.set / 'LR' / args.image if args.condition == 'clean'
               else root / 'degraded' / args.set / args.condition / args.image)
    lr = read_rgb(lr_path)
    outs = {n: f(lr)[0] for n, f in fns.items()}
    figure(lr, hr, outs, f'{args.set}/{args.image} ({args.condition})',
           f'{args.out}/{args.set}_{Path(args.image).stem}_{args.condition}.png')
