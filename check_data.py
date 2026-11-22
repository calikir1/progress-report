"""Count images and check that no benchmark image (near-)duplicates a training image.

Uses a 64-bit difference hash (dHash); pairs with Hamming distance <= --thresh are flagged.

python check_data.py --train data/DIV2K_train_HR data/DIV2K_valid_HR --bench data/benchmark/Set5 data/benchmark/Set14 data/benchmark/B100
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from utils import list_images, read_rgb


def dhash(img, size=8):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return np.packbits(bits)


def hamming(a, b):
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


ap = argparse.ArgumentParser()
ap.add_argument('--train', nargs='+', required=True)
ap.add_argument('--bench', nargs='+', required=True)
ap.add_argument('--thresh', type=int, default=6)
ap.add_argument('--out', default='results/leakage_check.csv')
args = ap.parse_args()

train = []
for d in args.train:
    files = list_images(d)
    print(f'{d}: {len(files)} images')
    for p in files:
        train.append((p, dhash(read_rgb(p))))   # dHash is resize-invariant

hits = []
for d in args.bench:
    files = list_images(d)
    print(f'{d}: {len(files)} images')
    for p in files:
        h = dhash(read_rgb(p))
        for tp, th in train:
            dist = hamming(h, th)
            if dist <= args.thresh:
                hits.append(dict(benchmark=str(p), train=str(tp), hamming=dist))

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['benchmark', 'train', 'hamming'])
    w.writeheader()
    w.writerows(hits)
print(f'leakage candidates: {len(hits)} (written to {args.out}; inspect any by eye)')
