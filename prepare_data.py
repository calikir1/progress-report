"""Convert DIV2K HR PNGs to .npy so training can memory-map random crops quickly.

python prepare_data.py --src data/DIV2K_train_HR --dst data/div2k_train_npy
python prepare_data.py --src data/DIV2K_valid_HR --dst data/div2k_valid_npy
"""
import argparse
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

from utils import list_images, read_rgb

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--dst', required=True)
args = ap.parse_args()

dst = Path(args.dst)
dst.mkdir(parents=True, exist_ok=True)
files = list_images(args.src)
for p in tqdm(files):
    out = dst / (p.stem + '.npy')
    if not out.exists():
        np.save(out, read_rgb(p))
print(f'{len(files)} images -> {dst}')
