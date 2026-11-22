"""Build the FIXED clean and degraded test sets once. Never regenerate them afterwards.

Degraded grid: blur sigma {0.5, 1.0, 2.0} x noise sigma {0, 5, 10}, JPEG quality 75,
applied to every Set5 / Set14 / B100 image (9 conditions per image). Noise uses a
per-(image, condition) seed derived from --seed, so the files are reproducible.

python make_testsets.py --bench_root data/benchmark --sets Set5 Set14 B100 --out testsets
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from degrade import degrade
from imresize import imresize
from utils import list_images, modcrop, read_rgb, sha256_of_dir, stable_int, write_rgb

BLURS = (0.5, 1.0, 2.0)
NOISES = (0, 5, 10)
JPEG_Q = 75

ap = argparse.ArgumentParser()
ap.add_argument('--bench_root', required=True)
ap.add_argument('--sets', nargs='+', default=['Set5', 'Set14', 'B100'])
ap.add_argument('--out', default='testsets')
ap.add_argument('--scale', type=int, default=2)
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

out = Path(args.out)
if (out / 'degraded_params.csv').exists():
    raise SystemExit(f'{out} already exists. The test set must stay fixed; delete it only on purpose.')

rows = []
conds = [(b, n) for b in BLURS for n in NOISES]
for set_name in args.sets:
    files = list_images(Path(args.bench_root) / set_name)
    print(f'{set_name}: {len(files)} images')
    for p in files:
        hr = modcrop(read_rgb(p), args.scale)
        name = p.stem + '.png'
        write_rgb(out / 'clean' / set_name / 'HR' / name, hr)
        write_rgb(out / 'clean' / set_name / 'LR' / name, imresize(hr, 1.0 / args.scale))
        for ci, (b, n) in enumerate(conds):
            cond = f'b{b}_n{n}_q{JPEG_Q}'
            noise_seed = [args.seed, stable_int(f'{set_name}/{name}'), ci]
            lr = degrade(hr, args.scale, b, n, JPEG_Q, np.random.default_rng(noise_seed))
            write_rgb(out / 'degraded' / set_name / cond / name, lr)
            rows.append(dict(set=set_name, image=name, condition=cond, sigma_blur=b,
                             sigma_noise=n, jpeg_q=JPEG_Q,
                             noise_seed='-'.join(map(str, noise_seed))))

with open(out / 'degraded_params.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

digest = sha256_of_dir(out)
(out / 'MANIFEST.txt').write_text(
    f'scale={args.scale} seed={args.seed} blurs={BLURS} noises={NOISES} jpeg_q={JPEG_Q}\n'
    f'sets={args.sets}\nimages_degraded={len(rows)}\nsha256={digest}\n')
print(f'{len(rows)} degraded LR images written; sha256 {digest}')
print('Commit degraded_params.csv and MANIFEST.txt; record the sha256 in the report.')
