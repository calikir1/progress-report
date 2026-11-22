"""Plot training loss and validation PSNR for one or more runs.

python plot_curves.py runs/clean runs/degraded
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

runs = sys.argv[1:]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
for r in runs:
    rows = list(csv.DictReader(open(Path(r) / 'log.csv')))
    it = [int(x['iter']) for x in rows]
    a1.plot(it, [float(x['sr_loss']) for x in rows], label=Path(r).name)
    v = [(int(x['iter']), float(x['val_psnr'])) for x in rows if x['val_psnr']]
    a2.plot(*zip(*v), marker='o', label=Path(r).name)
a1.set_xlabel('iteration'); a1.set_ylabel('training L1 loss'); a1.set_yscale('log')
a2.set_xlabel('iteration'); a2.set_ylabel('validation PSNR (dB, Y)')
for a in (a1, a2):
    a.grid(alpha=0.3); a.legend()
fig.tight_layout()
Path('results/figures').mkdir(parents=True, exist_ok=True)
fig.savefig('results/figures/training_curves.png', dpi=200)
print('saved results/figures/training_curves.png')
