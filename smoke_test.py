"""End-to-end check on tiny synthetic data (~1 minute). Run this first on Colab.

python smoke_test.py            # full pipeline including training
python smoke_test.py --no_torch # data + bicubic only
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from utils import write_rgb

NO_TORCH = '--no_torch' in sys.argv
T = Path('_smoke')
shutil.rmtree(T, ignore_errors=True)
rng = np.random.default_rng(0)


def synth(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    f = rng.uniform(0.02, 0.2, 3)
    img = np.stack([128 + 90 * np.sin(xx * f[0] + yy * f[1]),
                    128 + 90 * np.cos(yy * f[2]),
                    128 + 60 * np.sin((xx - yy) * f[0])], -1)
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


for i in range(6):
    write_rgb(T / 'train_hr' / f'{i:04d}.png', synth(200, 240))
for i in range(2):
    write_rgb(T / 'valid_hr' / f'{i:04d}.png', synth(200, 200))
for s, n in (('Set5', 2), ('Set14', 2)):
    for i in range(n):
        write_rgb(T / 'bench' / s / 'HR' / f'img{i}.png', synth(121, 97))   # odd sizes test modcrop


def run(*cmd):
    print('$', ' '.join(cmd))
    subprocess.run([sys.executable, *cmd], check=True)


run('prepare_data.py', '--src', str(T / 'train_hr'), '--dst', str(T / 'train_npy'))
run('prepare_data.py', '--src', str(T / 'valid_hr'), '--dst', str(T / 'valid_npy'))
run('check_data.py', '--train', str(T / 'train_hr'), '--bench', str(T / 'bench/Set5'), str(T / 'bench/Set14'),
    '--out', str(T / 'leak.csv'))
run('make_testsets.py', '--bench_root', str(T / 'bench'), '--sets', 'Set5', 'Set14', '--out', str(T / 'testsets'))
methods = ['bicubic']
if not NO_TORCH:
    for name, extra in (('clean', []), ('degaware', ['--model', 'degaware', '--degraded'])):
        run('train.py', '--name', name, '--runs', str(T / 'runs'), '--train_dir', str(T / 'train_npy'),
            '--val_dir', str(T / 'valid_npy'), '--iters', '40', '--val_every', '20', '--log_every', '10',
            '--patches_per_epoch', '320', '--batch', '8', '--workers', '0', '--val_crop', '96', *extra)
        methods.append(str(T / 'runs' / name / 'best.pt'))
    run('count_params.py')
    run('plot_curves.py', str(T / 'runs/clean'), str(T / 'runs/degaware'))
run('evaluate.py', '--methods', *methods, '--testsets', str(T / 'testsets'), '--out', str(T / 'results'))
run('visualize.py', '--methods', *methods, '--testsets', str(T / 'testsets'), '--set', 'Set5',
    '--image', 'img0.png', '--condition', 'b1.0_n5_q75', '--crop', '48', '--out', str(T / 'figs'))
print('\nSMOKE TEST PASSED')
