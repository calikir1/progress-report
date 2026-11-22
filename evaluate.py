"""Evaluate bicubic and/or checkpoints on the fixed test sets; print a PSNR/SSIM table.

python evaluate.py --methods bicubic runs/clean/best.pt runs/degraded/best.pt
"""
import argparse
import json
from pathlib import Path

from evaluation import bicubic_fn, evaluate, model_fn, write_csv

ap = argparse.ArgumentParser()
ap.add_argument('--methods', nargs='+', required=True, help="'bicubic' or paths to .pt checkpoints")
ap.add_argument('--testsets', default='testsets')
ap.add_argument('--scale', type=int, default=2)
ap.add_argument('--out', default='results')
ap.add_argument('--clean_only', action='store_true')
args = ap.parse_args()


def label(m):
    return 'bicubic' if m == 'bicubic' else Path(m).parent.name


all_rows = []
for m in args.methods:
    name = label(m)
    if m == 'bicubic':
        fn = bicubic_fn(args.scale)
    else:
        import torch
        from models import build_model, count_params
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ck = torch.load(m, map_location=dev, weights_only=False)
        model = build_model(ck['cfg']).to(dev).eval()
        model.load_state_dict(ck['model'])
        print(f'{name}: {count_params(model):,} params, trained {ck["iter"]} iterations')
        fn = model_fn(model, dev)
    rows = evaluate(fn, args.testsets, args.scale, include_degraded=not args.clean_only,
                    per_image_csv=f'{args.out}/per_image_{name}.csv')
    for r in rows:
        r['method'] = name
    all_rows += rows
    write_csv(f'{args.out}/eval_{name}.csv', rows)

write_csv(f'{args.out}/summary.csv', all_rows)

# ---- markdown table: method x (set, clean | degraded average)
ORDER = {'Set5': 0, 'Set14': 1, 'B100': 2, 'BSD100': 2, 'Urban100': 3}
sets = sorted({r['set'] for r in all_rows}, key=lambda s: (ORDER.get(s, 9), s))
cols = [(s, c) for c in ('clean_all', 'degraded_all') for s in sets]
cols = [c for c in cols if any(r['set'] == c[0] and r['condition'] == c[1] for r in all_rows)]
head = '| Method | ' + ' | '.join(f'{s} {"clean" if c == "clean_all" else "degraded"}' for s, c in cols) + ' |'
lines = [head, '|' + ' --- |' * (len(cols) + 1)]
for m in dict.fromkeys(r['method'] for r in all_rows):
    cells = []
    for s, c in cols:
        r = next((r for r in all_rows if r['method'] == m and r['set'] == s and r['condition'] == c), None)
        cells.append(f"{r['psnr']:.2f} / {r['ssim']:.4f}" if r else '')
    lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
table = '\n'.join(lines)
print('\nPSNR (dB) / SSIM, Y channel, border crop = scale\n' + table)
Path(args.out, 'table.md').write_text(table + '\n')

est = [r for r in all_rows if 'blur_err_mae' in r and r['condition'] == 'degraded_all']
if est:
    print('\nEstimator MAE (blur sigma, noise sigma, JPEG q):')
    for r in est:
        print(f"  {r['method']} {r['set']}: {r['blur_err_mae']:.3f}, {r['noise_err_mae']:.3f}, {r['jpeg_err_mae']:.2f}")
