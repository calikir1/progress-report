"""Train EDSR-lite / plain / DegAware-EDSR / SRCNN.

python train.py --name clean    --model edsr --iters 30000
python train.py --name degraded --model edsr --degraded --iters 30000
python train.py --name degaware --model degaware --degraded --iters 30000
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import degrade as D
from data import PatchDataset, to_tensor, to_uint8
from metrics import psnr_y
from models import build_model, count_params
from utils import set_seed

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--model', default='edsr', choices=['edsr', 'plain', 'degaware', 'oracle', 'srcnn'])
ap.add_argument('--blocks', type=int, default=8)
ap.add_argument('--channels', type=int, default=64)
ap.add_argument('--scale', type=int, default=2)
ap.add_argument('--degraded', action='store_true', help='train on blur/noise/JPEG-degraded LR')
ap.add_argument('--loss', default='l1', choices=['l1', 'l2'])
ap.add_argument('--lambda_deg', type=float, default=0.1)
ap.add_argument('--train_dir', default='data/div2k_train_npy')
ap.add_argument('--val_dir', default='data/div2k_valid_npy')
ap.add_argument('--val_n', type=int, default=10, help='DIV2K-valid images used for validation')
ap.add_argument('--val_crop', type=int, default=480, help='HR center crop for fast validation')
ap.add_argument('--lr_patch', type=int, default=48)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--patches_per_epoch', type=int, default=16000)
ap.add_argument('--iters', type=int, default=30000)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--decay_every', type=int, default=10000, help='halve LR every N iterations')
ap.add_argument('--val_every', type=int, default=2500)
ap.add_argument('--log_every', type=int, default=100)
ap.add_argument('--workers', type=int, default=2)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--n_images', type=int, default=None, help='use only the first N training images')
ap.add_argument('--runs', default='runs')
ap.add_argument('--resume', action='store_true')
ap.add_argument('--no_amp', action='store_true')
args = ap.parse_args()
cfg = vars(args)

set_seed(args.seed)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = dev.type == 'cuda' and not args.no_amp
run = Path(args.runs) / args.name
run.mkdir(parents=True, exist_ok=True)

model = build_model(cfg).to(dev)
n_params = count_params(model)
cfg['n_params'] = n_params
(run / 'config.json').write_text(json.dumps(cfg, indent=2))
print(f'{args.model}: {n_params:,} parameters | device {dev} | amp {use_amp}')

opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.decay_every, gamma=0.5)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
is_deg_model = args.model in ('degaware', 'oracle')

ds = PatchDataset(args.train_dir, args.scale, args.lr_patch, args.degraded,
                  args.patches_per_epoch, args.seed, args.n_images)
iters_per_epoch = args.patches_per_epoch // args.batch
print(f'{len(ds.files)} training images | {args.patches_per_epoch} patches/epoch '
      f'= {iters_per_epoch} iterations/epoch | {args.iters / iters_per_epoch:.1f} epochs')

# ---- fixed validation set (degraded with a fixed seed if training is degraded)
val = []
for i, f in enumerate(sorted(Path(args.val_dir).glob('*.npy'))[:args.val_n]):
    hr = np.load(f)
    h, w = hr.shape[:2]
    c = min(args.val_crop, h, w) // args.scale * args.scale
    hr = np.ascontiguousarray(hr[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c])
    rng = np.random.default_rng([1234, i])
    if args.degraded:
        lr = D.degrade(hr, args.scale, *D.sample_params(rng), rng)
    else:
        lr = D.imresize(hr, 1.0 / args.scale)
    val.append((lr, hr))


@torch.no_grad()
def validate():
    model.eval()
    ps = []
    for lr, hr in val:
        out = model(to_tensor(lr).unsqueeze(0).to(dev))
        if isinstance(out, tuple):
            out = out[0]
        ps.append(psnr_y(to_uint8(out[0].float()), hr, args.scale))
    model.train()
    return float(np.mean(ps))


it, best = 0, -1.0
log_path = run / 'log.csv'
if args.resume and (run / 'last.pt').exists():
    ck = torch.load(run / 'last.pt', map_location=dev, weights_only=False)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
    sched.load_state_dict(ck['sched']); scaler.load_state_dict(ck['scaler'])
    it, best = ck['iter'], ck['best']
    print(f'resumed at iteration {it} (best val {best:.3f} dB)')
elif log_path.exists():
    log_path.unlink()


def save(path):
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                    scaler=scaler.state_dict(), iter=it, best=best, cfg=cfg), path)


def log(row):
    new = not log_path.exists()
    with open(log_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['iter', 'epoch', 'loss', 'sr_loss', 'deg_loss', 'lr',
                                          'val_psnr', 'elapsed_s'])
        if new:
            w.writeheader()
        w.writerow(row)


model.train()
t0 = time.time()
run_sr = run_deg = 0.0
n_acc = 0
while it < args.iters:
    epoch = it // iters_per_epoch
    ds.set_epoch(epoch)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, drop_last=True,
                        num_workers=args.workers, pin_memory=dev.type == 'cuda')
    for lr_b, hr_b, d_b in loader:
        lr_b, hr_b, d_b = lr_b.to(dev, non_blocking=True), hr_b.to(dev, non_blocking=True), d_b.to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            if is_deg_model:
                sr, d_hat = model(lr_b, d_b)
            else:
                sr, d_hat = model(lr_b), None
            sr_loss = F.l1_loss(sr, hr_b) if args.loss == 'l1' else F.mse_loss(sr, hr_b)
        deg_loss = F.mse_loss(d_hat.float(), d_b) if d_hat is not None else torch.zeros((), device=dev)
        loss = sr_loss.float() + args.lambda_deg * deg_loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        it += 1
        run_sr += sr_loss.item(); run_deg += deg_loss.item(); n_acc += 1

        val_now = it % args.val_every == 0 or it == args.iters
        if it % args.log_every == 0 or val_now:
            vp = validate() if val_now else ''
            row = dict(iter=it, epoch=epoch, loss=f'{(run_sr + args.lambda_deg * run_deg) / n_acc:.5f}',
                       sr_loss=f'{run_sr / n_acc:.5f}', deg_loss=f'{run_deg / n_acc:.5f}',
                       lr=f'{sched.get_last_lr()[0]:.2e}',
                       val_psnr=f'{vp:.3f}' if vp != '' else '', elapsed_s=f'{time.time() - t0:.0f}')
            log(row)
            print(' '.join(f'{k}={v}' for k, v in row.items() if v != ''))
            run_sr = run_deg = 0.0
            n_acc = 0
            if val_now:
                if vp > best:
                    best = vp
                    save(run / 'best.pt')
                save(run / 'last.pt')
        if it >= args.iters:
            break

print(f'done: best val PSNR {best:.3f} dB -> {run / "best.pt"}')
