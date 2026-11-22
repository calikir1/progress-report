# Learned Single-Image Super-Resolution with Realistic Degradation (APS360)

EDSR-lite (8 residual blocks, 64 channels, no batch norm, PixelShuffle) for 2x super-resolution,
plus a degradation-aware variant whose estimator branch predicts blur / noise / JPEG levels from
the LR input and scales the channels of every residual block.

## Quick start (Google Colab, GPU runtime)

Open `run_colab.ipynb` in Colab and run the cells top to bottom. Or, by hand:

```bash
pip install -r requirements.txt
# data
wget -q http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -P data && unzip -q data/DIV2K_train_HR.zip -d data
wget -q http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip -P data && unzip -q data/DIV2K_valid_HR.zip -d data
wget -q https://cv.snu.ac.kr/research/EDSR/benchmark.tar -P data && tar -xf data/benchmark.tar -C data
python prepare_data.py --src data/DIV2K_train_HR --dst data/div2k_train_npy
python prepare_data.py --src data/DIV2K_valid_HR --dst data/div2k_valid_npy

python smoke_test.py                                   # 1-minute end-to-end check
python check_data.py --train data/DIV2K_train_HR data/DIV2K_valid_HR \
       --bench data/benchmark/Set5 data/benchmark/Set14 data/benchmark/B100
python make_testsets.py --bench_root data/benchmark --sets Set5 Set14 B100   # ONCE
python evaluate.py --methods bicubic                    # bicubic baseline table
python count_params.py

python train.py --name clean    --model edsr --iters 30000
python train.py --name degraded --model edsr --degraded --iters 30000
python evaluate.py --methods bicubic runs/clean/best.pt runs/degraded/best.pt
python plot_curves.py runs/clean runs/degraded
python visualize.py --set Set14 --image comic.png --condition clean --methods bicubic runs/clean/best.pt runs/degraded/best.pt
```

Benchmark folders may be laid out as `Set5/HR/*.png` or `Set5/*.png`; both work.

## Data pipeline

* Training: DIV2K 800 HR images (validation: first 10 DIV2K-valid images, 480x480 center crop).
* Each sample: random 96x96 HR crop -> random flip / 90-degree rotation -> (optional degradation) -> 48x48 LR.
* One epoch = 16,000 patches (1,000 iterations at batch 16).
* LR generation uses a MATLAB-compatible bicubic `imresize` (`imresize.py`) so numbers match the literature.
* Realistic degradation (training): Gaussian blur sigma ~ U(0.2, 2.0) on HR (21x21 kernel) -> bicubic x1/2 ->
  Gaussian noise sigma ~ U(0, 10) (0-255 scale) -> JPEG quality ~ U{40..95}.
* Metrics: PSNR / SSIM on Y (BT.601), `scale` pixels cropped from each border.

## Fixed degraded test set

`make_testsets.py` applies a 3x3 grid (blur sigma {0.5, 1.0, 2.0} x noise sigma {0, 5, 10}, JPEG q = 75)
to every Set5 / Set14 / B100 image, with seed 0 and a per-(image, condition) noise seed. The parameters
of every file are in `testsets/degraded_params.csv`, and `testsets/MANIFEST.txt` stores a SHA-256 of
the whole folder. The same files are used for every experiment; the script refuses to overwrite them.

## Models (`models.py`)

| Model | Flag | Notes |
| --- | --- | --- |
| EDSR-lite | `--model edsr` | 8 ResBlocks, global skip, PixelShuffle x2 |
| Plain stack | `--model plain` | ablation (a): same convolutions, no skip connections |
| DegAware-EDSR | `--model degaware` | estimator LR -> d_hat in [0,1]^3; each block: x + s(d_hat) * body(x) |
| Oracle | `--model oracle` | true degradation params instead of d_hat (upper bound) |
| SRCNN | `--model srcnn` | 9-1-5 baseline on bicubic-upsampled input |

DegAware loss: L1(SR, HR) + 0.1 * MSE(d_hat, d).

## Data sources and licenses

* DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/ (academic research use)
* Set5, Set14, B100: standard SR benchmarks, cited in the report.
