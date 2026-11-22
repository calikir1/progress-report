import torch
import torch.nn as nn
import torch.nn.functional as F

DIV2K_MEAN = (0.4488, 0.4371, 0.4040)


class MeanShift(nn.Module):
    def __init__(self, sign):
        super().__init__()
        self.register_buffer('mean', torch.tensor(DIV2K_MEAN).view(1, 3, 1, 1) * sign)

    def forward(self, x):
        return x + self.mean


class ResBlock(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class PlainBlock(nn.Module):
    """Same convolutions as ResBlock but no skip (ablation a)."""
    def __init__(self, c=64):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.body(x)


class Upsampler(nn.Sequential):
    """ESPCN sub-pixel upsampling. scale 2 or 3: one step; scale 4: two x2 steps."""
    def __init__(self, c, scale):
        layers = []
        if scale in (2, 3):
            layers += [nn.Conv2d(c, c * scale ** 2, 3, padding=1), nn.PixelShuffle(scale)]
        elif scale == 4:
            for _ in range(2):
                layers += [nn.Conv2d(c, c * 4, 3, padding=1), nn.PixelShuffle(2)]
        else:
            raise ValueError(scale)
        super().__init__(*layers)


class EDSRLite(nn.Module):
    def __init__(self, n_blocks=8, c=64, scale=2, residual=True):
        super().__init__()
        self.sub_mean, self.add_mean = MeanShift(-1), MeanShift(+1)
        self.head = nn.Conv2d(3, c, 3, padding=1)
        block = ResBlock if residual else PlainBlock
        self.body = nn.Sequential(*[block(c) for _ in range(n_blocks)],
                                  nn.Conv2d(c, c, 3, padding=1))
        self.up = Upsampler(c, scale)
        self.tail = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, x):
        f = self.head(self.sub_mean(x))
        f = f + self.body(f)                      # global skip
        return self.add_mean(self.tail(self.up(f)))


# ---------------------------------------------------------------- degradation-aware

class DegEstimator(nn.Module):
    """LR patch (B,3,H,W) -> d_hat (B,n_out) in [0,1]: normalized blur, noise, JPEG."""
    def __init__(self, c=32, n_out=3):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, c, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, c, 3, 2, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, 2 * c, 3, 2, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(2 * c, 64), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, n_out), nn.Sigmoid())

    def forward(self, x):
        return self.f(x)


class ModResBlock(nn.Module):
    """Residual block whose branch is scaled per channel by s(d) in (0, 2)."""
    def __init__(self, c=64, d=3):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(c, c, 3, padding=1))
        self.mod = nn.Sequential(nn.Linear(d, c), nn.ReLU(inplace=True), nn.Linear(c, c))
        nn.init.zeros_(self.mod[-1].weight)       # start as identity: s = 1
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, d):
        s = 2 * torch.sigmoid(self.mod(d))[:, :, None, None]
        return x + s * self.body(x)


class DegAwareEDSR(nn.Module):
    def __init__(self, n_blocks=8, c=64, scale=2, n_deg=3, oracle=False):
        super().__init__()
        self.oracle = oracle
        self.sub_mean, self.add_mean = MeanShift(-1), MeanShift(+1)
        self.estimator = DegEstimator(n_out=n_deg)
        self.head = nn.Conv2d(3, c, 3, padding=1)
        self.blocks = nn.ModuleList([ModResBlock(c, n_deg) for _ in range(n_blocks)])
        self.body_tail = nn.Conv2d(c, c, 3, padding=1)
        self.up = Upsampler(c, scale)
        self.tail = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, x, d_true=None):
        d_hat = self.estimator(x)
        d = d_true if (self.oracle and d_true is not None) else d_hat
        f0 = self.head(self.sub_mean(x))
        f = f0
        for blk in self.blocks:
            f = blk(f, d)
        f = f0 + self.body_tail(f)
        return self.add_mean(self.tail(self.up(f))), d_hat


# ---------------------------------------------------------------- SRCNN baseline

class SRCNN(nn.Module):
    """Dong et al. 2014 (9-1-5). Operates on the bicubic-upsampled input."""
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(nn.Conv2d(3, 64, 9, padding=4), nn.ReLU(inplace=True),
                                 nn.Conv2d(64, 32, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(32, 3, 5, padding=2))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        return x + self.net(x)


def build_model(cfg):
    name = cfg['model']
    if name == 'edsr':
        return EDSRLite(cfg['blocks'], cfg['channels'], cfg['scale'], residual=True)
    if name == 'plain':
        return EDSRLite(cfg['blocks'], cfg['channels'], cfg['scale'], residual=False)
    if name in ('degaware', 'oracle'):
        return DegAwareEDSR(cfg['blocks'], cfg['channels'], cfg['scale'], oracle=(name == 'oracle'))
    if name == 'srcnn':
        return SRCNN(cfg['scale'])
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
