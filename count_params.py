"""Print parameter counts for every model configuration used in the report."""
from models import build_model, count_params

base = dict(channels=64, scale=2)
configs = [('SRCNN', dict(model='srcnn', blocks=0)),
           ('EDSR-lite, 4 blocks', dict(model='edsr', blocks=4)),
           ('EDSR-lite, 8 blocks', dict(model='edsr', blocks=8)),
           ('EDSR-lite, 9 blocks (param-matched control)', dict(model='edsr', blocks=9)),
           ('EDSR-lite, 16 blocks', dict(model='edsr', blocks=16)),
           ('Plain, 8 blocks', dict(model='plain', blocks=8)),
           ('DegAware-EDSR, 8 blocks', dict(model='degaware', blocks=8))]
for name, c in configs:
    m = build_model({**base, **c})
    print(f'{name:45s} {count_params(m):>10,}')
m = build_model({**base, 'model': 'degaware', 'blocks': 8})
est = count_params(m.estimator)
mod = sum(count_params(b.mod) for b in m.blocks)
print(f'  of which estimator {est:,}, modulation MLPs {mod:,}')
