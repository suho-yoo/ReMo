"""Merge ordered collector shards, optionally retaining the first N successful clips."""
import argparse
from pathlib import Path
import torch
p=argparse.ArgumentParser()
p.add_argument('--inputs',nargs='+',required=True)
p.add_argument('--output',required=True)
p.add_argument('--n',type=int,default=0)
a=p.parse_args()
shards=[torch.load(f,map_location='cpu',weights_only=False) for f in a.inputs]
indices=torch.tensor([i for s in shards for i in s['indices']])
paths=[v for s in shards for v in s['paths']]
assert len(set(indices.tolist())) == len(indices), 'Duplicate indices'
assert len(set(paths)) == len(paths), 'Duplicate paths'
order=indices.argsort()
if a.n:
    assert len(order)>=a.n, 'Too few successful clips'
    order=order[:a.n]
    assert all(max(s['indices'])>=int(indices[order[-1]]) for s in shards), 'Shard ended before global cutoff; collect more'
out={k:torch.cat([s[k] for s in shards])[order] for k in ['audio','video','text']}
assert all(torch.isfinite(v).all() for v in out.values())
out.update({f'mu_{k}':out[k].mean(0) for k in ['audio','video','text']})
out.update(indices=indices[order].tolist(),paths=[paths[i] for i in order.tolist()])
classes=torch.unique(out['text'],dim=0).shape[0]
if not a.n:
    assert classes==309, f'Expected 309 classes, got {classes}'
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
torch.save(out,a.output)
print(f'Merged {len(order)} clips, {classes} classes -> {a.output}')
