#!/usr/bin/env python3
"""Fit per-resolution per-epoch train-loop costs for e5 (plain compile) and e1 (mega graph).

Model per recipe: train_loop = a + sum_res epochs_at(res) * c_res
Then decompose c_res = f + k * (res/32)^2  -> f = res-independent per-epoch overhead.
"""
import json, glob
import numpy as np

E5_SHIPPED = [28]*5 + [32]*4
E1_SHIPPED = [24]*3 + [28]*3 + [32]*3

def epochs_by_res(schedule, train_epochs):
    out = {}
    full = int(train_epochs)
    for e in range(full):
        r = schedule[min(e, len(schedule)-1)]
        out[r] = out.get(r, 0) + 1.0
    frac = train_epochs - full
    if frac > 1e-9:
        r = schedule[min(full, len(schedule)-1)]
        out[r] = out.get(r, 0) + frac
    return out

def load_cells():
    cells = []
    for d in sorted(glob.glob('runners/results/2026*/summary.json')):
        s = json.load(open(d))
        cfg = json.load(open(d.replace('summary.json', 'config.json')))
        if not s.get('valid') or s['time']['n'] < 8:
            continue
        ov = cfg.get('overrides') or {}
        cells.append(dict(label=s['label'], target=cfg['target'], ov=ov,
                          diag=s['diagnostics']['train_loop']['mean_s'],
                          n=s['time']['n'], acc=s['acc_mean']))
    return cells

def fit(recipe_cells, resolutions):
    # unknowns: a, c_res for each res
    rows, y = [], []
    for c in cells_sel(recipe_cells):
        pass
    A = []
    for c in recipe_cells:
        row = [1.0] + [c['ep'].get(r, 0.0) for r in resolutions]
        A.append(row);
    A = np.array(A); y = np.array([c['diag'] for c in recipe_cells])
    w = np.sqrt(np.array([c['n'] for c in recipe_cells], dtype=float))
    coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    resid = A @ coef - y
    return coef, resid

def cells_sel(cells):
    return cells

cells = load_cells()

def prep(target_prefix, default_sched, default_epochs):
    out = []
    for c in cells:
        if not c['target'].startswith(target_prefix):
            continue
        sched = c['ov'].get('res_schedule', None)
        if target_prefix == 'e5_full32':
            sched = [32]*9
        if sched is None:
            sched = default_sched
        te = c['ov'].get('train_epochs', default_epochs)
        c = dict(c); c['ep'] = epochs_by_res(sched, te)
        out.append(c)
    return out

# e5 family: targets e5 (shipped 28->32 unless res_schedule override) and e5_full32
e5_cells = prep('e5', E5_SHIPPED, 8.125)
for c in e5_cells:
    if c['target'] == 'e5_full32':
        c['ep'] = epochs_by_res([32]*9, c['ov'].get('train_epochs', 8.125))
e5_res = [24, 25, 28, 32]
have = [c for c in e5_cells]
coef, resid = fit(have, e5_res)
print("=== e5 (plain torch.compile; eager optimizer between steps) ===")
print(f"cells={len(have)}  const a={coef[0]*1000:.1f} ms")
c = dict(zip(e5_res, coef[1:]))
for r in e5_res:
    print(f"  c{r} = {c[r]*1000:6.1f} ms/epoch   ratio vs c32: {c[r]/c[32]:.3f}   theory (r/32)^2 = {(r/32)**2:.3f}")
for cc, rr in zip(have, resid):
    print(f"    {cc['label']:32s} diag={cc['diag']:.4f} resid={rr*1000:+5.1f} ms  ep={cc['ep']}")

# e1 family: e1a_*mega cells only (same graph mode), plus iso_full32_s252_mega
e1_cells = []
for cand in cells:
    lbl = cand['label']
    if cand['target'] == 'e1a_prog_mega':
        sched = cand['ov'].get('res_schedule', E1_SHIPPED)
        te = cand['ov'].get('train_epochs', 8.875)
    elif cand['target'] == 'e1a_full32_mega' or lbl == 'iso_full32_s252_mega_n200':
        sched = [32]*9
        te = cand['ov'].get('train_epochs', 8.875)
    else:
        continue
    cand = dict(cand); cand['ep'] = epochs_by_res(sched, te)
    e1_cells.append(cand)
e1_res = [24, 28, 32]
coef1, resid1 = fit(e1_cells, e1_res)
print("\n=== e1 (whole-run CUDA graph) ===")
print(f"cells={len(e1_cells)}  const a={coef1[0]*1000:.1f} ms")
c1 = dict(zip(e1_res, coef1[1:]))
for r in e1_res:
    print(f"  c{r} = {c1[r]*1000:6.1f} ms/epoch   ratio vs c32: {c1[r]/c1[32]:.3f}   theory (r/32)^2 = {(r/32)**2:.3f}")
for cc, rr in zip(e1_cells, resid1):
    print(f"    {cc['label']:32s} diag={cc['diag']:.4f} resid={rr*1000:+5.1f} ms  ep={cc['ep']}")

# implied per-epoch resolution-independent overhead f and compute k: c_r = f + k*(r/32)^2
for name, cc, rset in (("e5", c, [24, 28, 32]), ("e1", c1, [24, 28, 32])):
    A = np.array([[1.0, (r/32)**2] for r in rset])
    y = np.array([cc[r] for r in rset])
    (f, k), *_ = np.linalg.lstsq(A, y, rcond=None)
    print(f"\n{name}: per-epoch overhead f = {f*1000:.1f} ms ({f/ (f+k) *100:.0f}% of a 32px epoch), compute k = {k*1000:.1f} ms"
          f"  -> per-step overhead ~{f*1000/32:.2f} ms")
