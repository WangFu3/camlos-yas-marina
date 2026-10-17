#!/usr/bin/env python3
"""Robustness test: online runs on synthetic appearance changes of selected laps.

    python3 scripts/stress_test.py BAG --conds none,bright,overexpose,lowcontrast,lights_off,blur   # val 3, 7
    python3 scripts/stress_test.py BAG --laps 2,6 --maps maps_no26 --init-from out/lolo/run3.csv \
        --conds none,invert --roma 0,1 --out out/roma_tune                                          # tuning

For each condition and lap, run.py runs online from 20 s before the lap to its end, with every image
transformed (camloc/augment.py). The initial pose is the final online run's own causal estimate at
that moment (no GT). Nothing is tuned on these results.
"""
import argparse, os, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
PY = '/home/kunting/test/Download/ENTER/envs/camloc/bin/python'


def score(gt, sub, lap):
    """evaluate.py metrics on one lap (GT cropped to the lap window)."""
    g = gt[(gt.timestamp_ns >= lap.t0_ns) & (gt.timestamp_ns <= lap.t1_ns)]
    s = sub[(sub.timestamp_ns >= lap.t0_ns) & (sub.timestamp_ns <= lap.t1_ns)]
    t = s.timestamp_ns.values
    gyaw = np.unwrap(g['yaw'].values)
    gx, gy, gh = (np.interp(t, g.timestamp_ns, v) for v in (g.x, g.y, gyaw))
    dx, dy = s.x.values - gx, s.y.values - gy
    lon, lat = np.cos(gh) * dx + np.sin(gh) * dy, -np.sin(gh) * dx + np.cos(gh) * dy
    dyaw = np.degrees(np.angle(np.exp(1j * (s['yaw'].values - gh))))
    pos = np.hypot(dx, dy)
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    return dict(rmse=rms(pos), lat=rms(lat), lon=rms(lon), yaw=rms(dyaw), within1m=float(np.mean(pos < 1) * 100),
                max_err=float(pos.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--conds', default='none,bright,overexpose,lowcontrast,lights_off,blur')
    ap.add_argument('--out', default='out/stress')
    ap.add_argument('--laps', default='3,7')
    ap.add_argument('--maps', default='maps')
    ap.add_argument('--init-from', default='out/final/final.csv', help='causal online run used for the start pose')
    ap.add_argument('--roma', default='0', help='comma list of --roma-fallback values to compare, e.g. 0,1')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    laps = pd.read_csv('cache/laps.csv').set_index('lap')
    gt = pd.read_csv('gt.csv')
    rows = []
    for cond, roma, l in [(c, r, int(l)) for c in a.conds.split(',') for r in a.roma.split(',') for l in a.laps.split(',')]:
            lap = laps.loc[l]
            out = f'{a.out}/{cond}_roma{roma}_lap{l}.csv'
            if not os.path.exists(out):
                subprocess.run([PY, os.path.join(ROOT, 'run.py'), a.bag, out, '--work', 'cache', '--cams', 'fl,rl',
                                '--maps', a.maps, '--pf-config', 'configs/fl_rl.json', '--transform', cond,
                                '--roma-fallback', roma,
                                '--t-start', str(int(lap.t0_ns - 20e9)), '--t-end', str(int(lap.t1_ns)),
                                '--init-from', a.init_from], check=True)
            log = pd.read_csv(out.replace('.csv', '_log.csv'))
            inl = log[(log.timestamp_ns >= lap.t0_ns) & (log.timestamp_ns <= lap.t1_ns)]
            lg = inl[~inl.cam.str.endswith('_roma')]
            ro = inl[inl.cam.str.endswith('_roma')]
            r = dict(cond=cond, roma=int(roma), lap=l, **score(gt, pd.read_csv(out), lap),
                     fix_ok=float(((lg.ok == 1) & (lg.n_inl >= 15)).mean() * 100),
                     roma_calls=len(ro), roma_ok=int(((ro.ok == 1) & (ro.n_inl >= 15)).sum()))
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f'{a.out}/summary.csv', index=False)
    print(res.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
