#!/usr/bin/env python
"""Accuracy and speed of a pipeline variant on the held-out tuning laps only (2 and 6).

    python scripts/eval_variant.py BAG cache --cams fl,rl --n-kf 1 --merge 0 --every fl:1,rl:2 --tag v1

1. Relocalizes the image sets of the tuning laps with the variant, against maps that exclude those laps
   (maps_no26), and times each set (ALIKED + LightGlue + PnP; decode is timed too).
   Keyframe selection uses the causal online estimate of the earlier held-out run (out/lolo/run3.csv).
2. Replays the filter over the whole run with the frozen parameters (configs/fl_rl.json): fixes on the
   tuning laps come from the variant, elsewhere from the held-out run's log (same cameras).
3. Reports position RMSE on laps 2 and 6. Validation laps are never touched.
"""
import argparse, json, os, sys, time
import cv2
import numpy as np
import pandas as pd
import torch
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
from camloc.camera import load_cameras
from camloc.body_masks import body_masks
from camloc.features import Features
from camloc.kfmap import KeyframeMap
from camloc.pf import PFParams
from run_pf import run_filter

TUNE = [2, 6]


def sets_in(bag, cams, t0, t1):
    """(stamp, {cam: jpeg}) image sets within [t0, t1], grouped within 5 ms around the first camera."""
    by = {c: [] for c in cams}
    with open(bag, 'rb') as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, ch, _, msg in r.iter_decoded_messages(
                topics=[f'/constructor0/sensor/camera_{c}/compressed_image' for c in cams], start_time=t0 - 10**8, end_time=t1):
            t = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if t0 <= t <= t1:
                by[ch.topic.split('camera_')[1].split('/')[0]].append((t, bytes(msg.data)))
    ref = sorted(by[cams[0]])
    rt = np.array([t for t, _ in ref])
    out = [[t, {cams[0]: j}] for t, j in ref]
    for c in cams[1:]:
        for t, j in by[c]:
            i = int(np.argmin(np.abs(rt - t)))
            if abs(rt[i] - t) <= 5_000_000:
                out[i][1][c] = j
                out[i][0] = max(out[i][0], t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag'); ap.add_argument('cache')
    ap.add_argument('--cams', default='fl,rl')
    ap.add_argument('--n-kf', type=int, default=2)
    ap.add_argument('--merge', type=int, default=0)
    ap.add_argument('--every', default='', help='per-camera frame stride, e.g. fl:1,rl:2')
    ap.add_argument('--kpts', type=int, default=2048)
    ap.add_argument('--maps', default='maps_no26')
    ap.add_argument('--ref', default='out/lolo/run3')
    ap.add_argument('--config', default='configs/fl_rl.json')
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    cams = a.cams.split(',')
    every = {c: 1 for c in cams}
    every.update({kv.split(':')[0]: int(kv.split(':')[1]) for kv in a.every.split(',') if kv})
    laps = pd.read_csv(f'{a.cache}/laps.csv').set_index('lap')
    assert not (laps.loc[TUNE, 'split'] == 'val').any()

    calib = load_cameras(f'{a.cache}/calib.json')
    feats = Features(n_kpts=a.kpts)
    masks = body_masks()
    kfm = {c: KeyframeMap(f'{a.maps}/kf_{c}.npz', calib[c]) for c in cams}
    ref = pd.read_csv(a.ref + '.csv')
    ryaw = np.unwrap(ref['yaw'].values)
    pred = lambda t: (np.interp(t, ref.timestamp_ns, ref.x), np.interp(t, ref.timestamp_ns, ref.y),
                      np.interp(t, ref.timestamp_ns, ryaw))

    log, ms = [], []
    for lap in TUNE:
        sets = sets_in(a.bag, cams, int(laps.loc[lap, 't0_ns']), int(laps.loc[lap, 't1_ns']))
        for n, (t, jp) in enumerate(sets):
            torch.cuda.synchronize(); tic = time.perf_counter()
            for c, j in jp.items():
                if n % every[c]:
                    continue
                gray = cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_GRAYSCALE)
                r = kfm[c].localize(feats, feats.extract(gray, masks[c]), pred(t), n_kf=a.n_kf, merge=bool(a.merge))
                log.append(dict(timestamp_ns=t, cam=c, ok=int(r['ok']), n_inl=r['n_inl'], x=r['x'], y=r['y'], yaw=r['yaw']))
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - tic) * 1e3)
    ms = np.array(ms[10:])

    # fixes: variant on the tuning laps, the held-out run's log elsewhere (same cameras)
    base = pd.read_csv(a.ref + '_log.csv')
    inlap = np.zeros(len(base), bool)
    for lap in TUNE:
        inlap |= (base.timestamp_ns >= laps.loc[lap, 't0_ns']) & (base.timestamp_ns <= laps.loc[lap, 't1_ns'])
    var = pd.DataFrame(log)
    fx = pd.concat([base[~inlap & base.cam.isin(cams)], var])
    fx = fx[(fx.ok == 1) & (fx.n_inl >= 15)]
    fixes = {}
    for r in fx.itertuples():
        fixes.setdefault(int(r.timestamp_ns), []).append((r.x, r.y, r.yaw))
    stamps = np.union1d(base.timestamp_ns.unique(), var.timestamp_ns.unique())
    params = PFParams()
    for k, v in json.load(open(a.config))['params'].items():
        setattr(params, k, v)
    poses, _, _, _ = run_filter(a.cache, 'gt.csv', params, fixes, stamps=stamps)

    g = pd.read_csv('gt.csv')
    res = dict(tag=a.tag, cams=a.cams, n_kf=a.n_kf, merge=a.merge, every=a.every or '-', kpts=a.kpts,
               ms_mean=round(float(ms.mean()), 1), ms_p95=round(float(np.percentile(ms, 95)), 1))
    allm = np.zeros(len(poses), bool)
    for lap in TUNE:
        m = (poses.timestamp_ns >= laps.loc[lap, 't0_ns']) & (poses.timestamp_ns <= laps.loc[lap, 't1_ns'])
        allm |= m
        e = np.hypot(poses.x[m] - np.interp(poses.timestamp_ns[m], g.timestamp_ns, g.x),
                     poses.y[m] - np.interp(poses.timestamp_ns[m], g.timestamp_ns, g.y))
        res[f'lap{lap}_rmse'] = round(float(np.sqrt(np.mean(e ** 2))), 3)
    e = np.hypot(poses.x[allm] - np.interp(poses.timestamp_ns[allm], g.timestamp_ns, g.x),
                 poses.y[allm] - np.interp(poses.timestamp_ns[allm], g.timestamp_ns, g.y))
    res['tune_rmse'] = round(float(np.sqrt(np.mean(e ** 2))), 3)
    print(json.dumps(res))
    os.makedirs('out/variants', exist_ok=True)
    with open('out/variants/results.jsonl', 'a') as o:
        o.write(json.dumps(res) + '\n')


if __name__ == '__main__':
    main()
