#!/usr/bin/env python
"""Per-stage timing of the online pipeline on image sets of a tuning lap (development only).

    python scripts/profile_frame.py BAG cache maps_no26 --cams fl,rl --lap 2 --n 300

Stages per camera: JPEG decode, CLAHE, ALIKED, LightGlue (x n_kf), PnP; plus the filter update.
GT is used only to supply the predicted pose for keyframe selection.
"""
import argparse, os, sys, time
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from camloc.camera import load_cameras
from camloc.body_masks import body_masks
from camloc.features import Features
from camloc.kfmap import KeyframeMap
from exp_landmarks import read_images


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag'); ap.add_argument('cache'); ap.add_argument('maps')
    ap.add_argument('--cams', default='fl,rl')
    ap.add_argument('--lap', type=int, default=2)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--n-kf', type=int, default=2)
    a = ap.parse_args()
    cams = a.cams.split(',')
    laps = pd.read_csv(f'{a.cache}/laps.csv').set_index('lap')
    assert laps.loc[a.lap, 'split'] != 'val', 'profile on a training lap'
    t0 = int(laps.loc[a.lap, 't0_ns'] + 20e9); t1 = int(t0 + a.n * 1e8)
    imgs = {c: read_images(a.bag, c, t0, t1) for c in cams}   # decoded already; re-encode to time decode
    jpg = {c: [(t, cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]) for t, im in imgs[c]] for c in cams}
    g = pd.read_csv('gt.csv')
    gyaw = np.unwrap(g['yaw'].values)
    calib = load_cameras(f'{a.cache}/calib.json')
    feats = Features()
    masks = body_masks()
    kfm = {c: KeyframeMap(f'{a.maps}/kf_{c}.npz', calib[c]) for c in cams}

    T = defaultdict(list)
    n = min(len(jpg[c]) for c in cams)
    for i in range(n):
        tot = time.perf_counter()
        for c in cams:
            t, buf = jpg[c][i]
            pred = (np.interp(t, g.timestamp_ns, g.x), np.interp(t, g.timestamp_ns, g.y), np.interp(t, g.timestamp_ns, gyaw))
            s = time.perf_counter(); gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE); T[f'{c} decode'].append(time.perf_counter() - s)
            s = time.perf_counter(); feats.clahe.apply(gray); T[f'{c} clahe'].append(time.perf_counter() - s)
            sync(); s = time.perf_counter(); fq = feats.extract(gray, masks[c]); sync(); T[f'{c} aliked (+clahe)'].append(time.perf_counter() - s)
            # localize = LightGlue x n_kf + PnP; split them
            kfs = kfm[c].nearest(pred, a.n_kf)
            sync(); s = time.perf_counter()
            for k in kfs:
                feats.match(fq, kfm[c].keyframe(k))
            sync(); T[f'{c} lightglue x{len(kfs)}'].append(time.perf_counter() - s)
            s = time.perf_counter(); kfm[c].localize(feats, fq, pred, n_kf=a.n_kf); sync()
            T[f'{c} localize total'].append(time.perf_counter() - s)
        T['set total (incl. duplicate localize)'].append(time.perf_counter() - tot)

    skip = 10                                   # warm-up (CUDA kernels, keyframe cache)
    rows = [(k, np.mean(v[skip:]) * 1e3, np.median(v[skip:]) * 1e3) for k, v in T.items()]
    print(f'{n} image sets of lap {a.lap}, cams {cams}, GPU {torch.cuda.get_device_name(0)}')
    print(pd.DataFrame(rows, columns=['stage', 'mean_ms', 'median_ms']).round(1).to_string(index=False))


if __name__ == '__main__':
    main()
