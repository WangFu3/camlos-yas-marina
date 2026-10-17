#!/usr/bin/env python
"""Matcher comparison under appearance changes: LightGlue vs RoMa, on tuning laps 2 and 6 (development).

    python scripts/exp_roma.py BAG cache --conds none,lights_off,invert --n 30

Map: maps_no26/kf_fl.npz (built without laps 2 and 6), keyframe images unchanged (night).
Query: fl images of laps 2 and 6 with a synthetic transform (camloc/augment.py).
Keyframe selection uses the GT position: this compares the matchers, it is not an online result.
RoMa: dense warp keyframe -> query, sampled at the keyframe's triangulated keypoints, then PnP.
"""
import argparse, os, sys, time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from camloc.camera import load_cameras
from camloc.body_masks import body_masks
from camloc.features import Features
from camloc.kfmap import KeyframeMap
from camloc.augment import TRANSFORMS

TOPIC = '/constructor0/sensor/camera_{}/compressed_image'


def read_at(bag, cam, t):
    with open(bag, 'rb') as f:
        for _, _, _, msg in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                topics=[TOPIC.format(cam)], start_time=t - 200_000_000, end_time=t + 200_000_000):
            if msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec == t:
                return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_GRAYSCALE)
    raise KeyError(t)


def pnp(cam, img, obj, min_inl=15):
    if len(img) < min_inl:
        return None
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj.astype(np.float64), img.astype(np.float64), cam.K, cam.D,
                                             reprojectionError=3.0, iterationsCount=300, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < min_inl:
        return None
    inl = inl[:, 0]
    ok, rvec, tvec = cv2.solvePnP(obj[inl].astype(np.float64), img[inl].astype(np.float64), cam.K, cam.D, rvec, tvec, True)
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rvec)[0]; T[:3, 3] = tvec.ravel()
    T = np.linalg.inv(T) @ cam.T_cam_base
    return T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0]), len(inl)


@torch.no_grad()
def roma_localize(roma, cam, mask, kfm, k, kf_img, q_img, cert_thr):
    """Dense RoMa warp between query (A) and keyframe (B); read the query position of B's 3D keypoints."""
    pa = Image.fromarray(q_img).convert('RGB')
    pb = Image.fromarray(kf_img).convert('RGB')
    warp, cert = roma.match(pa, pb, device='cuda')
    if warp.dim() == 4:                                                # batched output: (1, H, 2W, 4)
        warp, cert = warp[0], cert[0]
    W2 = warp.shape[1] // 2
    s_warp, s_cert = warp[:, W2:], cert[:, W2:]                       # grid over B: (xA, yA, xB, yB)
    s = slice(kfm.off[k], kfm.off[k + 1])
    kb = torch.from_numpy(kfm.kpts[s]).cuda().float()
    h, w = kf_img.shape
    g = torch.stack((2 * kb[:, 0] / w - 1, 2 * kb[:, 1] / h - 1), -1)[None, None]   # (1, 1, N, 2)
    a = F.grid_sample(s_warp.permute(2, 0, 1)[None, :2], g, align_corners=False)[0, :, 0].T   # (N, 2) A norm
    c = F.grid_sample(s_cert[None, None], g, align_corners=False)[0, 0, 0]
    qa = torch.stack(((a[:, 0] + 1) * w / 2, (a[:, 1] + 1) * h / 2), -1).cpu().numpy()
    c = c.cpu().numpy()
    qi = np.clip(qa.round().astype(int), 0, [w - 1, h - 1])
    keep = (c > cert_thr) & ~mask[qi[:, 1], qi[:, 0]]
    return pnp(cam, qa[keep], kfm.X[s][keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag'); ap.add_argument('cache')
    ap.add_argument('--conds', default='none,lights_off,invert')
    ap.add_argument('--n', type=int, default=30, help='query frames per lap')
    ap.add_argument('--cert', type=float, default=0.2)
    a = ap.parse_args()
    cam = load_cameras(f'{a.cache}/calib.json')['fl']
    mask = body_masks()['fl']
    kfm = KeyframeMap('maps_no26/kf_fl.npz', cam)
    feats = Features()
    from romatch import roma_outdoor
    roma = roma_outdoor(device='cuda', coarse_res=420, upsample_res=(420, 560), use_custom_corr=False)

    g = pd.read_csv('gt.csv'); gyaw = np.unwrap(g['yaw'].values)
    fr = pd.read_csv(f'{a.cache}/frames.csv'); fl_t = np.sort(fr[fr.cam == 'fl'].timestamp_ns.values)
    laps = pd.read_csv(f'{a.cache}/laps.csv').set_index('lap')
    # keyframe stamps: the keyframe pose is the GT pose at its frame -> nearest GT sample, then nearest fl frame
    tree = cKDTree(np.c_[g.x, g.y])
    _, gi = tree.query(kfm.kf_pose[:, :2])
    kf_t = fl_t[np.clip(np.searchsorted(fl_t, g.timestamp_ns.values[gi]), 0, len(fl_t) - 1)]

    queries = []
    for lap in (2, 6):
        ts = fl_t[(fl_t > laps.loc[lap, 't0_ns']) & (fl_t < laps.loc[lap, 't1_ns'])]
        queries += list(ts[np.linspace(0, len(ts) - 1, a.n).astype(int)])
    rows = []
    for t in queries:
        x, y, yaw = (np.interp(t, g.timestamp_ns, v) for v in (g.x, g.y, gyaw))
        k = int(kfm.nearest((x, y, yaw), 1)[0])
        q0 = read_at(a.bag, 'fl', int(t)); kimg = read_at(a.bag, 'fl', int(kf_t[k]))
        for cond in a.conds.split(','):
            q = TRANSFORMS[cond](q0)
            for name in ('lightglue', 'roma'):
                torch.cuda.synchronize(); tic = time.perf_counter()
                if name == 'lightglue':
                    r = kfm.localize(feats, feats.extract(q, mask), (x, y, yaw), n_kf=2)
                    res = (r['x'], r['y'], r['yaw'], r['n_inl']) if r['ok'] and r['n_inl'] >= 15 else None
                else:
                    res = roma_localize(roma, cam, mask, kfm, k, kimg, q, a.cert)
                torch.cuda.synchronize(); ms = (time.perf_counter() - tic) * 1e3
                row = dict(t=t, cond=cond, matcher=name, ms=ms, ok=res is not None)
                if res is not None:
                    ex, ey = res[0] - x, res[1] - y
                    row.update(lat=abs(-np.sin(yaw) * ex + np.cos(yaw) * ey), lon=abs(np.cos(yaw) * ex + np.sin(yaw) * ey),
                               dyaw=abs(np.degrees(np.angle(np.exp(1j * (res[2] - yaw))))), inl=res[3])
                rows.append(row)
    r = pd.DataFrame(rows)
    os.makedirs('out/roma', exist_ok=True)
    r.to_csv('out/roma/exp_roma.csv', index=False)
    agg = r.groupby(['cond', 'matcher']).agg(
        solved=('ok', lambda v: round(v.mean() * 100)), lat_med=('lat', 'median'), lat_p90=('lat', lambda v: v.quantile(.9)),
        lon_med=('lon', 'median'), lon_p90=('lon', lambda v: v.quantile(.9)), within1m=('lat', lambda v: None),
        ms=('ms', 'median'))
    good = r[r.ok]
    agg['within1m'] = good.assign(w=np.hypot(good.lat, good.lon) < 1).groupby(['cond', 'matcher']).w.mean().mul(100).round()
    print(f'{len(queries)} query frames (laps 2, 6), keyframes from maps_no26, GT-selected keyframe')
    print(agg.round(2).to_string())


if __name__ == '__main__':
    main()
