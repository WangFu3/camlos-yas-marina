"""RoMa fallback relocalization (pretrained, github.com/Parskatt/RoMa).

Used only when LightGlue gives no accepted fix, at a limited rate (RoMa takes ~1 s per pair on an
RTX 2070). RoMa matches whole images densely and is far more robust to appearance changes (e.g. inverted
shading) than sparse keypoint matching. The dense warp keyframe -> query is read at the keyframe's
triangulated keypoints, and PnP gives the pose.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class RomaFallback:
    def __init__(self, kfmap, kf_img_dir, mask, coarse_res=420, cert_thr=0.2, min_inliers=15):
        from romatch import roma_outdoor
        self.roma = roma_outdoor(device='cuda', coarse_res=coarse_res, upsample_res=(coarse_res, coarse_res * 4 // 3),
                                 use_custom_corr=False)
        self.kfm, self.dir, self.mask = kfmap, kf_img_dir, mask
        self.cert_thr, self.min_inliers = cert_thr, min_inliers

    @torch.no_grad()
    def localize(self, gray, pred):
        """Returns dict(ok, x, y, yaw, n_inl, kf)."""
        out = dict(ok=False, x=np.nan, y=np.nan, yaw=np.nan, n_inl=0, kf=-1)
        kfs = self.kfm.nearest(pred, 1)
        if len(kfs) == 0:
            return out
        k = int(kfs[0])
        kimg = cv2.imread(os.path.join(self.dir, f'{k}.jpg'), cv2.IMREAD_GRAYSCALE)
        if kimg is None:
            return out
        warp, cert = self.roma.match(Image.fromarray(gray).convert('RGB'), Image.fromarray(kimg).convert('RGB'),
                                     device='cuda')
        if warp.dim() == 4:
            warp, cert = warp[0], cert[0]
        w2 = warp.shape[1] // 2
        s_warp, s_cert = warp[:, w2:], cert[:, w2:]               # grid over the keyframe: (xQ, yQ, xK, yK)
        s = slice(self.kfm.off[k], self.kfm.off[k + 1])
        h, w = gray.shape                                          # keyframe keypoints are in full-res pixels
        kp = torch.from_numpy(self.kfm.kpts[s]).cuda().float()
        grid = torch.stack((2 * kp[:, 0] / w - 1, 2 * kp[:, 1] / h - 1), -1)[None, None]
        q = F.grid_sample(s_warp.permute(2, 0, 1)[None, :2], grid, align_corners=False)[0, :, 0].T
        c = F.grid_sample(s_cert[None, None], grid, align_corners=False)[0, 0, 0].cpu().numpy()
        qp = torch.stack(((q[:, 0] + 1) * w / 2, (q[:, 1] + 1) * h / 2), -1).cpu().numpy()
        qi = np.clip(qp.round().astype(int), 0, [w - 1, h - 1])
        keep = (c > self.cert_thr) & ~self.mask[qi[:, 1], qi[:, 0]]
        img, obj = qp[keep].astype(np.float64), self.kfm.X[s][keep].astype(np.float64)
        out['kf'] = k
        if len(img) < self.min_inliers:
            return out
        cam = self.kfm.cam
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, cam.K, cam.D, reprojectionError=3.0,
                                                 iterationsCount=300, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < self.min_inliers:
            return out
        inl = inl[:, 0]
        ok, rvec, tvec = cv2.solvePnP(obj[inl], img[inl], cam.K, cam.D, rvec, tvec, True)
        T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rvec)[0]; T[:3, 3] = tvec.ravel()
        T = np.linalg.inv(T) @ cam.T_cam_base
        out.update(ok=True, x=T[0, 3], y=T[1, 3], yaw=np.arctan2(T[1, 0], T[0, 0]), n_inl=len(inl))
        return out
