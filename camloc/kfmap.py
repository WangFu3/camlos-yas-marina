"""Keyframe landmark map for relocalization with ALIKED + LightGlue (slide 16, method 3a).

Map file (npz): per keyframe its camera pose (x, y, yaw of base_link) and the keypoints that were
triangulated (pixel, descriptor, score, 3D point in map). Stored flat with an offset array.
"""
import cv2
import numpy as np

from camloc.features import from_numpy


class KeyframeMap:
    def __init__(self, path, cam):
        d = np.load(path)
        self.kf_pose = d['kf_pose']                    # (K, 3) x, y, yaw
        self.off = d['off']                            # (K + 1,) offsets into the flat arrays
        self.kpts, self.desc, self.score, self.X = d['kpts'], d['desc'], d['score'], d['X']
        self.image_size = tuple(d['image_size'])
        self.cam = cam
        self._gpu = {}

    def keyframe(self, k):
        if k not in self._gpu:
            if len(self._gpu) > 512:
                self._gpu.clear()
            s = slice(self.off[k], self.off[k + 1])
            self._gpu[k] = from_numpy(self.kpts[s], self.desc[s], self.score[s], self.image_size)
        return self._gpu[k]

    def nearest(self, pred, n=2, radius=15.0, max_dyaw=np.radians(45)):
        d = np.hypot(self.kf_pose[:, 0] - pred[0], self.kf_pose[:, 1] - pred[1])
        d[np.abs(np.angle(np.exp(1j * (self.kf_pose[:, 2] - pred[2])))) > max_dyaw] = np.inf
        d[d > radius] = np.inf
        idx = np.argsort(d)[:n]
        return idx[np.isfinite(d[idx])]

    def merged(self, kfs):
        """One feature dict holding the keypoints of several keyframes (one LightGlue call instead of n)."""
        s = np.concatenate([np.arange(self.off[k], self.off[k + 1]) for k in kfs])
        return from_numpy(self.kpts[s], self.desc[s], self.score[s], self.image_size), s

    def localize(self, feats, fq, pred, n_kf=2, merge=False, reproj=3.0, min_inliers=15):
        """Pose of base_link from query features fq (GPU dict) near the predicted pose.

        merge: match against the n_kf keyframes concatenated in a single LightGlue call.
        Returns dict(ok, x, y, yaw, n_match, n_inl, kf)."""
        out = dict(ok=False, x=np.nan, y=np.nan, yaw=np.nan, n_match=0, n_inl=0, kf=-1)
        kfs = self.nearest(pred, n_kf)
        if len(kfs) == 0:
            return out
        qk = fq['keypoints'][0].cpu().numpy()
        img, obj = [], []
        if merge and len(kfs) > 1:
            fk, rows = self.merged(kfs)
            iq, ik = feats.match(fq, fk)
            img.append(qk[iq]); obj.append(self.X[rows[ik]])
        else:
            for k in kfs:
                iq, ik = feats.match(fq, self.keyframe(k))
                img.append(qk[iq]); obj.append(self.X[self.off[k] + ik])
        img, obj = np.vstack(img).astype(np.float64), np.vstack(obj).astype(np.float64)
        out.update(n_match=len(img), kf=int(kfs[0]))
        if len(img) < min_inliers:
            return out
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, self.cam.K, self.cam.D, reprojectionError=reproj,
                                                 iterationsCount=300, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < min_inliers:
            out['n_inl'] = 0 if inl is None else len(inl)
            return out
        inl = inl[:, 0]
        ok, rvec, tvec = cv2.solvePnP(obj[inl], img[inl], self.cam.K, self.cam.D, rvec, tvec, True)
        T_cam_map = np.eye(4)
        T_cam_map[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cam_map[:3, 3] = tvec.ravel()
        T = np.linalg.inv(T_cam_map) @ self.cam.T_cam_base
        out.update(ok=True, x=T[0, 3], y=T[1, 3], yaw=np.arctan2(T[1, 0], T[0, 0]), n_inl=len(inl))
        return out
