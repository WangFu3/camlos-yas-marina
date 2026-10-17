"""Learned local features on the GPU: ALIKED keypoints + LightGlue matching (pretrained, cvg/LightGlue)."""
import cv2
import numpy as np
import torch
from lightglue import ALIKED, LightGlue

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Features:
    """extract(gray, mask) -> feature dict on the GPU; match(a, b) -> index arrays (ia, ib)."""

    def __init__(self, n_kpts=2048, resize=1024, clahe=True, mp=False, amp=False):
        """mp: LightGlue mixed precision; amp: ALIKED under fp16 autocast."""
        self.net = ALIKED(max_num_keypoints=n_kpts, detection_threshold=0.0).eval().to(DEV)
        self.lg = LightGlue(features='aliked', mp=mp).eval().to(DEV)
        self.resize, self.amp = resize, amp and DEV.type == 'cuda'
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None

    @torch.no_grad()
    def extract(self, gray, mask=None):
        g = self.clahe.apply(gray) if self.clahe is not None else gray
        img = torch.from_numpy(g).to(DEV).float().div_(255).expand(3, -1, -1)
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
            f = self.net.extract(img, resize=self.resize)
        f = {n: (v.float() if v.is_floating_point() else v) for n, v in f.items()}
        if mask is not None:
            k = f['keypoints'][0].round().long()
            k[:, 0].clamp_(0, gray.shape[1] - 1); k[:, 1].clamp_(0, gray.shape[0] - 1)
            keep = ~torch.from_numpy(mask).to(DEV)[k[:, 1], k[:, 0]]
            f = {n: (v[:, keep] if n != 'image_size' else v) for n, v in f.items()}
        return f

    @torch.no_grad()
    def match(self, a, b):
        m = self.lg({'image0': a, 'image1': b})['matches'][0]
        return m[:, 0].cpu().numpy(), m[:, 1].cpu().numpy()


def subset(f, idx):
    """Feature dict restricted to keypoint indices idx (numpy)."""
    t = torch.from_numpy(np.asarray(idx)).to(f['keypoints'].device)
    return {n: (v[:, t] if n != 'image_size' else v) for n, v in f.items()}


def to_numpy(f):
    return dict(kpts=f['keypoints'][0].cpu().numpy().astype(np.float32),
                desc=f['descriptors'][0].cpu().numpy().astype(np.float16),
                score=f['keypoint_scores'][0].cpu().numpy().astype(np.float16))


def from_numpy(kpts, desc, score, image_size):
    return {'keypoints': torch.from_numpy(kpts).to(DEV)[None],
            'descriptors': torch.from_numpy(desc.astype(np.float32)).to(DEV)[None],
            'keypoint_scores': torch.from_numpy(score.astype(np.float32)).to(DEV)[None],
            'image_size': torch.tensor([image_size], device=DEV, dtype=torch.float32)}
