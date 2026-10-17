"""Particle filter on the track map (slide 17, method 4).

State per particle: x, y, yaw, yaw-rate bias, speed scale.
Predict: body-frame increments from the car state (vx, vy, wz).
Update: drivable area + heading along the borders (map prior), and absolute pose fixes (relocalization).
"""
from dataclasses import dataclass
import numpy as np

from camloc.motion import wrap


@dataclass
class PFParams:
    n: int = 2000
    sigma_out: float = 0.5          # m, softness of the drivable-area constraint
    sigma_head: float = 15.0        # deg, heading vs border direction
    head_floor: float = 0.3         # min heading likelihood (robustness)
    q_yaw: float = 0.3              # deg/sqrt(s) yaw random walk
    q_bias: float = 0.02            # deg/s/sqrt(s) yaw-rate bias random walk
    q_scale: float = 0.002          # 1/sqrt(s) speed-scale random walk
    q_pos: float = 0.05             # m/sqrt(s) position random walk
    # relocalization fix
    fix_lat: float = 0.4            # m
    fix_lon: float = 1.5            # m
    fix_yaw: float = 0.5            # deg
    fix_floor: float = 0.05         # outlier floor of the fix likelihood
    fix_gate_pos: float = 10.0      # m, ignore fixes farther than this from the estimate
    fix_gate_yaw: float = 5.0       # deg


class ParticleFilter:
    def __init__(self, trackmap, pose0, p=PFParams(), seed=0):
        self.tm, self.p = trackmap, p
        self.rng = np.random.default_rng(seed)
        N = p.n
        x0, y0, yaw0 = pose0
        self.x = x0 + self.rng.normal(0, 0.1, N)
        self.y = y0 + self.rng.normal(0, 0.1, N)
        self.yaw = yaw0 + self.rng.normal(0, np.radians(0.5), N)
        self.bias = self.rng.normal(0, np.radians(0.1), N)
        self.scale = 1 + self.rng.normal(0, 0.005, N)
        self.w = np.full(N, 1.0 / N)

    def predict(self, dx, dy, dyaw, dt):
        p, N, r = self.p, self.p.n, self.rng
        sq = np.sqrt(dt)
        ym = self.yaw + 0.5 * (dyaw - self.bias * dt)
        self.x += self.scale * (dx * np.cos(ym) - dy * np.sin(ym)) + r.normal(0, p.q_pos * sq, N)
        self.y += self.scale * (dx * np.sin(ym) + dy * np.cos(ym)) + r.normal(0, p.q_pos * sq, N)
        self.yaw += dyaw - self.bias * dt + r.normal(0, np.radians(p.q_yaw) * sq, N)
        self.bias += r.normal(0, np.radians(p.q_bias) * sq, N)
        self.scale += r.normal(0, p.q_scale * sq, N)

    def update_map(self):
        p = self.p
        d = self.tm.dist_outside(self.x, self.y)
        lik = np.exp(-0.5 * (d / p.sigma_out) ** 2)
        dh = wrap(2 * (self.yaw - self.tm.border_dir(self.x, self.y))) / 2     # axial difference
        lik *= p.head_floor + (1 - p.head_floor) * np.exp(-0.5 * (dh / np.radians(p.sigma_head)) ** 2)
        self._reweight(lik)

    def update_fix(self, fx, fy, fyaw, sig_lat=None, sig_lon=None, gate_pos=None):
        """Absolute pose fix. Returns False if it was gated out. sig_lat / sig_lon / gate_pos override the
        parameters for a fix source with a different accuracy (e.g. the RoMa fallback)."""
        p = self.p
        mx, my, myaw = self.estimate()
        gate = p.fix_gate_pos if gate_pos is None else gate_pos
        if np.hypot(fx - mx, fy - my) > gate or abs(wrap(fyaw - myaw)) > np.radians(p.fix_gate_yaw):
            return False
        ex, ey = fx - self.x, fy - self.y
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        lon, lat = c * ex + s * ey, -s * ex + c * ey
        sl, sn = (p.fix_lat if sig_lat is None else sig_lat), (p.fix_lon if sig_lon is None else sig_lon)
        e = (lat / sl) ** 2 + (lon / sn) ** 2 + (wrap(fyaw - self.yaw) / np.radians(p.fix_yaw)) ** 2
        self._reweight(p.fix_floor + (1 - p.fix_floor) * np.exp(-0.5 * e))
        return True

    def _reweight(self, lik):
        w = self.w * (lik + 1e-300)
        self.w = w / w.sum()
        if 1.0 / np.sum(self.w ** 2) < self.p.n / 2:        # systematic resampling
            N = self.p.n
            idx = np.minimum(np.searchsorted(np.cumsum(self.w), (self.rng.random() + np.arange(N)) / N), N - 1)
            self.x, self.y, self.yaw = self.x[idx], self.y[idx], self.yaw[idx]
            self.bias, self.scale = self.bias[idx], self.scale[idx]
            self.w = np.full(N, 1.0 / N)

    def estimate(self):
        w = self.w
        return (np.sum(w * self.x), np.sum(w * self.y),
                np.arctan2(np.sum(w * np.sin(self.yaw)), np.sum(w * np.cos(self.yaw))))
