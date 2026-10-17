# Camera-based localization on a race track (Yas Marina)

Our solution for the Constructor Labs challenge at the National AI Hackathon 2026: estimate the race
car's pose (x, y, yaw in the `map` frame) at 10 Hz from its cameras, causally, starting from the first
pose of the run. The task description and data format are in [DATASET.md](DATASET.md).

## Results

Validation laps 3 and 7 of the training run. They were never used to build maps or to choose anything,
and were scored with the organizers' `tools/evaluate.py`.

| Version | Cameras | Lap 3 RMSE | Lap 7 RMSE (40 m/s) | Lateral RMSE | Time / frame |
|---|---|---|---|---|---|
| Car-state dead reckoning only | — | 43.6 m | 102.0 m | 33.2 / 79.2 m | <1 ms |
| Car state + track map (particle filter) | — | 4.02 m | 4.24 m | 3.09 / 2.61 m | 2 ms |
| + SIFT landmark relocalization (precomputed) | fl | 0.41 m | 0.43 m | 0.18 / 0.21 m | ~150 ms |
| + ALIKED + LightGlue, online | fl | 0.46 m | 0.49 m | 0.18 / 0.21 m | 102 ms |
| **Current: + rear-left camera, parameters chosen on held-out laps 2 and 6** | **fl + rl** | **0.21 m** | **0.27 m** | **0.13 / 0.11 m** | **201 ms** |

Timing is per image set on an NVIDIA GeForce RTX 2070 under WSL2. Coverage is 100 % in all versions.
Full analysis, failure cases and next steps: [RESULTS17.10morning.md](RESULTS17.10morning.md);
method and test procedure: [METHOD.md](METHOD.md).

## Method

1. **Motion model:** car-state speed and yaw rate (`vx`, `vy`, `wz` from `/constructor0/state_estimation`,
   allowed by the task) integrated between image stamps.
2. **Track-map prior:** the car must be inside the track or pit-lane borders, heading along them.
3. **Landmark relocalization (slide 16, method 3a):** keyframe maps built from the training laps by
   triangulating ALIKED keypoints with the GT poses. Online, each image is matched with LightGlue to the
   keyframes nearest the predicted pose, and PnP gives an absolute pose fix.
4. **Particle filter (slide 17, method 4):** fuses 1–3; outlier fixes are gated. Output at every image set.

Ground truth is used only to build the maps and to score; at inference only the first pose of the run is used.

**Pretrained models used:** ALIKED (`aliked-n16`) and LightGlue (`aliked_lightglue`) from
[cvg/LightGlue](https://github.com/cvg/LightGlue). No external datasets.

## Data split

| Use | Laps (see `cache/laps.csv`) |
|---|---|
| Build maps | training laps 0 (out-lap), 1, 2, 4, 5, 6, 8 (in-lap) |
| Choose cameras and parameters | laps 2 and 6, using maps built **without** them (`maps_no26/`) |
| Validation (final report only) | laps 3 and 7 |

## Setup

```bash
conda create -n camloc python=3.12 -y && conda activate camloc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git
```

Put the dataset (`c0_2026-10-13_22-53-42_camloc_0.mcap`) in the repository root. It is not in git
(licence: hackathon use only).

## Reproduce

```bash
BAG=c0_2026-10-13_22-53-42_camloc_0.mcap
# 1. training data (once): calibration, borders, state, sample images, GT, lap split
python tools/export_cache.py $BAG cache --img-every 10
python tools/extract_gt.py $BAG gt.csv
python scripts/split_laps.py gt.csv cache
# 2. keyframe maps from the training laps (~11 min per camera)
python scripts/build_kf_map.py $BAG cache gt.csv maps/kf_fl.npz --cam fl
python scripts/build_kf_map.py $BAG cache gt.csv maps/kf_rl.npz --cam rl
# 3. localization: one command, raw mcap -> submission CSV (causal, online)
python run.py $BAG out/final/final.csv --cams fl,rl --pf-config configs/fl_rl.json
# 4. score (full run and each validation lap)
python scripts/eval_laps.py gt.csv out/final/final.csv cache
```

`scripts/overnight.sh` runs the whole held-out selection (maps without laps 2 and 6, online run with
three cameras, parameter and camera sweep on laps 2 and 6) and then the final run.

## Where the results are

Results, maps and caches are **not in git** (size, dataset licence). After running the commands above
they are on the local disk:

| Path | Contents |
|---|---|
| `out/final/final.csv` | current submission-format poses (`timestamp_ns,x,y,yaw`) |
| `out/final/final_eval.txt` | `evaluate.py` report: full run and validation laps 3, 7 |
| `out/final/final_overview.png` | trajectory vs GT and lateral / longitudinal error over time |
| `out/final/final_timing.json` | mean / median / p95 time per frame, GPU |
| `out/final/final_log.csv` | every relocalization fix per camera and frame |
| `out/lolo/run3_log_sweep.csv` | camera-set × parameter sweep on held-out laps 2 and 6 |
| `out/lolo/run3_log_best.json` | the selected configuration (copied to `configs/fl_rl.json`) |
| `out/map_pf*`, `out/reloc_pf*`, `out/aliked_online*` | earlier versions (rows 2–4 of the results table) |
| `out/variants/results.jsonl` | speed-up variants scored on laps 2 and 6 (in progress) |
| `maps/kf_*.npz` | keyframe maps from all training laps (0.3–0.5 GB each) |
| `maps_no26/kf_*.npz` | keyframe maps without laps 2 and 6, for tuning |
| `cache/` | data exported from the bag, lap split, sample images |

## Repository layout

| Path | Contents |
|---|---|
| `run.py` | online pipeline: bag → submission CSV, with per-frame timing |
| `camloc/` | library: `pf.py` particle filter, `trackmap.py` track prior, `motion.py` odometry, `features.py` ALIKED + LightGlue, `kfmap.py` keyframe map + PnP, `camera.py` calibration, `body_masks.py` car-body masks |
| `scripts/` | map building, lap split, evaluation, parameter sweeps, profiling, experiments |
| `tools/` | organizers' `evaluate.py` and `extract_gt.py`, plus our export and plotting tools |
| `configs/fl_rl.json` | cameras and filter parameters selected on laps 2 and 6 |

## Open items

- Runtime is 201 ms/frame on the RTX 2070; full runtime points need ≤ 100 ms (work in progress on
  branch `feature/speedup`).
- The maps are night-only; a daylight test run is the main risk.
