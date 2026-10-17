# Progress summary: camera localization (Yas Marina, as of the morning of 2026-10-17)

## 1. Summary

The final pipeline reaches a **position RMSE of 0.21 m (lap 3) and 0.27 m (lap 7, 40 m/s)** on the
validation laps, with ~0.11–0.13 m lateral error and 100 % coverage. It uses two cameras, front `fl` and
rear-left `rl`: landmark relocalization with ALIKED + LightGlue, fused with car-state speed in a
particle filter. The camera set and filter parameters were **chosen only on held-out training laps
(laps 2 and 6)**. The main open item is speed: 201 ms per frame now, while full runtime points need
≤ 100 ms.

---

## 2. Data split

| Purpose | Laps | Notes |
|---|---|---|
| Map building | laps 0 (out-lap), 1, 4, 5, 8 (in-lap); the final maps also include laps 2 and 6 | Landmarks triangulated with GT poses (allowed by slide 16) |
| **Tuning / selection** | **lap 2** (slow, 29 m/s), **lap 6** (fast, 40 m/s) | The tuning maps **exclude** these two laps, so accuracy measured on them approximates unseen data |
| **Validation (final report only)** | **lap 3** (35 m/s), **lap 7** (40 m/s, fastest) | Never used to build maps or to choose anything |

The lap split is in `cache/laps.csv`, produced by `scripts/split_laps.py`.

---

## 3. Results by version (validation laps 3 / 7, organizers' `tools/evaluate.py`)

| # | Version | Cameras | Position RMSE | Lateral RMSE | Longitudinal RMSE | Yaw RMSE | Within 1 m | Time / frame | Online |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Car-state dead reckoning only | — | 43.6 / 102.0 m | 33.2 / 79.2 m | 28.3 / 64.3 m | 5.9 / 14.0° | 0 / 0 % | <1 ms | yes |
| 1 | Car state + track map (particle filter) | — | 4.02 / 4.24 m | 3.09 / 2.61 m | 2.57 / 3.34 m | 0.85 / 0.88° | 5.7 / 4.0 % | 2 ms | yes |
| 2 | + SIFT landmark relocalization | fl | 0.41 / 0.43 m | 0.18 / 0.21 m | 0.37 / 0.38 m | 0.15 / 0.17° | 98.6 / 96.4 % | ~150 ms | **no** (precomputed) |
| 3 | + ALIKED + LightGlue, default parameters | fl | 0.46 / 0.49 m | 0.18 / 0.21 m | 0.43 / 0.44 m | 0.08 / 0.12° | 97.4 / 95.3 % | 102 ms | yes |
| **4** | **+ second camera, parameters chosen on laps 2, 6** | **fl + rl** | **0.21 / 0.27 m** | **0.13 / 0.11 m** | **0.17 / 0.25 m** | **0.16 / 0.14°** | **100 / 97.8 %** | **201 ms** | **yes** |

Notes:
- Version 2 is a reference only: the predicted pose that selects landmarks came from version 1, so it was
  not fully online.
- The gain from version 3 to 4 has two sources: adding the `rl` camera, and tightening the filter
  parameters to match ALIKED's actual accuracy.
- One parameter set had been chosen by looking at the validation laps (replay result 0.23 / 0.32 m). **It
  was discarded** and is not part of the final pipeline.
- Plots: `out/map_pf_overview.png` (version 1), `out/reloc_pf_overview.png` (version 2),
  `out/aliked_online_overview.png` (version 3), `out/final/final_overview.png` (version 4).

---

## 4. Selection on the held-out laps (2 and 6) only

The tuning maps exclude laps 2 and 6 (`maps_no26/`). One full online run with all three cameras
recorded every fix; each camera set × 36 filter-parameter combinations was then compared by replaying
those fixes. The selection rule was fixed in advance: **lowest position RMSE on laps 2 and 6**.

| Camera set | Lap 2 | Lap 6 | Combined | Combined, default parameters |
|---|---|---|---|---|
| **fl + rl (selected)** | **0.191 m** | **0.175 m** | **0.185 m** | 0.303 m |
| fl + cl + rl | 0.266 m | 0.184 m | 0.236 m | 0.323 m |
| fl | 0.302 m | 0.300 m | 0.301 m | 0.400 m |
| fl + cl | 0.373 m | 0.239 m | 0.325 m | 0.403 m |

Selected parameters: camera-fix σ lateral 0.1 m, longitudinal 0.3 m, outlier gate 3 m, outlier floor 0.2
(`out/final/selected_config.json`).

Held-out laps: 0.185 m; validation laps: 0.21–0.27 m. Validation is slightly worse, which is the normal
generalization gap.

**Single-frame fix accuracy on the held-out laps:**

| Camera | Fixes solved | Lateral median | Lateral p90 | Longitudinal median | Longitudinal p90 |
|---|---|---|---|---|---|
| fl | 94 % | 0.07 m | 2.67 m | 0.19 m | 1.58 m |
| cl | 95 % | 0.14 m | 0.82 m | 0.13 m | 0.49 m |
| rl | 89 % | 0.11 m | 0.93 m | 0.13 m | 0.67 m |

`fl` is the most accurate laterally but has occasional large errors; `rl` is steadier longitudinally.
The two complement each other.

---

## 5. Failure cases and lessons (usable in the presentation)

1. **Ground border matching (slide 15, method 2) failed.** The cameras are only 0.34 m above the ground,
   so 1 m of lateral shift moves the border line only ~6 px in the image, about the same as the roll
   calibration error. The code is kept in `camloc/border_match.py`.
2. **A "racing line prior" doesn't help.** At the same spot, the lateral position differs by 1.4–3.2 m
   (RMS) between laps; the car does not drive the same line every lap.
3. **SIFT is nearly useless on the side and rear cameras** (16–24 % of frames solved). ALIKED + LightGlue
   raises this to 77–95 %.
4. **Adding `cl` made things worse.** Its single-frame accuracy looks fine, yet combining it increases the
   error. Possibly because `cl`'s extrinsics only have the older calibration (the README says `cl`/`cr`
   keep the previous calibration); not verified yet.
5. **Multi-camera timestamp bug.** The cameras' stamps differ by tens of microseconds; they are not
   exactly equal. Pairing by exact equality skipped almost every image of the second and third cameras.
   Images within ±5 ms now form one set, and the set's latest stamp is the output stamp (causal).
6. **CPU oversubscription.** 4 processes × 4 OpenCV threads each on a 4-core CPU fought for cores
   (load 16). Now 1 thread per process.
7. **Train / validation discipline.** The training-lap images built the map, so tuning on training laps
   always favours "tighter is better" and picks parameters that are worse on new data. Parameters are
   therefore chosen with held-out laps instead.

---

## 6. Limits and risks

1. **Lighting.** The maps come entirely from night images, and the hidden test run "may include
   different lighting". In daylight, many landmarks may fail to match; the filter then falls back to car
   state + track map, ~4 m. **This is the biggest risk.**
2. **Speed.** The hidden run may be faster; lap 7 (40 m/s) is already a bit worse than lap 3.
3. **Runtime.** 201 ms/frame (RTX 2070), above the 100 ms full-points threshold; runtime is 5 % of the score.
4. **Validation laps have been looked at several times.** During development, laps 3 and 7 were scored
   once each for versions 2 and 3, plus one discarded parameter replay. Version 4's choice did not use
   them, but the validation laps are no longer completely unseen. From now on, changes are evaluated only
   on laps 2 and 6.
5. **Initial pose.** Currently read from the bag's first `/tf` message. How the hidden run will provide
   the first pose is unclear; `run.py --init-pose` can set it manually.

---

## 7. Next steps (by priority)

| Priority | Work | Purpose | Evaluated on |
|---|---|---|---|
| 1 | Ask the organizers in the Q&A: is the test run day or night? How is the first pose given? | Decide whether lighting robustness is needed | — |
| 2 | **Speed up to ≤ 100 ms:** half-precision ALIKED, fewer keypoints, match 1 keyframe, localize `rl` every other frame, overlap image prefetch with inference | Recover the runtime points (5 %) | Laps 2 and 6 only, checking accuracy doesn't drop noticeably |
| 3 | **Submission material:** our own README (setup, one command, declaring the pretrained ALIKED / LightGlue), `requirements.txt`, git commit | Reproducibility is 10 % | Clean run from scratch |
| 4 | Lighting robustness: image normalization, stronger fallback when relocalization fails (loosen the gate after repeated failures) | Lower the daylight risk | Laps 2 and 6 with artificially darkened / brightened images |
| 5 | Find out why `cl` hurts (extrinsics?); also try `fr` and `r` | Possibly lower longitudinal error further | Laps 2 and 6 |
| 6 | Presentation: 5 minutes, max 6 slides, with system diagram, results, runtime, failure cases; trajectory plot or video | Presentation 5 %, technical assessment 15 % | — |

---

## 8. Reproduce

```bash
conda activate camloc                       # PyTorch 2.14 + CUDA, LightGlue
# training data preparation (once)
python3 tools/export_cache.py <bag> cache --img-every 10
python3 tools/extract_gt.py <bag> gt.csv
python3 scripts/split_laps.py gt.csv cache
python scripts/build_kf_map.py <bag> cache gt.csv maps/kf_fl.npz --cam fl
python scripts/build_kf_map.py <bag> cache gt.csv maps/kf_rl.npz --cam rl
# localization (one command, causal, online)
python run.py <bag> out/final/final.csv --cams fl,rl --pf-config configs/fl_rl.json
# scoring
python3 scripts/eval_laps.py gt.csv out/final/final.csv cache
```

Full held-out tuning pipeline: `scripts/overnight.sh`.
