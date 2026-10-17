#!/bin/bash
# Speed/accuracy variants, evaluated on held-out tuning laps 2 and 6 only.
cd "$(dirname "$0")/.."
P=/home/kunting/test/Download/ENTER/envs/camloc/bin/python
BAG=c0_2026-10-13_22-53-42_camloc_0.mcap
run() { echo "VARIANT $*"; $P scripts/eval_variant.py $BAG cache "$@" 2>&1 | grep -v -i warn | tail -1; }
run --tag A_now      --cams fl,rl --n-kf 2
run --tag B_merge    --cams fl,rl --n-kf 2 --merge 1
run --tag C_1kf      --cams fl,rl --n-kf 1
run --tag D_merge_rl2 --cams fl,rl --n-kf 2 --merge 1 --every fl:1,rl:2
run --tag E_1kf_rl2  --cams fl,rl --n-kf 1 --every fl:1,rl:2
run --tag F_fl_only  --cams fl --n-kf 2
echo "VARIANTS DONE"
