#!/usr/bin/env bash
# Collect shape-reconstruction windows until each split reaches its target (paper: 55,000 rollouts of
# 200 steps over 125 objects, i.e. about 440 per object -> 44,000 train / 11,000 held-out windows).
# Each round: record 500 steps of every env with a new seed, cut it into drop-free 200-step windows
# (build_recon_dataset.py -> one shard), delete the raw recording, update the counts.
# Usage (from the repo root, soft-syn env):  bash collect_recon.sh [checkpoint] [train_target] [test_target]
set -eo pipefail
CKPT=${1:-checkpoints/baseline_final.pth}
TARGET_train=${2:-44000}
TARGET_test=${3:-11000}
ENVS=1000
OUT=rollouts/recon_shards
mkdir -p "$OUT"

count() {  # windows so far in shards of split $1
  python - "$OUT" "$1" <<'PY'
import glob, sys, numpy as np
n = sum(int((np.load(f)["split"] == sys.argv[2]).sum()) for f in glob.glob(f"{sys.argv[1]}/*.npz"))
print(n)
PY
}

seed=$(( $(ls "$OUT" 2>/dev/null | wc -l) + 100 ))   # resume-safe: new seeds after existing shards
for SET in train test; do
  target_var=TARGET_$SET; target=${!target_var}
  while true; do
    have=$(count $SET)
    echo "[collect] $SET: $have / $target windows"
    [ "$have" -ge "$target" ] && break
    raw=rollouts/recon_raw_${SET}_s${seed}
    python isaacgymenvs/train.py task=AllegroArmMOAR test=True checkpoint=$CKPT headless=True seed=$seed \
      task.env.objSet=columns_$SET task.env.axis=z task.env.sensor=split4 \
      task.env.observationType=partial_stack task.env.ablation_mode=no-pc task.env.legacy_obs=True \
      task.env.numEnvs=$ENVS +task.env.objects_in_order=True +task.env.player_games=100000 \
      +task.env.recon_dir=$raw +task.env.recon_steps=500 2>&1 | grep -E "^\[recon\] (saved|[0-9]+ envs)|rror" || true
    [ -f $raw/recon_rollout.npz ] || { echo "[collect] recording failed (seed $seed)"; exit 1; }
    python build_recon_dataset.py $raw/recon_rollout.npz --out $OUT/${SET}_s${seed}.npz | grep -E "kept|windows over"
    rm -rf $raw
    seed=$((seed + 1))
  done
done
python - "$OUT" <<'PY'
import glob, sys, numpy as np
fs = sorted(glob.glob(f"{sys.argv[1]}/*.npz"))
obj = np.concatenate([np.load(f)["object"] for f in fs]); sp = np.concatenate([np.load(f)["split"] for f in fs])
for s in ("train", "test"):
    o, c = np.unique(obj[sp == s], return_counts=True)
    print(f"[collect] done {s}: {c.sum()} windows over {len(o)} objects, per object min {c.min()} median {int(np.median(c))} max {c.max()}")
PY
