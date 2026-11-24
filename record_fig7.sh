#!/usr/bin/env bash
# Record the Fig. 7-style touch rollouts (viewer snapshots + GIF frames + touch signals) for each policy.
#   bash record_fig7.sh                     # all four sets that are not recorded yet
#   bash record_fig7.sh park_best           # one set;  FORCE=1 re-records an existing one
#   OBJ=col_110 FORCE=1 bash record_fig7.sh # another held-out object (default col_092)
# Every set rotates the same held-out object with the same seed, so the figures are comparable.
# A viewer window opens for each recording. Output: rollouts/fig7/<set>/ (then: bash make_deliverables.sh)
set -eo pipefail
declare -A CKPT=([baseline_best]=checkpoints/policies/baseline_ep4500.pth
                 [baseline_latest]=checkpoints/policies/baseline_ep5000.pth
                 [park_best]=checkpoints/policies/park_ep3600.pth
                 [park_latest]=checkpoints/policies/park_ep5000.pth)
COMMON="task=AllegroArmMOAR test=True seed=7 task.env.objSet=columns_test task.env.axis=z task.env.sensor=split4
  task.env.observationType=partial_stack task.env.ablation_mode=no-pc task.env.legacy_obs=True task.env.numEnvs=16
  +task.env.record_object=${OBJ:-col_092} +task.env.record_gif=True"
PARK="+task.env.touch_model=park +task.env.park_E=338e3 +task.env.park_log_eps=0.003 +task.env.park_calib=False"
SETS=${@:-baseline_best park_best baseline_latest park_latest}
missing=0
for s in $SETS; do
  [ -n "${CKPT[$s]}" ] || { echo "ERROR: unknown set '$s' (baseline_best park_best baseline_latest park_latest)"; exit 1; }
  out=rollouts/fig7/$s
  if [ -f $out/touch_rollout.npz ] && [ -z "$FORCE" ]; then echo "== $s: already recorded ($out)"; continue; fi
  if [ ! -s "${CKPT[$s]}" ]; then
    echo "ERROR $s: policy checkpoint ${CKPT[$s]} not found (see REPRODUCE.md, organize.sh)"; missing=1; continue
  fi
  extra=""; [[ $s == park_* ]] && extra=$PARK
  echo "== $s: ${CKPT[$s]}"
  python isaacgymenvs/train.py $COMMON $extra checkpoint=${CKPT[$s]} +task.env.record_dir=$out
  [ -s $out/touch_rollout.npz ] || { echo "ERROR $s: recording did not produce $out/touch_rollout.npz"; missing=1; }
done
[ $missing -eq 0 ] || { echo "some recordings are missing (errors above)"; exit 1; }
