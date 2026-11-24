#!/usr/bin/env bash
# Evaluate late checkpoints of every run with the same rule (deterministic policy, first episode of every env).
#   bash eval_checkpoints.sh download     copy the checkpoints from the cluster drive to checkpoints/eval/
#   bash eval_checkpoints.sh run          evaluate each checkpoint on the train and held-out objects
#   python summarize_eval.py               table of time to fall, rotation and drops per run and epoch
# Train objects: 500 envs (5 per object); held-out: 500 envs (20 per object); up to 500 steps = 50 s.
set -eo pipefail
EPOCHS="4000 4500 4900 5000"
NS=erl-ucsd-supp
POD=${POD:-anderson-softsyn-park-fixed-zpv8f}     # any running pod that mounts the drive
declare -A DIR=( [baseline]=baseline-binary-split4 [park]=park-ds20-split4 [parkfixed]=park-ds20-fixed-split4 )
declare -A FLAGS=( [baseline]=""
                   [park]="+task.env.touch_model=park +task.env.park_E=338e3 +task.env.park_log_eps=0.003"
                   [parkfixed]="+task.env.touch_model=park +task.env.park_E=338e3 +task.env.park_log_eps=0.003 +task.env.park_calib=False" )

if [ "$1" = download ]; then
  mkdir -p checkpoints/eval
  for run in baseline park parkfixed; do for E in $EPOCHS; do
    dst=checkpoints/eval/${run}_ep${E}.pth
    [ -s $dst ] && { echo "have $dst"; continue; }
    src=$(kubectl exec -n $NS $POD -- bash -c "ls /data/soft-synesthesia/runs/${DIR[$run]}/*/nn/*_ep_${E}_rew_*.pth 2>/dev/null | head -1")
    [ -z "$src" ] && { echo "MISSING $run epoch $E"; continue; }
    kubectl exec -n $NS $POD -- cat "$src" > $dst && echo "got $dst  <- $(basename "$src")"
  done; done
  ls -lh checkpoints/eval
  exit 0
fi

if [ "$1" = run ]; then
  for run in baseline park parkfixed; do for E in $EPOCHS; do for SET in train test; do
    ck=checkpoints/eval/${run}_ep${E}.pth; out=rollouts/eval/${run}_ep${E}_${SET}
    [ -s $ck ] || { echo "skip $ck (missing)"; continue; }
    [ -f $out/eval.npz ] && { echo "done $out"; continue; }
    echo "== $run epoch $E $SET"
    python isaacgymenvs/train.py task=AllegroArmMOAR test=True checkpoint=$ck headless=True seed=7 \
      task.env.objSet=columns_$SET task.env.axis=z task.env.sensor=split4 \
      task.env.observationType=partial_stack task.env.ablation_mode=no-pc task.env.legacy_obs=True \
      task.env.numEnvs=500 +task.env.objects_in_order=True +task.env.player_games=100000 \
      ${FLAGS[$run]} +task.env.eval_dir=$out 2>&1 | grep -E "^\[eval\] (train|test)|rror" || true
  done; done; done
  exit 0
fi
echo "usage: bash eval_checkpoints.sh download|run"
