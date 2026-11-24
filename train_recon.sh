#!/usr/bin/env bash
# Train the shape-reconstruction CNN on one rollout set, 5 seeds, with no touch and with the touch signal of the
# policy that produced the set (baseline sets: binary, Park sets: Park).
#   bash train_recon.sh baseline_latest            # sets: baseline_latest baseline_best park_latest park_best
#   SEEDS="1 2 3" bash train_recon.sh park_best
# Input:  rollouts/recon/<set>/*.npz (shards from collect_recon.sh)
# Output: models/shape_recon/<set>/seed<k>/ results.json, predictions_<signal>.npz, model_<signal>.pt
set -eo pipefail
SET=$1
[ -n "$SET" ] || { sed -n 2,8p "$0"; exit 1; }
n=$( (find -L rollouts/recon/$SET -maxdepth 1 -name '*.npz' 2>/dev/null || true) | wc -l)  # never trips set -e
[ "$n" -gt 0 ] || { echo "ERROR: no rollout shards in rollouts/recon/$SET (collect them first, see REPRODUCE.md)"; exit 1; }
echo "$SET: $n shards"
touch_sig=binary; [[ $SET == park_* ]] && touch_sig=park
for s in ${SEEDS:-1 2 3 4 5}; do
  echo "== $SET seed $s"
  python shape_recon.py --data rollouts/recon/$SET --signals none $touch_sig --seed $s \
      --out models/shape_recon/$SET/seed$s | tail -6
  for sig in none $touch_sig; do
    [ -s models/shape_recon/$SET/seed$s/model_$sig.pt ] || { echo "ERROR: seed $s did not save model_$sig.pt"; exit 1; }
  done
done
