#!/usr/bin/env bash
# Rebuild the presentation figures from what is on disk and collect them in deliverables/.
#   bash make_deliverables.sh              # redraw every figure from the current inputs (overwrites them)
#   CLEAN=1 bash make_deliverables.sh      # empty deliverables/ first, so no figure from older inputs is left
#   STRICT=1 bash make_deliverables.sh     # stop if any input is missing (for a full reproduction)
# Re-run any time: figures whose inputs are missing are skipped with a note, so it can be run again as the
# cluster rollout sets and their CNN runs come in.
#
# Inputs (edit the paths below if yours differ):
#   TB_BASE / TB_PARK / TB_PARKCAL  TensorBoard folders of the three training runs (cluster_runs/...)
#   rollouts/fig7/<set>/            touch recordings (record_fig7.sh)
#   models/shape_recon/<set>/seed*/ trained CNNs and their results (train_recon.sh)
set -o pipefail
TB_BASE=${TB_BASE:-cluster_runs/runs/baseline-binary-split4}
TB_PARK=${TB_PARK:-cluster_runs/runs/park-ds20-fixed-split4}
TB_PARKCAL=${TB_PARKCAL:-cluster_runs/runs/park-ds20-split4}
D=deliverables

# ---- inputs check: what each figure needs, and whether it is there. STRICT=1 stops if anything is missing.
declare -A POLICY=([baseline_best]=baseline_ep4500 [baseline_latest]=baseline_ep5000 [park_best]=park_ep3600 [park_latest]=park_ep5000)
missing=0
ok() { printf "  %-4s %s\n" "$1" "$2"; [ "$1" = ok ] || missing=1; }
echo "== inputs"
for tb in $TB_BASE $TB_PARK; do
  ls $tb/*/summaries/events.out.tfevents* >/dev/null 2>&1 && ok ok "training log $tb" || ok MISS "training log $tb"
done
for s in baseline_best park_best baseline_latest park_latest; do
  [ -s checkpoints/policies/${POLICY[$s]}.pth ] && ok ok "policy checkpoints/policies/${POLICY[$s]}.pth" \
    || ok MISS "policy checkpoints/policies/${POLICY[$s]}.pth"
  [ -s rollouts/fig7/$s/touch_rollout.npz ] && ok ok "Fig. 7 recording rollouts/fig7/$s" \
    || ok MISS "Fig. 7 recording rollouts/fig7/$s  (bash record_fig7.sh $s)"
  sig=binary; [[ $s == park_* ]] && sig=park
  n=$(ls models/shape_recon/$s/seed*/model_$sig.pt 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && ok ok "CNN models/shape_recon/$s ($n seeds)" \
    || ok MISS "CNN models/shape_recon/$s  (shards in rollouts/recon/$s, then bash train_recon.sh $s)"
done
if [ $missing -ne 0 ]; then
  [ -n "$STRICT" ] && { echo "STRICT=1: stopping, inputs missing"; exit 1; }
  echo "  (figures for missing inputs are skipped or left blank; STRICT=1 makes this an error)"
fi

if [ -n "$CLEAN" ]; then rm -rf $D; echo "== CLEAN=1: emptied $D/"; fi
mkdir -p $D/1_training_curves $D/2_touch_rollouts $D/3_shape_reconstruction

echo "== 1. training curves (Fig. 6 style)"
if ls $TB_BASE $TB_PARK >/dev/null 2>&1; then
  python plot_training_curves.py --run "Binary (baseline)=$TB_BASE" --run "Park skin, fixed p*=$TB_PARK" \
      --mark "Binary (baseline)=4500" --mark "Park skin, fixed p*=3600" \
      --out $D/1_training_curves/training_curves.png
  if ls $TB_PARKCAL >/dev/null 2>&1; then
    python plot_training_curves.py --run "Binary (baseline)=$TB_BASE" --run "Park skin, fixed p*=$TB_PARK" \
        --run "Park skin, calibration noise=$TB_PARKCAL" \
        --out $D/1_training_curves/training_curves_with_calibrated_park.png
  fi
else
  echo "   skipped: set TB_BASE / TB_PARK to the TensorBoard folders"
fi

echo "== 2. touch rollouts (Fig. 7 style)"
for r in baseline_best park_best baseline_latest park_latest; do
  dir=rollouts/fig7/$r
  if [ -f $dir/touch_rollout.npz ]; then
    python plot_touch_rollout.py $dir --all --deliver $D/2_touch_rollouts
  else
    echo "   skipped $r: no $dir/touch_rollout.npz"
  fi
done

echo "== 3. shape reconstruction (Fig. 9 style) and held-out MSE"
python plot_recon_compare.py --out $D/3_shape_reconstruction --objects ${RECON_OBJECTS:-col_006 col_054 col_110}
python plot_recon_mse.py --out $D/3_shape_reconstruction/recon_mse.png | tee $D/3_shape_reconstruction/recon_mse_table.txt

echo
find $D -type f | sort
