# Reproducing the presentation figures

All commands run from the repo root in the `soft-syn` conda env (Isaac Gym Preview 4). Every figure is
rebuilt by `bash make_deliverables.sh` into `deliverables/`; it skips anything whose inputs are missing.

## Layout

| Folder | Contents | In git |
|---|---|---|
| `checkpoints/policies/` | the four policies used: `baseline_ep4500.pth` (best reward), `baseline_ep5000.pth` (latest), `park_ep3600.pth` (best reward), `park_ep5000.pth` (latest). Park = Dragon Skin 20, fixed p*, log readout | no (large) |
| `cluster_runs/runs/<run>/` | TensorBoard logs of the three training runs (5000 epochs, 1 seed each) | no |
| `rollouts/recon/<set>/` | reconstruction shards: drop-free 200-step windows, 44k train + 11k test per set | no |
| `rollouts/fig7/<set>/` | touch recordings: snapshots, GIF frames, touch signals | no |
| `models/shape_recon/<set>/seed<k>/` | trained CNNs (`model_<signal>.pt`), `results.json`, held-out predictions | no |
| `deliverables/` | the figures | |

`<set>` is one of `baseline_latest`, `baseline_best`, `park_latest`, `park_best`.

Presentation objects (held-out): Fig. 7 rotates `col_092` (four-sided); Fig. 9 shows `col_006`, `col_067`, `col_110`.
The held-out MSE is always over all 25 test objects. Override with `OBJ=<name> bash record_fig7.sh` and
`RECON_OBJECTS="a b c" bash make_deliverables.sh`.

## Steps

| Figure | Paper analogue | Command |
|---|---|---|
| Training curves: reward and time to fall | Fig. 6 | `bash make_deliverables.sh` (reads `cluster_runs/runs/`) |
| Snapshots + touch signals, GIFs | Fig. 7 | `bash record_fig7.sh`, then `bash make_deliverables.sh` |
| Shape reconstruction meshes, held-out MSE | Fig. 9 | `bash train_recon.sh <set>` for each set, then `bash make_deliverables.sh` |

Producing the inputs from scratch:
- Policies: `k8s/job-softsyn-baseline.yaml`, `k8s/job-softsyn-park-fixed.yaml` (cluster jobs, about 38 h each).
- Reconstruction shards: `bash collect_recon.sh <checkpoint>` locally, or `make_recon_job.sh` on the cluster.
  Park policies need the Park flags: `+task.env.touch_model=park +task.env.park_E=338e3 +task.env.park_log_eps=0.003 +task.env.park_calib=False`.
- Checks of the reconstruction pipeline: `python test_shape_recon.py --data <one train shard> <one test shard>`.
