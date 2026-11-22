# Sensor tests

The checks behind the touch sensors in the report. Run all of them from the repo root. The Park skin is Dragon Skin 20 (E = p* = 338 kPa) unless `--E` is given; Park's own EcoFlex 00-30 is `--E 69e3`.

The policy reads each pad through a log readout with resolution ε, r = max(0, log(g/ε)) / log(g_max/ε) with g = ΔR/R0 (`+task.env.park_log_eps`). This keeps typical touches visible next to saturated ones after input normalisation, and reads 0 below the resolution (about 0.2–0.4 N of pad force for Dragon Skin 20). The model tests check ΔR/R0 itself. `diagnose_park_firing.py` breaks a rollout's "on" readings down by the pad force behind them.

## Park skin model

| Test | What it shows | Command | Output |
|---|---|---|---|
| Analytic model | Every cell and every pad follows Park's Eq. 4. The same force on one cell never reads lower than on the whole pad, and a closed cell saturates the pad. | `python test_park_skin.py --no-show` | PASS/FAIL, `park_sensors.png`, `park_model.png` |
| Dead weight, whole pad (sim) | Plates of known weight on each of the 13 upward-facing pads follow Eq. 4, and the pad force matches the plate weight. | `python weight_test_park.py` | `weight_test/` |
| Same force, different area (sim) | Whole pad vs each single cell: the per-cell model holds, and the smaller area reads higher or the channel closes. | `python area_test_park.py` | `area_test/` |
| Policy rollout (sim) | The policy receives exactly the Park model of the simulated cell forces. Also reports how often pads saturate. | record with `+task.env.touch_model=park +task.env.park_E=338e3 +task.env.record_dir=rollouts/park_check task.env.latency=0 task.env.sensorNoise=0`, then `python verify_park_sensors.py rollouts/park_check` | PASS/FAIL, `park_sim_*.png` |

## Forces in the task

| Test | What it shows | Command | Output |
|---|---|---|---|
| Signal visibility | How far a touch moves each policy input after normalisation, Park (linear and log readouts) vs binary. This is why the readout is logarithmic. | `python signal_scale.py rollouts/park_check/touch_rollout.npz` | printout |
| Pad forces in rollouts | Distribution of pad forces under the trained policy, against the binary threshold and each skin's range. This is why the skin is Dragon Skin 20. | eval with `+task.env.force_stats=30` (see the `plot_force_stats.py` docstring), then `python plot_force_stats.py force_stats.npz` | `force_stats_*.png` |

## Split-pad binary baseline

| Test | What it shows | Command |
|---|---|---|
| Split sensor logic | Binary touch fires when the summed cell force crosses the threshold; no false firings from hand self-contact. | `python verify_split_sensors.py <rollout dir>` |
| Split vs original pad | A state replay of the same rollout on both robots: same contact events, with the split pad sum 1.26 to 1.45× the original. | `python compare_replay.py ...` (see its docstring) |

## Figures

- `plot_touch_rollout.py`: Fig. 7-style touch chart with snapshots and GIF.
- `visualize_columns_3d.py`: the training and test objects.

## Known limitation

With many hands in one simulation (80), the reported fingertip-pad forces under a static plate became unreliable, up to 20× the plate weight. The area test therefore runs each case as its own 16-hand simulation. Under the trained policy, 16 and 1024 envs give the same force distribution. Its top 1% (over 600 N) is likely solver spikes rather than real force.
