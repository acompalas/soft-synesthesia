"""Figure 7-style plot (Yin et al. 2023) from a recorded touch rollout.

Top row: viewer snapshots of env 0. Bottom: the 16 touch signals the policy received over the
rollout, one trace per sensor, with shaded windows marking when each snapshot was taken.
Binary rollouts show on/off traces. Park rollouts (recorded with touch_model=park) show each pad's
continuous reading as a filled trace: the policy's readout of dR/R0 (log readout when park_log_eps
is set), 0 = no load, full height = channel closed.

Input is the directory written by the task recorder (+task.env.record_dir=...):
  touch_rollout.npz, snap_XXXX.png

Usage:
  python plot_touch_rollout.py rollouts/smoke
  python plot_touch_rollout.py rollouts/smoke --crop 0.6 --no-show
  python plot_touch_rollout.py rollouts/camtest --cams      # contact sheet of camera-test images
  python plot_touch_rollout.py rollouts/smoke --gif         # animated GIF (needs +task.env.record_gif=True)
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

COLORS = ["#1f3a93", "#b22222", "#d35400", "#1e8449", "#6c3483"]  # cycled per sensor row
WIN_HALF = 7  # shaded window half-width in control steps (about 15 steps wide, as in Fig. 7)


def signal_info(d):
    """Title text for the recorded touch model."""
    model = str(d["touch_model"]) if "touch_model" in d.files else "binary"
    if model != "park":
        return False, "Binary touch signals"
    eps = float(d["park_log_eps"]) if "park_log_eps" in d.files else 0.0
    E = float(d["park_E"]) if "park_E" in d.files else 69e3
    readout = f"log readout of dR/R0 (eps {eps:g})" if eps > 0 else "dR/R0 / 9"
    return True, f"Park skin readings (E = {E / 1e3:.0f} kPa, {readout}; full height = channel closed)"


def draw_traces(ax, sensed, park, lw):
    t = np.arange(sensed.shape[0])
    for i in range(sensed.shape[1]):
        c = COLORS[i % len(COLORS)]
        if park:
            ax.fill_between(t, i, i + 0.8 * sensed[:, i], color=c, alpha=0.35, lw=0, step="mid")
            ax.plot(t, i + 0.8 * sensed[:, i], color=c, lw=lw, drawstyle="steps-mid")
        else:
            ax.plot(t, i + 0.6 * sensed[:, i], color=c, lw=lw)


def center_crop(img, frac):
    if frac >= 1.0:
        return img
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def make_gif(args):
    """Animated version: viewer frame on top, touch chart with a moving time cursor below.
    Dots on the cursor mark the sensors that are on at that step."""
    from PIL import Image

    d = np.load(os.path.join(args.rec_dir, "touch_rollout.npz"))
    sensed = d["sensed"]
    T, n_sensors = sensed.shape
    resets = np.where(d["reset"])[0]
    frames = sorted(glob.glob(os.path.join(args.rec_dir, "frames", "frame_*.png")))
    if not frames:
        raise SystemExit("no frames found; record with +task.env.record_gif=True")
    steps = [int(os.path.basename(f)[6:10]) for f in frames]
    stride = steps[1] - steps[0] if len(steps) > 1 else 1

    fig = plt.figure(figsize=(7.5, 7.0), dpi=90)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.12)
    ax_img = fig.add_subplot(gs[0]); ax_img.axis("off")
    first = center_crop(plt.imread(frames[0]), args.crop)
    im = ax_img.imshow(first)
    ax = fig.add_subplot(gs[1])
    park, label = signal_info(d)
    draw_traces(ax, sensed, park, 0.8)
    for r in resets:
        ax.axvline(r + 0.5, color="0.4", ls="--", lw=0.8)
    ax.set_xlim(0, T); ax.set_ylim(-0.5, n_sensors + 0.2)
    ax.set_yticks(range(n_sensors)); ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("control step (10 Hz)"); ax.set_ylabel("sensor")
    ax.set_title(f"{label}, object {d['object']}", fontsize=8)
    cursor = ax.axvline(0, color="#5b5bff", lw=2, alpha=0.7)
    dots = ax.scatter([], [], s=18, color="k", zorder=5)

    images = []
    for f, s in zip(frames, steps):
        im.set_data(center_crop(plt.imread(f), args.crop))
        cursor.set_xdata([s, s])
        row = sensed[min(s, T - 1)]
        on = np.where(row > 0)[0]
        dots.set_offsets(np.column_stack([np.full(len(on), s), on + (0.8 * row[on] if park else 0.6)])
                         if len(on) else np.empty((0, 2)))
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
    plt.close(fig)

    out = args.out or os.path.join(args.rec_dir, "touch_rollout.gif")
    duration_ms = int(round(100 * stride / args.gif_speed))  # one control step = 100 ms
    images[0].save(out, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=True)
    print(f"saved {os.path.abspath(out)} ({len(images)} frames, {duration_ms} ms each)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec_dir")
    ap.add_argument("--crop", type=float, default=1.0, help="centre-crop fraction for snapshots (default 1.0 = none)")
    ap.add_argument("--out", default=None, help="output PNG (default <rec_dir>/touch_rollout.png)")
    ap.add_argument("--no-show", dest="show", action="store_false")
    ap.add_argument("--cams", action="store_true",
                    help="tile the camera-test images (cam_*.png) into one contact sheet instead")
    ap.add_argument("--gif", action="store_true",
                    help="make touch_rollout.gif from frames/ (recorded with +task.env.record_gif=True)")
    ap.add_argument("--gif-speed", type=float, default=1.0, help="playback speed, 1.0 = real time (10 control steps/s)")
    args = ap.parse_args()
    if not args.show or args.gif:
        matplotlib.use("Agg")

    if args.gif:
        make_gif(args)
        return

    if args.cams:
        files = sorted(glob.glob(os.path.join(args.rec_dir, "cam_*.png")))
        cols = 4
        rows = int(np.ceil(len(files) / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), squeeze=False)
        for ax in axs.ravel():
            ax.axis("off")
        for ax, f in zip(axs.ravel(), files):
            ax.imshow(center_crop(plt.imread(f), args.crop))
            ax.set_title(os.path.basename(f)[4:-4], fontsize=9)  # e.g. d0.15_az-045_el35
        fig.tight_layout()
        out = args.out or os.path.join(args.rec_dir, "camera_sheet.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"saved {os.path.abspath(out)}")
        if args.show:
            plt.show()
        return

    d = np.load(os.path.join(args.rec_dir, "touch_rollout.npz"))
    sensed = d["sensed"]                 # (T, 16): binary {0, 1}, or Park readings in [0, 1]
    park, label = signal_info(d)
    resets = np.where(d["reset"])[0]     # steps after which env 0 was reset
    snaps = list(d["snaps"])
    T, n_sensors = sensed.shape
    snap_files = {int(os.path.basename(f)[5:9]): f for f in glob.glob(os.path.join(args.rec_dir, "snap_*.png"))}

    n = len(snaps)
    fig = plt.figure(figsize=(3.2 * n, 6.4))
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 1.15], hspace=0.08, wspace=0.03)

    for k, s in enumerate(snaps):
        ax = fig.add_subplot(gs[0, k])
        ax.axis("off")
        if s in snap_files:
            ax.imshow(center_crop(plt.imread(snap_files[s]), args.crop))
        else:
            ax.text(0.5, 0.5, f"no snapshot\nfor step {s}", ha="center", va="center")
        ax.text(0.97, 0.04, str(k + 1), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=12, color="white", bbox=dict(facecolor="black", alpha=0.4, pad=1, lw=0))

    ax = fig.add_subplot(gs[1, :])
    draw_traces(ax, sensed, park, 1.0)
    for k, s in enumerate(snaps):
        ax.axvspan(s - WIN_HALF, s + WIN_HALF, color="#5b5bff", alpha=0.25, lw=0)
        ax.text(s, n_sensors - 0.1, str(k + 1), ha="center", va="bottom", fontsize=9)
    for r in resets:
        ax.axvline(r + 0.5, color="0.4", ls="--", lw=0.8)
    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, n_sensors + 0.6)
    ax.set_yticks(range(n_sensors))
    if "sensor_names" in d.files:
        ax.set_yticklabels([str(n).replace("_fsr", "") for n in d["sensor_names"]])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("control step (10 Hz)")
    ax.set_ylabel("sensor")
    ax.grid(axis="x", color="0.85", lw=0.6)
    title = f"{label}, env 0, object {d['object']}"
    if len(resets):
        title += " (dashed: episode reset)"
    ax.set_title(title, fontsize=9)

    out = args.out or os.path.join(args.rec_dir, "touch_rollout.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {os.path.abspath(out)}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
