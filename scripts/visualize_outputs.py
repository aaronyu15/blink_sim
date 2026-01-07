import argparse
import glob
import os
import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

# Add repo root so `src` imports resolve when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.flow_viz import flow_to_image


def load_pngs(dir_path):
    return sorted(Path(dir_path).glob('*.png'))


def load_npys(dir_path):
    return sorted(Path(dir_path).glob('*.npy'))


def load_events(events_h5):
    with h5py.File(events_h5, 'r') as f:
        ts = f['events/t'][:]
        y = f['events/y'][:]
        x = f['events/x'][:]
        p = f['events/p'][:]
    return ts, x, y, p


def events_to_image(ts, x, y, p, t0, t1, shape):
    h, w = shape
    mask0 = np.searchsorted(ts, t0, side='left')
    mask1 = np.searchsorted(ts, t1, side='left')
    sub_x = x[mask0:mask1]
    sub_y = y[mask0:mask1]
    sub_p = p[mask0:mask1]
    img = np.zeros((h, w, 3), dtype=np.float32)
    if sub_x.size == 0:
        return img
    pos = sub_p > 0
    neg = ~pos
    sub_x = np.clip(sub_x, 0, w - 1)
    sub_y = np.clip(sub_y, 0, h - 1)
    img[sub_y[pos], sub_x[pos], 0] += 1.0  # red for positive
    img[sub_y[neg], sub_x[neg], 2] += 1.0  # blue for negative
    if img.max() > 0:
        img = img / img.max()
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='Path to sequence output dir, e.g., output/train/000000')
    parser.add_argument('--event_fps', type=float, default=200.0, help='FPS used for event_input and DVS events')
    parser.add_argument('--flow_events_only', action='store_true', help='Show only forward flow and events (skip RGB/event-input images)')
    parser.add_argument('--event_window_ms', type=float, default=None, help='Optional event slice window (ms). If set, increases/decreases event time resolution independent of frame counts.')
    parser.add_argument('--auto_play', action='store_true', help='Enable automatic animation instead of manual slider')
    parser.add_argument('--flow_eps', type=float, default=1e-4, help='Zero-out flow vectors with magnitude below this threshold to avoid noisy first-frame artifacts')
    args = parser.parse_args()

    out = Path(args.output_dir)
    rgb_dir = out / 'rgb_reference'
    event_dir = out / 'rgb_event_input'
    events_h5 = out / 'events_left' / 'events.h5'
    flow_dir = out / 'forward_flow'

    # We load event/pngs regardless for timing calculations, but render conditionally
    rgb_files = load_pngs(rgb_dir)
    event_files = load_pngs(event_dir)
    flow_files = load_npys(flow_dir)
    ts, ex, ey, ep = load_events(events_h5)

    have_flow = len(flow_files) > 0
    if args.flow_events_only:
        if not have_flow:
            raise RuntimeError('No forward_flow files found')
        base_frames = len(flow_files)
        flow_sample = np.load(flow_files[0])
        h_hdr, w_hdr = flow_sample.shape[0], flow_sample.shape[1]
    else:
        if len(event_files) == 0:
            raise RuntimeError('No rgb_event_input frames found in output_dir')
        base_frames = len(event_files)
        sample = plt.imread(event_files[0])
        h_hdr, w_hdr = sample.shape[0], sample.shape[1]
        if len(rgb_files) == 0:
            print("Warning: RGB reference missing; viewer will skip RGB panel")

    # Precompute event timing; allow manual window or default to event_fps pacing
    if len(ts) > 0:
        total_us = max(1.0, ts[-1] - ts[0])
        t_start_us = ts[0]
    else:
        # Fallback: derive total duration from event frames and fps
        total_us = max(1.0, len(event_files) * (1e6 / args.event_fps))
        t_start_us = 0.0

    if args.event_window_ms is not None:
        interval_us = max(1.0, args.event_window_ms * 1000.0)
    else:
        interval_us = 1e6 / args.event_fps

    # Trim playback to the actual event span so DVS does not vanish before the animation ends
    if len(ts) > 0:
        time_frames = int(np.ceil(total_us / interval_us))
        num_frames = min(base_frames, time_frames) if base_frames > 0 else time_frames
        if time_frames < base_frames:
            print(f"Info: event stream shorter than image/flow count (event frames: {time_frames}, base: {base_frames}); trimming playback to event duration.")
    else:
        num_frames = base_frames

    duration_sec = total_us / 1e6
    rgb_count = len(rgb_files)
    evt_count = len(event_files)
    flow_count = len(flow_files)
    rgb_fps_est = rgb_count / duration_sec if rgb_count > 0 else 0.0
    flow_fps_est = flow_count / duration_sec if flow_count > 0 else 0.0
    event_fps_est = args.event_fps

    if args.flow_events_only:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax_flow, ax_evt = axes.flatten()
        ims = {
            'flow': ax_flow.imshow(np.zeros((h_hdr, w_hdr, 3))),
            'evt': ax_evt.imshow(np.zeros((h_hdr, w_hdr, 3)))
        }
        ax_flow.set_title('Optical flow (forward)')
        ax_evt.set_title('DVS events')
        for ax in axes.flatten():
            ax.axis('off')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        ax_rgb, ax_hdr, ax_flow, ax_evt = axes.flatten()
        ims = {
            'rgb': ax_rgb.imshow(np.zeros((h_hdr, w_hdr, 3))),
            'hdr': ax_hdr.imshow(np.zeros((h_hdr, w_hdr, 3))),
            'flow': ax_flow.imshow(np.zeros((h_hdr, w_hdr, 3))),
            'evt': ax_evt.imshow(np.zeros((h_hdr, w_hdr, 3)))
        }
        ax_rgb.set_title('RGB reference')
        ax_hdr.set_title('Event input (LDR)')
        ax_flow.set_title('Optical flow (forward)')
        ax_evt.set_title('DVS events')
        for ax in axes.flatten():
            ax.axis('off')

    def update(i):
        i = int(np.clip(i, 0, num_frames - 1))
        t0 = t_start_us + i * interval_us
        t_mid_s = (t0 - t_start_us) / 1e6  # time since start in seconds

        if args.flow_events_only:
            flow_idx = min(int(t_mid_s * flow_fps_est + 1e-6), len(flow_files) - 1)
        else:
            if rgb_count > 0:
                rgb_idx = min(int(t_mid_s * rgb_fps_est + 1e-6), len(rgb_files) - 1)
                rgb = plt.imread(rgb_files[rgb_idx])
                ims['rgb'].set_data(rgb)
            flow_idx = min(int(t_mid_s * flow_fps_est + 1e-6), len(flow_files) - 1) if have_flow else 0
            if evt_count > 0:
                evt_idx = min(int(t_mid_s * event_fps_est + 1e-6), len(event_files) - 1)
                evt_frame = plt.imread(event_files[evt_idx])
                ims['hdr'].set_data(evt_frame)

        if have_flow and flow_idx >= 0:
            flow_arr = np.load(flow_files[flow_idx])
            flow_arr = np.nan_to_num(flow_arr, copy=False)
            if flow_arr.ndim == 3 and flow_arr.shape[2] == 2:
                flow_vec = flow_arr.copy()
                mag = np.linalg.norm(flow_vec, axis=2)
                flow_vec[mag < args.flow_eps] = 0.0
                flow_img = flow_to_image(flow_vec).astype(np.float32) / 255.0
            elif flow_arr.ndim == 3 and flow_arr.shape[2] == 3:
                # Interpret as [u, v, valid] but visualize only u, v with standard color wheel
                flow_vec = flow_arr[:, :, :2]
                mag = np.linalg.norm(flow_vec, axis=2)
                flow_vec = flow_vec.copy()
                flow_vec[mag < args.flow_eps] = 0.0
                flow_img = flow_to_image(flow_vec).astype(np.float32) / 255.0
            else:
                flow_img = np.ones((h_hdr, w_hdr, 3))
        else:
            flow_img = np.ones((h_hdr, w_hdr, 3))
        ims['flow'].set_data(flow_img)

        t1 = t_start_us + (i + 1) * interval_us
        evt_img = events_to_image(ts, ex, ey, ep, t0, t1, (h_hdr, w_hdr))
        ims['evt'].set_data(evt_img)

        fig.canvas.draw_idle()

    if args.auto_play:
        interval_ms = args.event_window_ms if args.event_window_ms is not None else 1000.0 / args.event_fps
        _ = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval_ms, blit=False)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.show()
    else:
        slider_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])
        frame_slider = Slider(slider_ax, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)
        frame_slider.on_changed(update)
        update(0)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.show()


if __name__ == '__main__':
    main()

"""

python scripts/visualize_outputs.py --output_dir output/train/000000 --event_fps 200 --event_window_ms 2 --auto_play
python scripts/visualize_outputs.py --output_dir output/train/000000 --event_fps 200 --flow_events_only --event_window_ms 2 --auto_play
"""