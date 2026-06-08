#!/usr/bin/env python3
"""
Visualize event frames (accumulated from event stream) alongside ground-truth flow.

This script reads the saved event stream from:
    events.h5
and flow metadata from:
    flow.h5

For each RGB/flow frame i, events are accumulated over the interval
[frame_event_start[i], frame_event_end[i]) in event-frame index space,
which corresponds to the time between consecutive GT flow frames.

Usage:
  python scripts/visualize_events_and_flow.py output/train/<sample_name>
"""

import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.widgets import Slider

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.flow_viz import flow_to_image


def load_event_stream(sample_dir):
    events_path = os.path.join(sample_dir, "events.h5")
    if not os.path.exists(events_path):
        # Backward compatibility with older layout
        events_path = os.path.join(sample_dir, "events_left", "events.h5")
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Missing event stream file: {events_path}")

    with h5py.File(events_path, "r") as f:
        t_us = f["events/t"][:]  # uint64 microseconds
        y = f["events/y"][:]     # uint16
        x = f["events/x"][:]     # uint16
        p = f["events/p"][:]     # uint8 in {0,1}

    return t_us, y, x, p


def load_flow_data(sample_dir):
    flow_path = os.path.join(sample_dir, "flow.h5")
    if not os.path.exists(flow_path):
        # Backward compatibility with consolidated render file
        flow_path = os.path.join(sample_dir, "hdf5", "flow.hdf5")
    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"Missing flow file: {flow_path}")

    with h5py.File(flow_path, "r") as f:
        if "flow/forward" in f:
            forward_flow = f["flow/forward"][:]                 # [N, H, W, 2]
            event_start_us = f["flow/event_start"][:]           # [N] absolute us
            event_end_us = f["flow/event_end"][:]               # [N] absolute us
            # Optionally, also get event indices (not used for slicing, but available)
            frame_event_start_idx = f["flow/frame_event_start"][:] if "flow/frame_event_start" in f else None
            frame_event_end_idx = f["flow/frame_event_end"][:] if "flow/frame_event_end" in f else None
        else:
            # Backward compatibility for old files
            forward_flow = f["forward_flow"][:]            # [N, H, W, 2]
            event_start_us = None
            event_end_us = None
            frame_event_start_idx = f["frame_event_start"][:] if "frame_event_start" in f else None
            frame_event_end_idx = f["frame_event_end"][:] if "frame_event_end" in f else None

    return forward_flow, event_start_us, event_end_us, frame_event_start_idx, frame_event_end_idx


def flow_to_middlebury_rgb(flow):
    # Use shared Middlebury color wheel visualization.
    return flow_to_image(flow)


def build_binary_events_image(
    t_us,
    y,
    x,
    p,
    t0_us,
    t1_us,
    height,
    width,
):
    # Events are sorted by time; use searchsorted for fast slicing.
    i0 = np.searchsorted(t_us, t0_us, side="left")
    i1 = np.searchsorted(t_us, t1_us, side="left")

    evt_img = np.zeros((height, width), dtype=np.uint8)
    if i1 <= i0:
        return evt_img

    xs = np.clip(x[i0:i1].astype(np.int32), 0, width - 1)
    ys = np.clip(y[i0:i1].astype(np.int32), 0, height - 1)
    # Binary event frame: pixel is 1 if at least one event occurred in interval.
    evt_img[ys, xs] = 1
    return evt_img


def event_idx_to_time_us(event_idx, event_fps):
    # event_idx is frame index in the event input stream.
    # Convert to microseconds for slicing the event stream.
    return int(round((float(event_idx) / float(event_fps)) * 1e6))


def infer_event_time_offset_us(sample_dir, event_fps):
    """Infer absolute event time offset introduced by initial trim."""
    # Preferred: explicit offset if present in events file
    events_path = os.path.join(sample_dir, "events.h5")
    if os.path.exists(events_path):
        try:
            with h5py.File(events_path, "r") as f:
                if "t_offset_us" in f:
                    return int(f["t_offset_us"][()])
        except Exception:
            pass

    print("Could not find explicit event time offset in events.h5; attempting to infer from config...")
    # Fallback: derive from per-sample config
    cfg_path = os.path.join(sample_dir, "config_job.yaml")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            trim_rgb = int(cfg.get("trim_initial_rgb_frames", 0))
            rgb_fps = float(cfg.get("rgb_image_fps", 30.0))
            evt_fps = float(cfg.get("event_image_fps", event_fps))
            trim_evt = int(round(trim_rgb * evt_fps / rgb_fps))
            return int(round((trim_evt / evt_fps) * 1e6))
        except Exception:
            pass

    return 0


def main():
    parser = argparse.ArgumentParser(description="Visualize event frame vs flow with slider")
    parser.add_argument("sample_dir", type=str, help="Path to one sample directory")
    parser.add_argument("--event-fps", type=float, default=300.0, help="Event input FPS used in simulation")
    parser.add_argument(
        "--event-time-offset-us",
        type=int,
        default=None,
        help="Optional absolute timestamp offset added to frame_event_start/end time conversion. "
             "If omitted, inferred from sample metadata.",
    )
    parser.add_argument(
        "--event-index-offset",
        type=int,
        default=None,
        help="Optional manual offset subtracted from frame_event_start/end before time conversion. "
             "If omitted, uses 0 (recommended for new flow.h5 metadata).",
    )
    args = parser.parse_args()

    sample_dir = args.sample_dir
    if not os.path.isdir(sample_dir):
        print(f"Error: sample directory does not exist: {sample_dir}")
        sys.exit(1)

    print("Loading event stream from events.h5 ...")
    t_us, y, x, p = load_event_stream(sample_dir)

    print("Loading flow and frame-event metadata from flow.h5 ...")
    forward_flow, event_start_us, event_end_us, frame_event_start_idx, frame_event_end_idx = load_flow_data(sample_dir)

    n = forward_flow.shape[0]
    h, w = forward_flow.shape[1], forward_flow.shape[2]

    print(f"RGB/flow frames: {n}")
    print(f"Event count: {len(t_us)}")
    if len(t_us) > 0:
        print(f"Event time range: {t_us[0]}us .. {t_us[-1]}us")
        duration_s = max(0.0, (float(t_us[-1]) - float(t_us[0])) / 1e6)
        if duration_s > 0.0:
            event_rate_kev_s = (len(t_us) / duration_s) / 1000.0
        else:
            event_rate_kev_s = 0.0
        print(f"Event rate: {event_rate_kev_s:.3f} kev/s")
    else:
        print("Event rate: 0.000 kev/s")

    fig, (ax_evt, ax_flow) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.20)

    def draw(frame_idx):
        ax_evt.clear()
        ax_flow.clear()

        # Use new absolute event time window keys for slicing
        if event_start_us is not None and event_end_us is not None:
            t0_us = int(event_start_us[frame_idx])
            t1_us = int(event_end_us[frame_idx])
            idx_info = f"us [{t0_us},{t1_us})"
        elif frame_event_start_idx is not None and frame_event_end_idx is not None:
            # Fallback: use event indices if absolute times are missing
            t0_us = t_us[frame_event_start_idx[frame_idx]] if frame_event_start_idx[frame_idx] < len(t_us) else t_us[0]
            t1_us = t_us[frame_event_end_idx[frame_idx]] if frame_event_end_idx[frame_idx] < len(t_us) else t_us[-1]
            idx_info = f"idx [{frame_event_start_idx[frame_idx]},{frame_event_end_idx[frame_idx]})"
        else:
            raise RuntimeError("No valid event window keys found in flow.h5")

        evt_vis = build_binary_events_image(t_us, y, x, p, t0_us, t1_us, h, w)

        flow = forward_flow[frame_idx]
        flow_vis = flow_to_middlebury_rgb(flow)

        # Compute max flow magnitude
        flow_magnitude = np.linalg.norm(flow, axis=-1)
        max_flow = np.max(flow_magnitude)

        ax_evt.imshow(evt_vis, cmap="gray", vmin=0, vmax=1)
        ax_evt.set_title(
            f"Binary Events | frame {frame_idx}/{n-1} | {idx_info}"
        )
        ax_evt.axis("off")

        if max_flow < 1e-3:
            ax_flow.set_title("Forward Flow (not shown: max flow near zero)")
            ax_flow.axis("off")
            ax_flow.text(
                0.5, 0.5,
                "Max flow is near zero\n(no flow displayed)",
                color="white", fontsize=14, fontweight="bold",
                ha="center", va="center", transform=ax_flow.transAxes,
                bbox=dict(facecolor="black", alpha=0.7, boxstyle="round,pad=0.4")
            )
        else:
            ax_flow.imshow(flow_vis)
            ax_flow.set_title("Forward Flow")
            ax_flow.axis("off")
            # Annotate max flow
            ax_flow.text(
                0.02, 0.96,
                f"Max flow: {max_flow:.2f}",
                color="white", fontsize=12, fontweight="bold",
                ha="left", va="top", transform=ax_flow.transAxes,
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2")
            )

    draw(0)

    ax_slider = plt.axes([0.20, 0.08, 0.60, 0.03])
    slider = Slider(ax_slider, "Frame", 0, n - 1, valinit=0, valstep=1)

    def on_slider(_):
        draw(int(slider.val))
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "left":
            slider.set_val(max(0, int(slider.val) - 1))
        elif event.key == "right":
            slider.set_val(min(n - 1, int(slider.val) + 1))
        elif event.key == "home":
            slider.set_val(0)
        elif event.key == "end":
            slider.set_val(n - 1)

    slider.on_changed(on_slider)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
