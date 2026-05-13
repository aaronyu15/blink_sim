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
            frame_event_start = f["flow/frame_event_start"][:]  # [N]
            frame_event_end = f["flow/frame_event_end"][:]      # [N]
        else:
            forward_flow = f["forward_flow"][:]            # [N, H, W, 2]
            frame_event_start = f["frame_event_start"][:]  # [N]
            frame_event_end = f["frame_event_end"][:]      # [N]

    return forward_flow, frame_event_start, frame_event_end


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


def main():
    parser = argparse.ArgumentParser(description="Visualize event frame vs flow with slider")
    parser.add_argument("sample_dir", type=str, help="Path to one sample directory")
    parser.add_argument("--event-fps", type=float, default=300.0, help="Event input FPS used in simulation")
    parser.add_argument(
        "--event-index-offset",
        type=int,
        default=None,
        help="Optional manual offset subtracted from frame_event_start/end before time conversion. "
             "If omitted, uses min(frame_event_start) automatically.",
    )
    args = parser.parse_args()

    sample_dir = args.sample_dir
    if not os.path.isdir(sample_dir):
        print(f"Error: sample directory does not exist: {sample_dir}")
        sys.exit(1)

    print("Loading event stream from events.h5 ...")
    t_us, y, x, p = load_event_stream(sample_dir)

    print("Loading flow and frame-event metadata from flow.h5 ...")
    forward_flow, frame_event_start, frame_event_end = load_flow_data(sample_dir)

    n = forward_flow.shape[0]
    h, w = forward_flow.shape[1], forward_flow.shape[2]

    print(f"RGB/flow frames: {n}")
    print(f"Event count: {len(t_us)}")
    if len(t_us) > 0:
        print(f"Event time range: {t_us[0]}us .. {t_us[-1]}us")

    # In this pipeline, frame_event_start/end may include an initial trim offset
    # (e.g., 30 event frames) while event timestamps in events.h5 start at t=0.
    # Remove that offset before converting event-frame indices to times.
    if args.event_index_offset is None:
        event_index_offset = int(np.min(frame_event_start))
    else:
        event_index_offset = int(args.event_index_offset)
    print(f"Using event index offset: {event_index_offset}")

    fig, (ax_evt, ax_flow) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.20)

    def draw(frame_idx):
        ax_evt.clear()
        ax_flow.clear()

        evt_start_idx = int(frame_event_start[frame_idx])
        evt_end_idx = int(frame_event_end[frame_idx])

        evt_start_idx_aligned = max(0, evt_start_idx - event_index_offset)
        evt_end_idx_aligned = max(0, evt_end_idx - event_index_offset)

        t0_us = event_idx_to_time_us(evt_start_idx_aligned, args.event_fps)
        t1_us = event_idx_to_time_us(evt_end_idx_aligned, args.event_fps)

        evt_vis = build_binary_events_image(t_us, y, x, p, t0_us, t1_us, h, w)

        flow_vis = flow_to_middlebury_rgb(forward_flow[frame_idx])

        ax_evt.imshow(evt_vis, cmap="gray", vmin=0, vmax=1)
        ax_evt.set_title(
            f"Binary Events | frame {frame_idx}/{n-1} | idx [{evt_start_idx_aligned},{evt_end_idx_aligned})"
        )
        ax_evt.axis("off")

        ax_flow.imshow(flow_vis)
        ax_flow.set_title("Forward Flow")
        ax_flow.axis("off")

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
