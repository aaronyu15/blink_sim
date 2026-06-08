#!/usr/bin/env python3
"""Compute average event rate and average event-masked flow magnitude for a dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate average event rate (Mev/s) and average masked flow magnitude "
            "for blink_sim dataset directories."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to split directory (e.g. ../outputs/hflow320/train)",
    )
    parser.add_argument(
        "--frame-fps",
        type=float,
        default=30.0,
        help="Frame rate used to cap inter-frame window estimation (default: 30)",
    )
    parser.add_argument(
        "--event-window-seconds",
        type=float,
        default=1.0 / 30.0,
        help="Time window used to build event masks per flow frame (default: 1/30 s)",
    )
    return parser.parse_args()


def load_event_rate_stats(event_h5_path: Path) -> tuple[float | None, int, float]:
    """Return per-sample event rate (events/s), event count, and duration in seconds."""
    with h5py.File(event_h5_path, "r") as f:
        if not all(key in f for key in ("events/x", "events/t")):
            return None, 0, 0.0

        t = f["events/t"][:]
        n_events = int(t.size)
        if n_events == 0:
            return None, 0, 0.0

        duration_us = float(t[-1] - t[0])
        if duration_us <= 0:
            return None, n_events, 0.0

        duration_s = duration_us * 1e-6
        rate_eps = n_events / duration_s
        return rate_eps, n_events, duration_s


def accumulate_masked_flow_stats(
    flow_h5_path: Path,
    event_h5_path: Path,
    frame_fps: float,
    event_window_seconds: float,
) -> tuple[float, float, int, int]:
    """
    Return masked magnitude sum, squared-sum, pixel count, and processed frame count.

    Masking follows the same logic as analyze_dataset_stats.py:
    - event window is the last event_window_seconds before each frame boundary
    - combined mask is (flow valid mask, if present) AND (pixels with >=1 event in window)
    """
    mag_sum = 0.0
    mag_sq_sum = 0.0
    mag_count = 0
    processed_frames = 0
    event_window_us = event_window_seconds * 1e6

    with h5py.File(flow_h5_path, "r") as flow_f, h5py.File(event_h5_path, "r") as event_f:
        if "flow" not in flow_f:
            return mag_sum, mag_sq_sum, mag_count, processed_frames
        if not all(key in event_f for key in ("events/x", "events/y", "events/t")):
            return mag_sum, mag_sq_sum, mag_count, processed_frames

        flow_ds = flow_f["flow/forward"]
        valid_ds = flow_f["flow/valid"]
        event_start = flow_f["flow/event_start"]
        event_end = flow_f["flow/event_end"]
        x = event_f["events/x"][:]
        y = event_f["events/y"][:]
        t = event_f["events/t"][:]

        if t.size == 0 or flow_ds.shape[0] == 0:
            return mag_sum, mag_sq_sum, mag_count, processed_frames

        num_frames = int(flow_ds.shape[0])

        for frame_idx in range(num_frames):
            flow_frame = flow_ds[frame_idx]
            valid_mask = valid_ds[frame_idx]
            if flow_frame.ndim != 3 or flow_frame.shape[2] not in (2, 3):
                continue

            flow_u = flow_frame[:, :, 0]
            flow_v = flow_frame[:, :, 1]
            h, w = flow_u.shape

            valid_mask = valid_mask[:, :, 0] > 0.5

            t1 = event_end[frame_idx]
            t0 = event_start[frame_idx]
            start_idx = np.searchsorted(t, t0, side="left")
            end_idx = np.searchsorted(t, t1, side="left")

            if end_idx <= start_idx:
                continue

            event_mask = np.zeros((h, w), dtype=bool)
            x_slice = x[start_idx:end_idx]
            y_slice = y[start_idx:end_idx]
            valid_coords = (x_slice >= 0) & (x_slice < w) & (y_slice >= 0) & (y_slice < h)
            if not np.any(valid_coords):
                continue

            event_mask[y_slice[valid_coords].astype(int), x_slice[valid_coords].astype(int)] = True
            magnitude_mask = np.sqrt(flow_u**2 + flow_v**2) > 0.01
            combined_mask = valid_mask & event_mask & magnitude_mask
            if not np.any(combined_mask):
                continue

            magnitude = np.sqrt(flow_u[combined_mask] ** 2 + flow_v[combined_mask] ** 2)
            mag_sum += float(np.sum(magnitude))
            mag_sq_sum += float(np.sum(magnitude ** 2))
            mag_count += int(magnitude.size)
            processed_frames += 1

    return mag_sum, mag_sq_sum, mag_count, processed_frames


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path

    if not dataset_path.exists() or not dataset_path.is_dir():
        raise SystemExit(f"Dataset path not found or not a directory: {dataset_path}")

    sample_dirs = sorted([p for p in dataset_path.iterdir() if p.is_dir()])

    event_rates_eps = []
    total_events = 0
    total_duration_s = 0.0

    total_masked_mag_sum = 0.0
    total_masked_mag_sq_sum = 0.0
    total_masked_mag_count = 0
    total_flow_frames_used = 0
    per_frame_masked_means = []

    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        flow_h5 = sample_dir / "flow.h5"
        event_h5 = sample_dir / "events.h5"

        if not event_h5.exists():
            continue

        rate_eps, n_events, duration_s = load_event_rate_stats(event_h5)
        if rate_eps is not None:
            event_rates_eps.append(rate_eps)
        total_events += n_events
        total_duration_s += duration_s

        if not flow_h5.exists():
            continue

        mag_sum, mag_sq_sum, mag_count, used_frames = accumulate_masked_flow_stats(
            flow_h5,
            event_h5,
            frame_fps=args.frame_fps,
            event_window_seconds=args.event_window_seconds,
        )
        total_masked_mag_sum += mag_sum
        total_masked_mag_sq_sum += mag_sq_sum
        total_masked_mag_count += mag_count
        total_flow_frames_used += used_frames
        if mag_count > 0:
            per_frame_masked_means.append(mag_sum / mag_count)

    mean_event_rate_mevs = (
        (float(np.mean(event_rates_eps)) / 1e6) if event_rates_eps else float("nan")
    )
    global_event_rate_mevs = (
        (total_events / total_duration_s / 1e6) if total_duration_s > 0 else float("nan")
    )
    mean_masked_flow_mag = (
        (total_masked_mag_sum / total_masked_mag_count)
        if total_masked_mag_count > 0
        else float("nan")
    )
    mean_masked_flow_mag_per_sequence = (
        float(np.mean(per_frame_masked_means)) if per_frame_masked_means else float("nan")
    )
    std_masked_flow_mag = float("nan")
    if total_masked_mag_count > 0:
        variance = (total_masked_mag_sq_sum / total_masked_mag_count) - (mean_masked_flow_mag ** 2)
        std_masked_flow_mag = float(np.sqrt(max(variance, 0.0)))

    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Dataset path: {dataset_path}")
    print(f"Sample directories scanned: {len(sample_dirs)}")
    print()
    print("Event rate")
    print(f"  Average per-sample event rate: {mean_event_rate_mevs:.6f} Mev/s")
    print(f"  Global event rate (total events / total duration): {global_event_rate_mevs:.6f} Mev/s")
    print(f"  Samples with valid event rate: {len(event_rates_eps)}")
    print()
    print("Masked flow magnitude")
    print(f"  Average masked flow magnitude: {mean_masked_flow_mag:.6f} px/frame")
    print(f"  Average masked flow magnitude (per-sequence mean): {mean_masked_flow_mag_per_sequence:.6f} px/frame")
    print(f"  Std masked flow magnitude: {std_masked_flow_mag:.6f} px/frame")
    print(f"  Total masked pixels used: {total_masked_mag_count:,}")
    print(f"  Flow frames contributing: {total_flow_frames_used:,}")


if __name__ == "__main__":
    main()
