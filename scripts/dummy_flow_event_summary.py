#!/usr/bin/env python3
"""Dummy utility to recursively summarize flow frames and event duration.

The script scans a root path recursively, finds directories containing both
`flow.h5` and `events.h5`, and computes:
- number of optical flow frames (from flow.h5)
- event stream duration in seconds (from events.h5, using events/t)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def get_flow_frame_count(flow_h5_path: Path) -> int:
    with h5py.File(flow_h5_path, "r") as handle:
        if "flow/forward" in handle:
            return int(handle["flow/forward"].shape[0])
        if "forward_flow" in handle:
            return int(handle["forward_flow"].shape[0])
        if "flow" in handle:
            return int(handle["flow"].shape[0])
    raise KeyError(
        f"No supported flow dataset found in {flow_h5_path}. "
        "Expected one of: flow/forward, forward_flow, flow"
    )


def get_event_duration_seconds(events_h5_path: Path) -> float:
    with h5py.File(events_h5_path, "r") as handle:
        if "events/t" not in handle:
            raise KeyError(f"Missing 'events/t' in {events_h5_path}")
        timestamps = handle["events/t"]
        if len(timestamps) == 0:
            return 0.0
        return float(timestamps[-1] - timestamps[0]) / 1e6


def find_sample_dirs(root_dir: Path) -> list[Path]:
    sample_dirs = []
    for flow_path in root_dir.rglob("flow.h5"):
        sample_dir = flow_path.parent
        events_path = sample_dir / "events.h5"
        if events_path.exists():
            sample_dirs.append(sample_dir)
    return sorted(sample_dirs)


def summarize(root_dir: Path) -> None:
    sample_dirs = find_sample_dirs(root_dir)
    if not sample_dirs:
        print(f"No sample directories found with both flow.h5 and events.h5 under: {root_dir}")
        return

    total_samples = 0
    total_flow_frames = 0
    total_event_duration_s = 0.0
    failed_samples = 0

    for sample_dir in sample_dirs:
        flow_h5 = sample_dir / "flow.h5"
        events_h5 = sample_dir / "events.h5"

        try:
            flow_frames = get_flow_frame_count(flow_h5)
            event_duration_s = get_event_duration_seconds(events_h5)
        except Exception as exc:
            failed_samples += 1
            print(f"[WARN] {sample_dir}: {exc}")
            continue

        total_samples += 1
        total_flow_frames += flow_frames
        total_event_duration_s += event_duration_s

        print(
            f"{sample_dir}: flow_frames={flow_frames}, "
            f"event_duration_s={event_duration_s:.6f}"
        )

    print()
    print(f"processed_samples={total_samples}")
    print(f"failed_samples={failed_samples}")
    print(f"total_flow_frames={total_flow_frames}")
    print(f"total_event_duration_seconds={total_event_duration_s:.6f}")
    print(f"total_event_duration_hours={total_event_duration_s / 3600.0:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively read flow/events files and report total optical flow "
            "frames plus total event duration in seconds"
        )
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root path to scan recursively",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Path not found: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_dir}")

    summarize(root_dir)


if __name__ == "__main__":
    main()
