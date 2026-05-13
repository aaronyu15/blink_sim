#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py


def get_flow_frame_count(flow_h5_path: Path) -> int:
    with h5py.File(flow_h5_path, "r") as handle:
        if "flow" not in handle:
            raise KeyError(f"Missing 'flow' dataset in {flow_h5_path}")
        return int(handle["flow"].shape[0])


def get_event_duration_seconds(events_h5_path: Path) -> float:
    with h5py.File(events_h5_path, "r") as handle:
        if "events/t" not in handle:
            raise KeyError(f"Missing 'events/t' dataset in {events_h5_path}")
        timestamps = handle["events/t"]
        if len(timestamps) == 0:
            return 0.0
        return float(timestamps[-1] - timestamps[0]) / 1e6


def summarize_dataset(dataset_dir: Path, fps: float) -> None:
    sample_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())

    total_samples = 0
    total_flow_frames = 0
    total_duration_seconds = 0.0

    for sample_dir in sample_dirs:
        flow_h5_path = sample_dir / "forward_flow" / "flow_gt.h5"
        events_h5_path = sample_dir / "events_left" / "events.h5"
        rgb_dir = sample_dir / "rgb_reference"

        if not flow_h5_path.exists():
            continue

        flow_frames = get_flow_frame_count(flow_h5_path)

        if events_h5_path.exists():
            duration_seconds = get_event_duration_seconds(events_h5_path)
        elif rgb_dir.exists():
            rgb_frames = len(list(rgb_dir.glob("*.png")))
            duration_seconds = max(0.0, (rgb_frames - 1) / fps) if rgb_frames > 0 else 0.0
        else:
            duration_seconds = flow_frames / fps

        total_samples += 1
        total_flow_frames += flow_frames
        total_duration_seconds += duration_seconds

        print(
            f"{sample_dir.name}: flow_frames={flow_frames}, duration_s={duration_seconds:.3f}"
        )

    print()
    print(f"samples={total_samples}")
    print(f"total_flow_frames={total_flow_frames}")
    print(f"total_duration_seconds={total_duration_seconds:.3f}")
    print(f"total_duration_minutes={total_duration_seconds / 60.0:.3f}")
    print(f"total_duration_hours={total_duration_seconds / 3600.0:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize total flow frames and duration for a sequence-format dataset"
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Directory containing sample folders, e.g. outputs/hflow320/train",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Fallback FPS for duration estimation when events.h5 is missing",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Directory not found: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {dataset_dir}")

    summarize_dataset(dataset_dir, fps=args.fps)


if __name__ == "__main__":
    main()