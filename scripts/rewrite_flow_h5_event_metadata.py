#!/usr/bin/env python3
"""
Rewrite flow.h5 files to a direct event-index/time schema with no runtime offset/FPS assumptions.

Output schema (group: flow):
  - forward            (copied)
  - valid              (copied)
  - event_start        (us, uint64)
  - event_end          (us, uint64)
  - frame_event_start  (event-stream index, uint64)
  - frame_event_end    (event-stream index, uint64)

Notes:
- frame_event_start/end are direct indices into events/t.
- event_start/end are exact timestamps at those indices:
    events_t[frame_event_start[i]] == event_start[i]
    events_t[frame_event_end[i]]   == event_end[i]
- This script writes an inclusive end index to keep frame_event_end indexable.

Typical usage:
  python scripts/rewrite_flow_h5_event_metadata.py output/train
  python scripts/rewrite_flow_h5_event_metadata.py output/train --in-place
  python scripts/rewrite_flow_h5_event_metadata.py output/train/sample_x/flow.h5 --event-fps 300
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np
import yaml


def read_first_existing(h5: h5py.File, keys: Sequence[str]):
    for key in keys:
        if key in h5:
            return h5[key][()]
    raise KeyError(f"None of keys found: {keys}")


def get_events_path(sample_dir: Path) -> Path:
    direct = sample_dir / "events.h5"
    if direct.exists():
        return direct
    legacy = sample_dir / "events_left" / "events.h5"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"Missing events file for sample: {sample_dir}")


def infer_event_fps(flow_h5_path: Path, sample_dir: Path, user_fps: Optional[float]) -> float:
    if user_fps is not None:
        return float(user_fps)

    cfg_path = sample_dir / "config_job.yaml"
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if "event_image_fps" in cfg:
                return float(cfg["event_image_fps"])
        except Exception:
            pass

    with h5py.File(flow_h5_path, "r") as hf:
        frame_t_us = None
        for key in ("flow/frame_t_us", "frame_t_us"):
            if key in hf:
                frame_t_us = hf[key][()]
                break

        frame_event_start = None
        for key in ("flow/frame_event_start", "frame_event_start"):
            if key in hf:
                frame_event_start = hf[key][()]
                break

        if frame_t_us is not None and frame_event_start is not None:
            t = np.asarray(frame_t_us, dtype=np.float64)
            idx = np.asarray(frame_event_start, dtype=np.float64)
            if t.size >= 2 and idx.size >= 2:
                dt = np.diff(t)
                di = np.diff(idx)
                mask = (dt > 0) & (di > 0)
                if np.any(mask):
                    slope = float(np.median(di[mask] / dt[mask]))
                    fps = slope * 1e6
                    if math.isfinite(fps) and fps > 0:
                        return fps

    return 300.0


def convert_boundaries_to_indices(
    events_t_us: np.ndarray,
    old_start: np.ndarray,
    old_end: np.ndarray,
    event_t_offset_us: int,
    event_fps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if events_t_us.size == 0:
        raise ValueError("events/t is empty; cannot build direct event-index metadata")

    old_start = np.asarray(old_start, dtype=np.float64)
    old_end = np.asarray(old_end, dtype=np.float64)

    t0_us = event_t_offset_us + np.rint((old_start / event_fps) * 1e6).astype(np.int64)
    t1_us = event_t_offset_us + np.rint((old_end / event_fps) * 1e6).astype(np.int64)

    start_idx = np.searchsorted(events_t_us, t0_us, side="left").astype(np.int64)
    end_excl = np.searchsorted(events_t_us, t1_us, side="left").astype(np.int64)

    n_events = int(events_t_us.shape[0])
    start_idx = np.clip(start_idx, 0, n_events - 1)

    # Keep end index directly indexable while preserving interval semantics as much as possible.
    end_idx = np.maximum(start_idx, end_excl - 1)
    end_idx = np.clip(end_idx, 0, n_events - 1)

    event_start = events_t_us[start_idx]
    event_end = events_t_us[end_idx]

    return (
        start_idx.astype(np.uint64),
        end_idx.astype(np.uint64),
        event_start.astype(np.uint64),
        event_end.astype(np.uint64),
    )


def process_flow_file(flow_h5_path: Path, in_place: bool, output_name: str, user_event_fps: Optional[float]) -> None:
    sample_dir = flow_h5_path.parent
    events_path = get_events_path(sample_dir)

    with h5py.File(events_path, "r") as ef:
        if "events/t" not in ef:
            raise KeyError(f"Missing dataset 'events/t' in {events_path}")
        events_t_us = ef["events/t"][()].astype(np.uint64)

    with h5py.File(flow_h5_path, "r") as hf:
        forward = read_first_existing(hf, ["flow/forward", "forward_flow"]) 
        valid = read_first_existing(hf, ["flow/valid", "valid"]) 
        frame_event_start_old = read_first_existing(hf, ["flow/frame_event_start", "frame_event_start"]) 
        frame_event_end_old = read_first_existing(hf, ["flow/frame_event_end", "frame_event_end"]) 

        if forward.shape[0] != frame_event_start_old.shape[0] or forward.shape[0] != frame_event_end_old.shape[0]:
            raise ValueError(
                f"Mismatched lengths in {flow_h5_path}: "
                f"forward={forward.shape[0]}, frame_event_start={frame_event_start_old.shape[0]}, "
                f"frame_event_end={frame_event_end_old.shape[0]}"
            )

        # Detect already-converted files by presence of the new schema key, not by index magnitude
        # (old event-frame indices are always small and would otherwise always pass the magnitude check).
        already_converted = "flow/event_start" in hf

        if already_converted:
            frame_event_start = np.asarray(frame_event_start_old, dtype=np.uint64)
            frame_event_end = np.asarray(frame_event_end_old, dtype=np.uint64)
            frame_event_start = np.clip(frame_event_start, 0, events_t_us.shape[0] - 1)
            frame_event_end = np.clip(frame_event_end, 0, events_t_us.shape[0] - 1)
            event_start = events_t_us[frame_event_start]
            event_end = events_t_us[frame_event_end]
        else:
            event_t_offset_us = int(read_first_existing(hf, ["flow/event_t_offset_us", "event_t_offset_us"]) if ("flow/event_t_offset_us" in hf or "event_t_offset_us" in hf) else 0)
            event_fps = infer_event_fps(flow_h5_path, sample_dir, user_event_fps)
            frame_event_start, frame_event_end, event_start, event_end = convert_boundaries_to_indices(
                events_t_us=events_t_us,
                old_start=np.asarray(frame_event_start_old),
                old_end=np.asarray(frame_event_end_old),
                event_t_offset_us=event_t_offset_us,
                event_fps=event_fps,
            )

    out_path = flow_h5_path if in_place else (sample_dir / output_name)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with h5py.File(tmp_path, "w") as out:
        out.create_dataset("flow/forward", data=forward, compression="gzip", compression_opts=4)
        out.create_dataset("flow/valid", data=valid, compression="gzip", compression_opts=4)
        out.create_dataset("flow/event_start", data=event_start, compression="gzip", compression_opts=4)
        out.create_dataset("flow/event_end", data=event_end, compression="gzip", compression_opts=4)
        out.create_dataset("flow/frame_event_start", data=frame_event_start, compression="gzip", compression_opts=4)
        out.create_dataset("flow/frame_event_end", data=frame_event_end, compression="gzip", compression_opts=4)

    tmp_path.replace(out_path)
    print(f"[ok] wrote {out_path}")


def collect_flow_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return sorted(path.rglob("flow.h5"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite flow.h5 metadata to direct event-index/time schema")
    parser.add_argument("path", type=Path, help="Sample directory, dataset root, or a specific flow.h5 path")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each flow.h5 in place. If omitted, writes sibling file (default: flow_clean.h5).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="flow_clean.h5",
        help="Output file name when not using --in-place (default: flow_clean.h5)",
    )
    parser.add_argument(
        "--event-fps",
        type=float,
        default=300,
        help="Optional event FPS for legacy frame-index conversion. If omitted, inferred from metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flow_files = collect_flow_files(args.path)
    if not flow_files:
        raise FileNotFoundError(f"No flow.h5 files found under: {args.path}")

    print(f"Found {len(flow_files)} flow file(s)")
    for flow_file in flow_files:
        try:
            process_flow_file(
                flow_h5_path=flow_file,
                in_place=args.in_place,
                output_name=args.output_name,
                user_event_fps=args.event_fps,
            )
        except Exception as exc:
            print(f"[fail] {flow_file}: {exc}")


if __name__ == "__main__":
    main()
