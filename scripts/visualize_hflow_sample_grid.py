import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Add repo root so src imports resolve when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.flow_viz import flow_to_image


def load_rgb_files(rgb_dir: Path):
    return sorted(rgb_dir.glob("*.png"))


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32, copy=False)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def load_events(events_h5: Path):
    with h5py.File(events_h5, "r") as f:
        t = f["events/t"][:]
        x = f["events/x"][:]
        y = f["events/y"][:]
        p = f["events/p"][:]
    return t, x, y, p


def load_flow(flow_h5: Path):
    with h5py.File(flow_h5, "r") as f:
        flow = f["flow"][:]
    return flow


def events_to_rgb(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t0: float,
    t1: float,
    height: int,
    width: int,
) -> np.ndarray:
    start_idx = np.searchsorted(t, t0, side="left")
    end_idx = np.searchsorted(t, t1, side="left")

    xs = x[start_idx:end_idx].astype(np.int64, copy=False)
    ys = y[start_idx:end_idx].astype(np.int64, copy=False)
    ps = p[start_idx:end_idx]

    img = np.zeros((height, width, 3), dtype=np.float32)
    if xs.size == 0:
        return img

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if not np.any(valid):
        return img

    xs = xs[valid]
    ys = ys[valid]
    ps = ps[valid]

    # Explicit polarity convention: positive -> red, negative -> blue.
    # Works for common encodings {-1, +1} and {0, 1}.
    pos = ps > 0
    neg = ~pos

    pos_count = np.zeros((height, width), dtype=np.int32)
    neg_count = np.zeros((height, width), dtype=np.int32)

    if np.any(pos):
        np.add.at(pos_count, (ys[pos], xs[pos]), 1)
    if np.any(neg):
        np.add.at(neg_count, (ys[neg], xs[neg]), 1)

    # Per-pixel hard assignment: red if positive dominates (or ties), blue otherwise.
    red_mask = pos_count >= neg_count
    blue_mask = neg_count > pos_count
    any_event = (pos_count + neg_count) > 0

    img[:, :, 0] = (red_mask & any_event).astype(np.float32)
    img[:, :, 2] = (blue_mask & any_event).astype(np.float32)

    return img


def flow_to_rgb(flow_frame: np.ndarray, mask_invalid: bool = True) -> np.ndarray:
    if flow_frame.ndim != 3 or flow_frame.shape[2] < 2:
        raise ValueError(f"Unexpected flow frame shape: {flow_frame.shape}")

    uv = flow_frame[:, :, :2]
    # Uses Middlebury color wheel mapping (via src.flow_viz.flow_to_image).
    flow_rgb = flow_to_image(uv).astype(np.float32) / 255.0

    if mask_invalid and flow_frame.shape[2] >= 3:
        valid_mask = flow_frame[:, :, 2] > 0.5
        flow_rgb[~valid_mask] = 0.0

    return flow_rgb


def main():
    parser = argparse.ArgumentParser(
        description="Visualize one hflow320 sample as a 3xN RGB/events/flow grid"
    )
    parser.add_argument(
        "sample_dir",
        type=str,
        help="Path to one sequence directory (contains rgb_reference, events_left, forward_flow)",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start frame index")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Alias for --start-index (first plotted frame index)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="RGB frame rate used for event alignment (default: 30)",
    )
    parser.add_argument("--save", type=str, default=None, help="Optional output image path")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for save")
    parser.add_argument(
        "--mask-invalid",
        dest="mask_invalid",
        action="store_true",
        help="Black out invalid flow pixels using the validity channel",
    )
    parser.add_argument(
        "--no-mask-invalid",
        dest="mask_invalid",
        action="store_false",
        help="Do not black out invalid flow pixels",
    )
    parser.set_defaults(mask_invalid=True)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window",
    )
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    rgb_dir = sample_dir / "rgb_reference"
    events_h5 = sample_dir / "events_left" / "events.h5"
    flow_h5 = sample_dir / "forward_flow" / "flow_gt.h5"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"Missing directory: {rgb_dir}")
    if not events_h5.exists():
        raise FileNotFoundError(f"Missing file: {events_h5}")
    if not flow_h5.exists():
        raise FileNotFoundError(f"Missing file: {flow_h5}")

    rgb_files = load_rgb_files(rgb_dir)
    if len(rgb_files) < 2:
        raise RuntimeError("Need at least 2 RGB frames to form event intervals")

    flow = load_flow(flow_h5)
    num_flow = flow.shape[0]

    example_rgb = normalize_rgb(plt.imread(rgb_files[0]))
    height, width = int(example_rgb.shape[0]), int(example_rgb.shape[1])

    t, x, y, p = load_events(events_h5)
    if t.size == 0:
        raise RuntimeError("No events found in events.h5")

    max_usable = min(len(rgb_files) - 1, num_flow)
    if max_usable <= 0:
        raise RuntimeError("No aligned RGB/event/flow frames available")

    start = args.index if args.index is not None else args.start_index
    idx = start

    if start < 0:
        raise ValueError("--start-index must be >= 0")
    if idx >= max_usable:
        raise ValueError(
            f"Requested index {idx} exceeds usable frames (0..{max_usable - 1})"
        )

    dt_us = 1e6 / args.fps
    t_start = float(t[0])

    fig, axes = plt.subplots(1, 3, figsize=(6, 4), squeeze=False)

    rgb = normalize_rgb(plt.imread(rgb_files[idx]))

    t0 = t_start + idx * dt_us
    t1 = t_start + (idx + 0.5) * dt_us
    event_rgb = events_to_rgb(t, x, y, p, t0, t1, height, width)

    flow_rgb = flow_to_rgb(flow[idx], mask_invalid=args.mask_invalid)

    axes[0, 0].imshow(rgb)
    axes[0, 1].imshow(event_rgb)
    axes[0, 2].imshow(flow_rgb)

    for c in range(3):
        axes[0, c].axis("off")

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)

    if args.save is not None:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0)
        print(f"Saved: {out_path}")

    fig.savefig("hflow_sample_grid.pdf", bbox_inches="tight", pad_inches=0)

    if not args.no_show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
