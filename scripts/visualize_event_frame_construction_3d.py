#!/usr/bin/env python3
"""Create a conceptual 3D visualization of how event frames are formed from a sequence.

The figure shows:
1) A 3D event cloud in (x, y, time)
2) Event-frame slices placed at selected time intervals used for flow-frame alignment

This helps explain the distinction between:
- Sequence sample: one directory containing a full event stream + all flow frames
- Flow-frame sample: one time interval between consecutive flow-frame boundaries
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add repo root so src imports resolve when running as a script.
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from src.flow_viz import flow_to_image


# Paper style configuration
plt.style.use("paper.mplstyle")
_PAPER_WIDTH = 433.6
_PT_TO_IN = 1.0 / 72.27
_FIG_WIDTH = _PAPER_WIDTH * _PT_TO_IN
_GOLDEN = (1 + np.sqrt(5)) / 2
_SCALE = 0.9


DEFAULT_DOT_SIZE = 1.5
DEFAULT_TIME_STRETCH = 20.0
DEFAULT_SAVE_MAX_POINTS = 1000
DEFAULT_PREVIEW_MAX_POINTS = 3000
DEFAULT_SAVE_SURFACE_STRIDE = 4
DEFAULT_PREVIEW_SURFACE_STRIDE = 8
DEFAULT_BOX_ASPECT = (5.5, 1.4, 1.0)


def build_event_image(x: np.ndarray, y: np.ndarray, p: np.ndarray, h: int, w: int) -> np.ndarray:
    """Create an RGB event image where red=positive, blue=negative."""
    pos = np.zeros((h, w), dtype=np.float32)
    neg = np.zeros((h, w), dtype=np.float32)

    if x.size > 0:
        x = x.astype(np.int32)
        y = y.astype(np.int32)
        p = p.astype(np.int32)

        pos_mask = p > 0
        neg_mask = ~pos_mask

        if np.any(pos_mask):
            np.add.at(pos, (y[pos_mask], x[pos_mask]), 1.0)
        if np.any(neg_mask):
            np.add.at(neg, (y[neg_mask], x[neg_mask]), 1.0)

    # Robust normalization keeps both sparse and dense intervals readable.
    max_count = max(float(np.percentile(pos, 99.5)), float(np.percentile(neg, 99.5)), 1.0)
    pos_n = np.clip(pos / max_count, 0.0, 1.0)
    neg_n = np.clip(neg / max_count, 0.0, 1.0)

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = pos_n
    rgb[..., 2] = neg_n
    rgb[..., 1] = 0.25 * (pos_n + neg_n)

    return rgb


def build_rgb_image(rgb_path: Path) -> np.ndarray:
    """Load and normalize an RGB image."""
    img = Image.open(rgb_path).convert("RGB")
    rgb = np.array(img, dtype=np.float32) / 255.0
    return rgb


def build_flow_image(flow: np.ndarray) -> np.ndarray:
    """Convert optical flow to RGB using Middlebury color scheme."""
    uv = flow[:, :, :2]
    flow_rgb = flow_to_image(uv).astype(np.float32) / 255.0
    return flow_rgb


def choose_consecutive_frame_indices(num_frames: int, start_frame: int, n_show: int) -> np.ndarray:
    if num_frames <= 0:
        return np.array([], dtype=np.int32)
    start = max(0, min(int(start_frame), num_frames - 1))
    n = max(1, int(n_show))
    end = min(num_frames, start + n)
    return np.arange(start, end, dtype=np.int32)


def create_visualization_figure(
    x_vis: np.ndarray,
    y_vis: np.ndarray,
    t_vis: np.ndarray,
    p_vis: np.ndarray,
    edges_us: np.ndarray,
    frame_indices: np.ndarray,
    z_ref_start: float,
    z_ref_duration_us: float,
    h: int,
    w: int,
    max_points: int,
    surface_stride: int,
    sample_name: str,
    num_flow_frames: int,
    mode: str = "events",
    rgb_dir: Path | None = None,
    flow_data: np.ndarray | None = None,
    rasterize_artists: bool = False,
) -> plt.Figure:
    n_events = t_vis.size
    if n_events > max_points:
        rng = np.random.default_rng(0)
        scatter_idx = np.sort(rng.choice(n_events, size=max_points, replace=False))
    else:
        scatter_idx = np.arange(n_events)

    xs = x_vis[scatter_idx]
    ys = y_vis[scatter_idx]
    ts = ((t_vis[scatter_idx] - z_ref_start) * 1e-6) * DEFAULT_TIME_STRETCH
    ps = p_vis[scatter_idx]

    fig_width = _FIG_WIDTH * _SCALE
    fig_height = fig_width / _GOLDEN
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111, projection="3d")

    # Only show event scatter for event mode
    if mode == "events":
        colors = np.where(ps > 0, "#d62728", "#1f77b4")
        ax.scatter(
            ts,
            xs,
            ys,
            c=colors,
            s=DEFAULT_DOT_SIZE,
            alpha=0.4,
            linewidths=0,
            rasterized=rasterize_artists,
        )

    stride = max(1, int(surface_stride))

    for fi in frame_indices:
        ta = edges_us[fi]
        tb = edges_us[fi + 1]

        time_plane_val = ((ta - z_ref_start) * 1e-6) * DEFAULT_TIME_STRETCH
        y_coords = np.arange(0, w, stride)
        z_coords = np.arange(0, h, stride)
        Y_plane, Z_plane = np.meshgrid(y_coords, z_coords)
        X_plane = np.full_like(Y_plane, time_plane_val, dtype=np.float64)

        if mode == "events":
            start = np.searchsorted(t_vis, ta, side="left")
            end = np.searchsorted(t_vis, tb, side="left")
            x_slice = x_vis[start:end]
            y_slice = y_vis[start:end]
            p_slice = p_vis[start:end]
            frame_rgb = build_event_image(x_slice, y_slice, p_slice, h=h, w=w)
        elif mode == "rgb":
            # Load RGB frame by index
            rgb_files = sorted(rgb_dir.glob("*.png")) if rgb_dir else []
            if fi < len(rgb_files):
                frame_rgb = build_rgb_image(rgb_files[fi])
            else:
                frame_rgb = np.zeros((h, w, 3), dtype=np.float32)
        elif mode == "flow":
            # Use flow data directly
            if fi < flow_data.shape[0]:
                frame_rgb = build_flow_image(flow_data[fi])
            else:
                frame_rgb = np.zeros((h, w, 3), dtype=np.float32)
        else:
            frame_rgb = np.zeros((h, w, 3), dtype=np.float32)

        frame_rgb_ds = frame_rgb[::stride, ::stride, :]

        surf = ax.plot_surface(
            X_plane,
            Y_plane,
            Z_plane,
            rstride=1,
            cstride=1,
            facecolors=np.clip(frame_rgb_ds, 0.0, 1.0),
            shade=False,
            linewidth=0,
            antialiased=False,
            alpha=0.95,
        )
        surf.set_rasterized(rasterize_artists)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_title("")
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticks([])

    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        try:
            axis.pane.fill = False
        except Exception:
            pass
    
    # Hide Y and Z axes completely
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)

    ax.set_xlim(0, (z_ref_duration_us * 1e-6) * DEFAULT_TIME_STRETCH)
    ax.set_ylim(0, w - 1)
    ax.set_zlim(h - 1, 0)
    ax.set_box_aspect(DEFAULT_BOX_ASPECT)
    elev = 20
    azim = -65
    ax.view_init(elev=elev, azim=azim)
    
    # Add camera position text annotation with dynamic update
    camera_text = fig.text(
        0.02, 0.98,
        f"Camera: elev={elev}°, azim={azim}°",
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )
    
    # Callback to update camera position text when view changes
    def update_camera_text(event):
        try:
            elev_current = ax.elev
            azim_current = ax.azim
            camera_text.set_text(f"Camera: elev={elev_current:.0f}°, azim={azim_current:.0f}°")
        except Exception:
            pass
    
    fig.canvas.mpl_connect("motion_notify_event", update_camera_text)
    
    fig.tight_layout(pad=0.02)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize how event frames arise from event streams.")
    parser.add_argument(
        "--sample-dir",
        type=str,
        required=True,
        help="Path to one sample directory, e.g. outputs/hflow320/train/girl1_RunningToTurn_1",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["events", "rgb", "flow"],
        default="events",
        help="Which frame type to visualize: events, rgb, or flow",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/visualizations/event_frame_construction_3d.pdf",
        help="Output figure path (.pdf or .png)",
    )
    parser.add_argument(
        "--frames-to-show",
        type=int,
        default=7,
        help="How many consecutive flow-frame intervals to visualize as event-frame slices",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start index for consecutive flow-frame intervals",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip interactive preview window before saving",
    )
    parser.add_argument(
        "--rasterize-pdf",
        dest="rasterize_pdf",
        action="store_true",
        help="Rasterize 3D frame planes/scatter when saving PDF so frames are embedded as images",
    )
    parser.add_argument(
        "--no-rasterize-pdf",
        dest="rasterize_pdf",
        action="store_false",
        help="Keep fully vector PDF output",
    )
    parser.set_defaults(rasterize_pdf=True)
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    event_h5 = sample_dir  / "events.h5"
    flow_h5 = sample_dir  / "flow.h5"
    rgb_dir = sample_dir / "rgb_reference"

    if not event_h5.exists():
        raise FileNotFoundError(f"Missing event file: {event_h5}")
    if not flow_h5.exists():
        raise FileNotFoundError(f"Missing flow file: {flow_h5}")
    if args.mode == "rgb" and not rgb_dir.exists():
        raise FileNotFoundError(f"Missing RGB directory: {rgb_dir}")

    with h5py.File(event_h5, "r") as ef, h5py.File(flow_h5, "r") as ff:
        for key in ["events/x", "events/y", "events/t", "events/p"]:
            if key not in ef:
                raise KeyError(f"Expected key '{key}' in {event_h5}")
        if "flow/forward" not in ff:
            raise KeyError(f"Expected key 'flow/forward' in {flow_h5}")

        x = ef["events/x"][:]
        y = ef["events/y"][:]
        t = ef["events/t"][:].astype(np.float64)
        p = ef["events/p"][:]
        num_flow_frames = int(ff["flow/forward"].shape[0])
        h = int(ff["flow/forward"].shape[1])
        w = int(ff["flow/forward"].shape[2])
        
        # Load flow data if needed
        flow_data = None
        if args.mode == "flow":
            flow_data = ff["flow/forward"][:]

    if t.size == 0:
        raise ValueError("No events found in sample.")
    if num_flow_frames <= 0:
        raise ValueError("No flow frames found in sample.")

    t0 = float(t[0])
    t1 = float(t[-1])
    duration_us = max(1.0, t1 - t0)

    # Flow-frame interval boundaries across the sequence timeline.
    edges_us = np.linspace(t0, t1, num_flow_frames + 1)
    frame_indices = choose_consecutive_frame_indices(num_flow_frames, args.start_frame, args.frames_to_show)

    if frame_indices.size == 0:
        raise ValueError("No frame indices selected. Check --start-frame / --frames-to-show.")

    z_ref_start = float(edges_us[int(frame_indices[0])])
    z_ref_end = float(edges_us[int(frame_indices[-1]) + 1])

    z_ref_duration_us = max(1.0, z_ref_end - z_ref_start)

    in_window = (t >= z_ref_start) & (t <= z_ref_end)
    t_vis = t[in_window]
    x_vis = x[in_window]
    y_vis = y[in_window]
    p_vis = p[in_window]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rasterize_for_output = args.rasterize_pdf and out.suffix.lower() == ".pdf"

    # Show interactive preview before writing to disk unless explicitly skipped.
    if not args.no_preview:
        preview_fig = create_visualization_figure(
            x_vis,
            y_vis,
            t_vis,
            p_vis,
            edges_us,
            frame_indices,
            z_ref_start,
            z_ref_duration_us,
            h,
            w,
            min(DEFAULT_PREVIEW_MAX_POINTS, DEFAULT_SAVE_MAX_POINTS),
            DEFAULT_PREVIEW_SURFACE_STRIDE,
            sample_dir.name,
            num_flow_frames,
            mode=args.mode,
            rgb_dir=rgb_dir if args.mode == "rgb" else None,
            flow_data=flow_data,
            rasterize_artists=False,
        )
        plt.show()
        plt.close(preview_fig)

    fig = create_visualization_figure(
        x_vis,
        y_vis,
        t_vis,
        p_vis,
        edges_us,
        frame_indices,
        z_ref_start,
        z_ref_duration_us,
        h,
        w,
        DEFAULT_SAVE_MAX_POINTS,
        DEFAULT_SAVE_SURFACE_STRIDE,
        sample_dir.name,
        num_flow_frames,
        mode=args.mode,
        rgb_dir=rgb_dir if args.mode == "rgb" else None,
        flow_data=flow_data,
        rasterize_artists=rasterize_for_output,
    )
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"Saved: {out}")
    print(f"Sample: {sample_dir.name}")
    print(f"Mode: {args.mode}")
    print(f"Events shown in selected window: {t_vis.size:,}, Flow frames in sequence: {num_flow_frames}")
    print(f"Defaults: dot_size={DEFAULT_DOT_SIZE}, time_stretch={DEFAULT_TIME_STRETCH}, save_max_points={DEFAULT_SAVE_MAX_POINTS}")


if __name__ == "__main__":
    main()
