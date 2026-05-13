# BlinkFlow Dataset Usage Guide

The **BlinkFlow** dataset (variant: `hflow320`) is a synthetically generated dataset containing event camera data, optical flow ground truth, and RGB reference frames for human motion sequences. This guide explains how to load and work with the dataset.

## Dataset Overview

### What's Included

- **Event Camera Data**: Asynchronous event streams simulating DVS-like sensors (polarity, timestamp, x/y coordinates)
- **Optical Flow Ground Truth**: Dense per-frame forward optical flow (u, v components)
- **RGB Reference Frames**: High-resolution reference images for context and validation
- **Multiple Splits**: Train/validation/test splits with 720/180/180 animation samples respectively

### Key Statistics

- **Resolution**: 320×320 pixels
- **Motion Duration**: ≤2.0 seconds per sample
- **Optical Flow Frames**: 59 RGB reference frames → 58 optical flow frames per sample (last frame dropped)
- **Event Count**: ~45K–150K events per sample (varies by motion speed)
- **Actors**: 6 unique actors (boy1–3, girl1–3)
- **Actions**: 120 distinct motion captures from CMU mocap dataset

## Directory Structure

```
outputs/hflow320/
├── train/                    # 720 animation samples
├── valid/                    # 180 animation samples
└── test/                     # 180 animation samples

Each animation sample contains:
{actor}_{action}_{trial}/
├── events_left/
│   └── events.h5             # Event stream (polarity, timestamp, x, y)
├── forward_flow/
│   └── flow_gt.h5            # Optical flow (59 frames → 58 flow maps)
├── rgb_reference/
│   ├── 0000.png              # Frame 0
│   ├── 0001.png              # Frame 1
│   └── ... (59 total)
└── rgb_event_input/          # Internal; not typically used
    ├── 0000.png
    └── ... (59 total)
```

## Loading the Dataset

### Quick Start: Loading Events and Optical Flow

```python
import h5py
import numpy as np
from pathlib import Path

# Path to a sample
sample_dir = Path("outputs/hflow320/train/boy1_BaseballHit_0")

# Load events
with h5py.File(sample_dir / "events_left" / "events.h5", "r") as f:
    events_data = f["events"][:]
    # Shape: (num_events, 4)
    # Columns: [x, y, polarity, timestamp]
    
    x = events_data[:, 0]          # Pixel x-coordinate (0-319)
    y = events_data[:, 1]          # Pixel y-coordinate (0-319)
    polarity = events_data[:, 2]   # Event polarity (0 or 1)
    timestamp = events_data[:, 3]  # Timestamp in seconds

# Load optical flow
with h5py.File(sample_dir / "forward_flow" / "flow_gt.h5", "r") as f:
    flow = f["flow"][:]
    # Shape: (58, 320, 320, 2)
    # Channels: [u_component, v_component]
    
    flow_frame_0 = flow[0]        # First flow frame
    u_component = flow[0, :, :, 0]
    v_component = flow[0, :, :, 1]
```

### Python Class: DatasetLoader

For efficient batch processing, use a custom loader:

```python
class BlinkFlowDataset:
    def __init__(self, split="train", root_dir="outputs/hflow320"):
        """
        Args:
            split: one of "train", "valid", "test"
            root_dir: path to hflow320 directory
        """
        self.split_dir = Path(root_dir) / split
        self.samples = sorted([d for d in self.split_dir.iterdir() if d.is_dir()])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        
        # Load events
        with h5py.File(sample_dir / "events_left" / "events.h5", "r") as f:
            events = f["events"][:]  # (N, 4): [x, y, polarity, timestamp]
        
        # Load flow
        with h5py.File(sample_dir / "forward_flow" / "flow_gt.h5", "r") as f:
            flow = f["flow"][:]       # (58, 320, 320, 2)
        
        # Load reference frames (optional)
        rgb_dir = sample_dir / "rgb_reference"
        rgb_paths = sorted(rgb_dir.glob("*.png"))
        
        return {
            "name": sample_dir.name,
            "events": events,         # Event stream
            "flow": flow,             # Optical flow ground truth
            "rgb_paths": rgb_paths,   # Paths to reference images
        }

# Usage
dataset = BlinkFlowDataset(split="train")
sample = dataset[0]
print(f"Sample: {sample['name']}")
print(f"Events shape: {sample['events'].shape}")
print(f"Flow shape: {sample['flow'].shape}")
```

## Working with Events

### Filtering Events by Time Window

```python
# Get events between t1 and t2 seconds
t1, t2 = 0.5, 1.0
mask = (events[:, 3] >= t1) & (events[:, 3] < t2)
events_window = events[mask]
```

### Creating Event Frames

```python
def create_event_frame(events, img_height=320, img_width=320, polarity_channels=True):
    """
    Build a 2D or 3D representation of events.
    
    Args:
        events: (N, 4) array [x, y, polarity, timestamp]
        polarity_channels: if True, returns (H, W, 2); if False, returns (H, W)
    
    Returns:
        event_frame: (H, W, 2) or (H, W) array
    """
    if polarity_channels:
        event_frame = np.zeros((img_height, img_width, 2), dtype=np.float32)
    else:
        event_frame = np.zeros((img_height, img_width), dtype=np.float32)
    
    x_int = events[:, 0].astype(np.int32)
    y_int = events[:, 1].astype(np.int32)
    pol = events[:, 2].astype(np.int32)
    
    if polarity_channels:
        event_frame[y_int, x_int, pol] += 1
    else:
        event_frame[y_int, x_int] += 1
    
    return event_frame

# Usage
event_frame = create_event_frame(events)
print(f"Event frame shape: {event_frame.shape}")  # (320, 320, 2)
```

### Computing Event Statistics

```python
# Total events
num_events = len(events)

# Polarity distribution
n_positive = np.sum(events[:, 2] == 1)
n_negative = np.sum(events[:, 2] == 0)
polarity_ratio = n_positive / n_negative

# Temporal extent
time_min = events[:, 3].min()
time_max = events[:, 3].max()
duration = time_max - time_min

# Event rate (events/sec)
event_rate = num_events / duration

print(f"Total events: {num_events:,}")
print(f"Positive: {n_positive:,} ({100*n_positive/num_events:.1f}%)")
print(f"Negative: {n_negative:,} ({100*n_negative/num_events:.1f}%)")
print(f"Duration: {duration:.2f}s")
print(f"Event rate: {event_rate:.0f} events/sec")
```

## Working with Optical Flow

### Optical Flow Properties

```python
# Load flow
with h5py.File(sample_dir / "forward_flow" / "flow_gt.h5", "r") as f:
    flow = f["flow"][:]  # (58, 320, 320, 2)

# Extract u and v components
u_component = flow[:, :, :, 0]
v_component = flow[:, :, :, 1]

# Compute magnitude and direction
magnitude = np.sqrt(u_component**2 + v_component**2)
direction = np.arctan2(v_component, u_component)

# Frame 0 statistics
frame_0_magnitude = magnitude[0]
mean_flow = frame_0_magnitude.mean()
max_flow = frame_0_magnitude.max()

print(f"Frame 0 - Mean flow: {mean_flow:.2f}, Max flow: {max_flow:.2f}")
```

### Per-Frame Statistics

```python
def compute_flow_statistics(flow):
    """
    Compute per-frame flow statistics.
    
    Args:
        flow: (T, H, W, 2) array
    
    Returns:
        stats: dict with per-frame metrics
    """
    u = flow[:, :, :, 0]
    v = flow[:, :, :, 1]
    mag = np.sqrt(u**2 + v**2)
    
    stats = {
        "frame_mean": mag.mean(axis=(1, 2)),      # (T,)
        "frame_max": mag.max(axis=(1, 2)),         # (T,)
        "frame_std": mag.std(axis=(1, 2)),         # (T,)
    }
    
    return stats

stats = compute_flow_statistics(flow)
print(f"Mean flow per frame: {stats['frame_mean']}")
```

### Creating Optical Flow Visualizations

```python
import matplotlib.pyplot as plt

def visualize_flow(u, v, scale=1.0):
    """
    Visualize optical flow using arrows.
    
    Args:
        u, v: (H, W) components
        scale: arrow scale factor
    """
    h, w = u.shape
    y, x = np.meshgrid(np.arange(0, h, 8), np.arange(0, w, 8), indexing='ij')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.quiver(x, y, u[::8, ::8], v[::8, ::8], scale=scale)
    ax.set_aspect('equal')
    plt.show()

# Usage
visualize_flow(flow[0, :, :, 0], flow[0, :, :, 1])
```

## Aligning Events with Optical Flow Frames

The dataset aligns events to optical flow frames as follows:

- **59 RGB reference frames** are provided at regular intervals
- **58 optical flow frames** are computed from consecutive RGB frame pairs
- **Event sequences** are temporally aligned with each optical flow frame window
- **Last frame** is intentionally dropped (optical flow requires two frames)

```python
def get_events_for_flow_frame(events, frame_idx, t_per_frame=1/30):
    """
    Get events corresponding to a specific optical flow frame.
    
    Args:
        events: (N, 4) array
        frame_idx: which flow frame (0-57)
        t_per_frame: time per frame in seconds
    
    Returns:
        events_window: events for this frame window
    """
    t_start = frame_idx * t_per_frame
    t_end = (frame_idx + 1) * t_per_frame
    
    mask = (events[:, 3] >= t_start) & (events[:, 3] < t_end)
    return events[mask]

# Usage
for flow_idx in range(58):  # 58 flow frames
    events_for_flow = get_events_for_flow_frame(events, flow_idx)
    print(f"Flow frame {flow_idx}: {len(events_for_flow)} events")
```

## Dataset Analysis

To analyze dataset statistics (event distributions, flow properties, etc.), run:

```bash
cd /home/aaron/Research_Projs/blink_sim/scripts
python analyze_dataset_stats.py \
    --dataset-path ../outputs/hflow320/train/ \
    --output-dir ../output/dataset_analysis/
```

This generates 16 individual PDF plots including:
- Flow magnitude, direction, and velocity histograms
- Event count and coordinate distributions
- Polarity balance and event rate statistics

## Important Notes

1. **Event Polarity Convention**: 0 = negative polarity, 1 = positive polarity (standard DVS convention)
2. **Coordinate System**: (0,0) is top-left; x increases right, y increases down
3. **Flow Direction**: Forward optical flow computed from frame $i$ to frame $i+1$
4. **Temporal Alignment**: Event timestamps are in seconds, normalized to [0, duration)
5. **Motion Range**: All samples have ≤2.0s duration and 59 RGB frames (≤60 flow frames)

## Common Issues and Solutions

### Missing Data
- Ensure the full dataset path is correct (e.g., `outputs/hflow320/train/`)
- Verify HDF5 files exist: `events.h5` and `flow_gt.h5`

### Memory Issues
- Process samples one at a time rather than loading entire splits
- Use time windowing to process events incrementally
- Consider downsampling events or flow for exploratory analysis

### Frame Alignment
- Always use the first 58 flow frames (indices 0-57) with events time-windowed accordingly
- The 59th RGB frame has no corresponding optical flow

## Example: Complete Workflow

```python
import h5py
import numpy as np
from pathlib import Path

# Setup
root = Path("outputs/hflow320")
sample_name = "boy1_BaseballHit_0"
sample_dir = root / "train" / sample_name

# Load data
with h5py.File(sample_dir / "events_left" / "events.h5") as f:
    events = f["events"][:]

with h5py.File(sample_dir / "forward_flow" / "flow_gt.h5") as f:
    flow = f["flow"][:]

# Process
num_flow_frames = flow.shape[0]  # 58
for frame_idx in range(num_flow_frames):
    # Get events for this frame
    t_start = frame_idx / 30.0
    t_end = (frame_idx + 1) / 30.0
    mask = (events[:, 3] >= t_start) & (events[:, 3] < t_end)
    frame_events = events[mask]
    
    # Get flow
    frame_flow = flow[frame_idx]
    
    # Example: compute flow magnitude
    flow_mag = np.sqrt(frame_flow[..., 0]**2 + frame_flow[..., 1]**2)
    
    print(f"Frame {frame_idx}: {len(frame_events)} events, "
          f"max flow: {flow_mag.max():.2f}")
```

## References

- **Simulator**: [BlinkSim GitHub](https://github.com/zju3dv/blink_sim)
- **Benchmark**: [BlinkVision](https://www.blinkvision.net/)
- **Dataset**: [BlinkFlow](https://zju3dv.github.io/blinkflow/)

---

For questions or issues, refer to the project repository or analysis scripts in `scripts/`.
