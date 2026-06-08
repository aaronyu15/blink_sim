#!/usr/bin/env python3
"""
Dataset Statistics Analysis Script (Memory-Efficient Version)
Analyzes flow and event data from the blink_sim training dataset.
Uses incremental processing to avoid loading all data into memory.
"""

import numpy as np
import h5py
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
from tqdm import tqdm
import argparse
import json
from collections import defaultdict

# Paper style configuration
_STYLE_PATH = Path(__file__).with_name("paper.mplstyle")
plt.style.use(str(_STYLE_PATH))
_PAPER_WIDTH = 433.62
_PT_TO_IN = 1.0 / 72.27
_FIG_WIDTH = _PAPER_WIDTH * _PT_TO_IN
_GOLDEN = (1 + np.sqrt(5)) / 2
_STACKED_SUBPLOT_HEIGHT_PT = 140.0
_STACKED_SUBPLOT_HEIGHT_IN = _STACKED_SUBPLOT_HEIGHT_PT * _PT_TO_IN
_EVENT_WINDOW_SECONDS = 1.0 / 30.0
_EVENT_WINDOW_US = _EVENT_WINDOW_SECONDS * 1e6
_DIRECTION_MAG_THRESHOLD = 0.01


class DatasetAnalyzer:
    def __init__(self, dataset_path, histogram_bins=1000, max_histogram_samples=1000000, log_file=None):
        self.dataset_path = Path(dataset_path)
        self.histogram_bins = histogram_bins
        self.max_histogram_samples = max_histogram_samples
        self.log_file = log_file
        
        # Figure size helpers
        self.fig_width = _FIG_WIDTH
        self.fig_height = self.fig_width / _GOLDEN
        
        # Running statistics for flow
        self.flow_stats = {
            'sample_count': 0,
            'total_pixels': 0,
            'mag_sum': 0.0,
            'mag_sum_sq': 0.0,
            'mag_min': float('inf'),
            'mag_max': float('-inf'),
            'u_sum': 0.0,
            'u_sum_sq': 0.0,
            'u_min': float('inf'),
            'u_max': float('-inf'),
            'v_sum': 0.0,
            'v_sum_sq': 0.0,
            'v_min': float('inf'),
            'v_max': float('-inf'),
            'per_sample_mean': [],
            'per_sample_max': [],
            'per_sample_min': [],
            'per_sample_std': [],
            # Direction statistics
            'direction_bins': np.linspace(-np.pi, np.pi, 36),  # 10 degree bins
            'direction_counts': np.zeros(35, dtype=np.int64),
            'quadrant_counts': {'right': 0, 'up': 0, 'left': 0, 'down': 0},
        }
        
        # Event statistics
        self.event_stats = {
            'total_events': 0,
            'events_per_sample': [],
            'polarity_counts': {},
            'sample_count': 0,
            'x_coords': [],
            'y_coords': [],
            'time_spans': [],
            'polarities': [],
            'positive_fraction_per_sample': [],
            'event_rate_per_sec': [],
            'events_per_flow_frame': []
        }
        
        # Online histograms - bins will be defined dynamically based on data range
        # We'll use two-pass approach: first pass gets range, second pass builds histogram
        self.histogram_data = {
            'magnitude': {'bins': None, 'counts': None, 'range': [0, 400]},  # Initial guess
            'u': {'bins': None, 'counts': None, 'range': [-250, 250]},
            'v': {'bins': None, 'counts': None, 'range': [-250, 250]}
        }
        self.histogram_initialized = False
        self.event_window_magnitude_counts = None
        self.event_window_u_counts = None
        self.event_window_v_counts = None
        
        # Track top magnitude samples
        self.top_magnitude_samples = []  # List of (max_magnitude, sample_name) tuples
        
        # Event-based valid mask statistics (separate tracking)
        self.event_mask_stats = {
            'sample_count': 0,
            'total_pixels': 0,
            'total_valid_pixels': 0,  # Pixels with events
            'mag_sum': 0.0,
            'mag_sum_sq': 0.0,
            'mag_min': float('inf'),
            'mag_max': float('-inf'),
            'u_sum': 0.0,
            'u_sum_sq': 0.0,
            'u_min': float('inf'),
            'u_max': float('-inf'),
            'v_sum': 0.0,
            'v_sum_sq': 0.0,
            'v_min': float('inf'),
            'v_max': float('-inf'),
            'per_sample_mean': [],
            'per_sample_max': [],
            'per_sample_coverage': [],  # Fraction of pixels with events
            'direction_counts': np.zeros(35, dtype=np.int64),
            'quadrant_counts': {'right': 0, 'up': 0, 'left': 0, 'down': 0},
        }

    def _extract_valid_flow(self, flow_frame):
        """Extract valid u, v, magnitude, and angles from one flow frame."""
        if len(flow_frame.shape) != 3 or flow_frame.shape[2] not in (2, 3):
            return None

        flow_u = flow_frame[:, :, 0]
        flow_v = flow_frame[:, :, 1]

        if flow_frame.shape[2] == 3:
            valid = flow_frame[:, :, 2] > 0.5
            if not valid.any():
                return None
            flow_u_valid = flow_u[valid]
            flow_v_valid = flow_v[valid]
        else:
            flow_u_valid = flow_u.reshape(-1)
            flow_v_valid = flow_v.reshape(-1)

        magnitude = np.sqrt(flow_u_valid**2 + flow_v_valid**2)
        angles = np.arctan2(flow_v_valid, flow_u_valid)
        return flow_u_valid, flow_v_valid, magnitude, angles

    def _update_flow_statistics(self, flow_u_valid, flow_v_valid, magnitude, angles, sample_name=None):
        """Update aggregate flow statistics from one frame worth of valid flow values."""
        n = len(magnitude)
        if n == 0:
            return

        # Only use nonzero flows for statistics
        nonzero_mask = magnitude > _DIRECTION_MAG_THRESHOLD
        if not np.any(nonzero_mask):
            return
        mag_nz = magnitude[nonzero_mask]
        u_nz = flow_u_valid[nonzero_mask]
        v_nz = flow_v_valid[nonzero_mask]
        n_nz = len(mag_nz)

        self.flow_stats['total_pixels'] += n_nz
        self.flow_stats['sample_count'] += 1

        self.flow_stats['mag_sum'] += np.sum(mag_nz)
        self.flow_stats['mag_sum_sq'] += np.sum(mag_nz**2)
        self.flow_stats['mag_min'] = min(self.flow_stats['mag_min'], float(np.min(mag_nz)))
        self.flow_stats['mag_max'] = max(self.flow_stats['mag_max'], float(np.max(mag_nz)))

        self.flow_stats['u_sum'] += np.sum(u_nz)
        self.flow_stats['u_sum_sq'] += np.sum(u_nz**2)
        self.flow_stats['u_min'] = min(self.flow_stats['u_min'], float(np.min(u_nz)))
        self.flow_stats['u_max'] = max(self.flow_stats['u_max'], float(np.max(u_nz)))

        self.flow_stats['v_sum'] += np.sum(v_nz)
        self.flow_stats['v_sum_sq'] += np.sum(v_nz**2)
        self.flow_stats['v_min'] = min(self.flow_stats['v_min'], float(np.min(v_nz)))
        self.flow_stats['v_max'] = max(self.flow_stats['v_max'], float(np.max(v_nz)))

        sample_max_mag = float(np.max(mag_nz))
        self.flow_stats['per_sample_mean'].append(float(np.mean(mag_nz)))
        self.flow_stats['per_sample_max'].append(sample_max_mag)
        self.flow_stats['per_sample_min'].append(float(np.min(mag_nz)))
        self.flow_stats['per_sample_std'].append(float(np.std(mag_nz)))

        if sample_name is not None:
            self.top_magnitude_samples.append((sample_max_mag, sample_name))

        # Exclude tiny flow vectors from direction stats; angle is unstable near zero.
        direction_hist, _ = np.histogram(angles[nonzero_mask], bins=self.flow_stats['direction_bins'])
        self.flow_stats['direction_counts'] += direction_hist

        self.flow_stats['quadrant_counts']['right'] += np.sum(u_nz > 0)
        self.flow_stats['quadrant_counts']['left'] += np.sum(u_nz < 0)
        self.flow_stats['quadrant_counts']['down'] += np.sum(v_nz > 0)
        self.flow_stats['quadrant_counts']['up'] += np.sum(v_nz < 0)

    def _accumulate_flow_histograms(self, flow_u_valid, flow_v_valid, magnitude):
        if not self.histogram_initialized:
            return
        self._accumulate_histogram('magnitude', magnitude)
        self._accumulate_histogram('u', flow_u_valid)
        self._accumulate_histogram('v', flow_v_valid)
    

    def analyze_flow_h5_file(self, flow_h5_path):
        """Analyze one sequence-level flow.h5 file."""
        try:
            relative_path = flow_h5_path.relative_to(self.dataset_path)
            with h5py.File(flow_h5_path, 'r') as f:
                if 'flow/forward' not in f:
                    self._log(f"Warning: Expected dataset 'flow/forward' not found in {flow_h5_path}")
                    return 0
                flow_ds = f['flow/forward']
                valid_ds = f['flow/valid'] if 'flow/valid' in f else None
                frame_count = flow_ds.shape[0]
                for frame_idx in range(frame_count):
                    frame = flow_ds[frame_idx]  # (H, W, 2)
                    if valid_ds is not None:
                        frame = np.concatenate([frame, valid_ds[frame_idx]], axis=-1)  # (H, W, 3)
                    extracted = self._extract_valid_flow(frame)
                    if extracted is None:
                        continue
                    sample_name = f"{relative_path}:{frame_idx}"
                    self._update_flow_statistics(*extracted, sample_name)
                return frame_count
        except Exception as e:
            self._log(f"Error processing {flow_h5_path}: {e}")
        return 0

    def accumulate_flow_h5_histograms(self, flow_h5_path):
        """Second-pass histogram accumulation for one sequence-level flow.h5 file."""
        try:
            with h5py.File(flow_h5_path, 'r') as f:
                if 'flow/forward' not in f:
                    return
                flow_ds = f['flow/forward']
                valid_ds = f['flow/valid'] if 'flow/valid' in f else None
                for frame_idx in range(flow_ds.shape[0]):
                    frame = flow_ds[frame_idx]  # (H, W, 2)
                    if valid_ds is not None:
                        frame = np.concatenate([frame, valid_ds[frame_idx]], axis=-1)  # (H, W, 3)
                    extracted = self._extract_valid_flow(frame)
                    if extracted is None:
                        continue
                    flow_u_valid, flow_v_valid, magnitude, _angles = extracted
                    self._accumulate_flow_histograms(flow_u_valid, flow_v_valid, magnitude)
        except Exception as e:
            self._log(f"Error building histograms from {flow_h5_path}: {e}")
    
    def analyze_event_windows(self, flow_h5_path, event_h5_path):
        """Analyze event statistics per flow-frame window using flow/event_start and flow/event_end.

        Each sample is defined as the event window associated with one ground-truth flow frame.
        Returns the number of windows processed.
        """
        num_windows = 0
        try:
            with h5py.File(flow_h5_path, 'r') as flow_f, h5py.File(event_h5_path, 'r') as event_f:
                if not all(k in event_f for k in ['events/x', 'events/y', 'events/t', 'events/p']):
                    self._log(f"Warning: Expected events/x,y,t,p not found in {event_h5_path}")
                    return 0
                if 'flow/event_start' not in flow_f or 'flow/event_end' not in flow_f:
                    self._log(f"Warning: flow/event_start or flow/event_end not found in {flow_h5_path}")
                    return 0

                t = event_f['events/t'][:]
                x = event_f['events/x'][:]
                y = event_f['events/y'][:]
                p = event_f['events/p'][:]
                event_start_arr = flow_f['flow/event_start'][:]  # absolute timestamps, us
                event_end_arr   = flow_f['flow/event_end'][:]

                for i in range(len(event_start_arr)):
                    t0 = int(event_start_arr[i])
                    t1 = int(event_end_arr[i])
                    si = np.searchsorted(t, t0, side='left')
                    ei = np.searchsorted(t, t1, side='left')

                    win_x = x[si:ei]
                    win_y = y[si:ei]
                    win_t = t[si:ei]
                    win_p = p[si:ei]
                    num_events = int(ei - si)

                    self.event_stats['total_events'] += num_events
                    self.event_stats['events_per_sample'].append(num_events)
                    self.event_stats['sample_count'] += 1

                    if len(self.event_stats['x_coords']) < self.max_histogram_samples:
                        self.event_stats['x_coords'].extend(win_x.tolist())
                        self.event_stats['y_coords'].extend(win_y.tolist())
                        self.event_stats['polarities'].extend(win_p.tolist())

                    time_span = float(t1 - t0)
                    self.event_stats['time_spans'].append(time_span)
                    if time_span > 0:
                        self.event_stats['event_rate_per_sec'].append(num_events / (time_span * 1e-6))

                    if num_events > 0:
                        unique_p, counts_p = np.unique(win_p, return_counts=True)
                        positive_events = 0
                        for pol, cnt in zip(unique_p, counts_p):
                            pol_int = int(pol)
                            self.event_stats['polarity_counts'][pol_int] = (
                                self.event_stats['polarity_counts'].get(pol_int, 0) + int(cnt)
                            )
                            if pol > 0:
                                positive_events += int(cnt)
                        self.event_stats['positive_fraction_per_sample'].append(positive_events / num_events)

                    num_windows += 1

        except Exception as e:
            self._log(f"Error processing event windows from {flow_h5_path}: {e}")
        return num_windows
    
    def analyze_flow_with_event_mask(self, flow_path, event_path_curr, event_path_prev=None):
        """
        Analyze flow using event-based valid mask
        
        For each flow frame i (representing motion from frame i to i+1),
        we create a mask from events that occurred between frame i and frame i+1.
        Only pixels with events are considered valid.
        
        Args:
            flow_path: Path to flow .npy file
            event_path_curr: Path to event .h5 file for current frame (frame i+1)
            event_path_prev: Path to event .h5 file for previous frame (frame i), optional
        """
        try:
            # Load flow
            flow = np.load(flow_path)
            
            if len(flow.shape) == 3 and flow.shape[2] == 3:
                flow_u = flow[:, :, 0]
                flow_v = flow[:, :, 1]
                valid_mask = flow[:, :, 2]
                h, w = flow_u.shape
            elif len(flow.shape) == 3 and flow.shape[2] == 2:
                flow_u = flow[:, :, 0]
                flow_v = flow[:, :, 1]
                h, w = flow_u.shape
                valid_mask = np.ones((h, w))
            else:
                self._log(f"Warning: Unexpected flow shape {flow.shape} in {flow_path}")
                return
            
            # Create event mask from event files
            event_mask = np.zeros((h, w), dtype=bool)
            
            # Load current frame events
            try:
                with h5py.File(event_path_curr, 'r') as f:
                    if 'events/x' in f and 'events/y' in f:
                        x = f['events/x'][:]
                        y = f['events/y'][:]
                        
                        # Mark pixels with events
                        valid_coords = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                        x_valid = x[valid_coords].astype(int)
                        y_valid = y[valid_coords].astype(int)
                        event_mask[y_valid, x_valid] = True
                        
            except Exception as e:
                self._log(f"Warning: Could not load events from {event_path_curr}: {e}")
                return
            
            # Optionally load previous frame events (for inter-frame events)
            if event_path_prev and os.path.exists(event_path_prev):
                try:
                    with h5py.File(event_path_prev, 'r') as f:
                        if 'events/x' in f and 'events/y' in f:
                            x = f['events/x'][:]
                            y = f['events/y'][:]
                            
                            valid_coords = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                            x_valid = x[valid_coords].astype(int)
                            y_valid = y[valid_coords].astype(int)
                            event_mask[y_valid, x_valid] = True
                except Exception as e:
                    pass  # Previous frame events are optional
            
            # Combine with existing valid mask
            combined_mask = (valid_mask > 0.5) & event_mask
            
            if not combined_mask.any():
                return
            
            # Extract valid flow
            flow_u_valid = flow_u[combined_mask]
            flow_v_valid = flow_v[combined_mask]
            magnitude = np.sqrt(flow_u_valid**2 + flow_v_valid**2)
            angles = np.arctan2(flow_v_valid, flow_u_valid)
            
            # Update statistics
            n = len(magnitude)
            self.event_mask_stats['total_pixels'] += h * w
            self.event_mask_stats['total_valid_pixels'] += n
            self.event_mask_stats['sample_count'] += 1
            
            # Running sums
            self.event_mask_stats['mag_sum'] += np.sum(magnitude)
            self.event_mask_stats['mag_sum_sq'] += np.sum(magnitude**2)
            self.event_mask_stats['mag_min'] = min(self.event_mask_stats['mag_min'], float(np.min(magnitude)))
            self.event_mask_stats['mag_max'] = max(self.event_mask_stats['mag_max'], float(np.max(magnitude)))
            
            self.event_mask_stats['u_sum'] += np.sum(flow_u_valid)
            self.event_mask_stats['u_sum_sq'] += np.sum(flow_u_valid**2)
            self.event_mask_stats['u_min'] = min(self.event_mask_stats['u_min'], float(np.min(flow_u_valid)))
            self.event_mask_stats['u_max'] = max(self.event_mask_stats['u_max'], float(np.max(flow_u_valid)))
            
            self.event_mask_stats['v_sum'] += np.sum(flow_v_valid)
            self.event_mask_stats['v_sum_sq'] += np.sum(flow_v_valid**2)
            self.event_mask_stats['v_min'] = min(self.event_mask_stats['v_min'], float(np.min(flow_v_valid)))
            self.event_mask_stats['v_max'] = max(self.event_mask_stats['v_max'], float(np.max(flow_v_valid)))
            
            # Per-sample stats
            self.event_mask_stats['per_sample_mean'].append(float(np.mean(magnitude)))
            self.event_mask_stats['per_sample_max'].append(float(np.max(magnitude)))
            self.event_mask_stats['per_sample_coverage'].append(float(n) / (h * w))
            
            # Direction statistics
            direction_hist, _ = np.histogram(angles, bins=self.flow_stats['direction_bins'])
            self.event_mask_stats['direction_counts'] += direction_hist
            
            # Quadrant counts
            nonzero_mask = magnitude > 0.01
            if nonzero_mask.any():
                u_nz = flow_u_valid[nonzero_mask]
                v_nz = flow_v_valid[nonzero_mask]
                self.event_mask_stats['quadrant_counts']['right'] += np.sum(u_nz > 0)
                self.event_mask_stats['quadrant_counts']['left'] += np.sum(u_nz < 0)
                self.event_mask_stats['quadrant_counts']['down'] += np.sum(v_nz > 0)
                self.event_mask_stats['quadrant_counts']['up'] += np.sum(v_nz < 0)
                
        except Exception as e:
            self._log(f"Error processing event-masked flow {flow_path}: {e}")

                    num_windows += 1

        except Exception as e:
            self._log(f"Error processing event windows from {flow_h5_path}: {e}")
        return num_windows
    
    def analyze_flow_h5_with_event_mask(self, flow_h5_path, event_h5_path, frame_fps=30.0):
        """Analyze flow.h5 using an event mask built from the shared sequence event stream."""
        try:
            with h5py.File(flow_h5_path, 'r') as flow_f, h5py.File(event_h5_path, 'r') as event_f:
                if 'flow/forward' not in flow_f:
                    self._log(f"Warning: Expected dataset 'flow/forward' not found in {flow_h5_path}")
                    return 0
                if not all(key in event_f for key in ['events/x', 'events/y', 'events/t']):
                    self._log(f"Warning: Expected events/x,y,t not found in {event_h5_path}")
                    return 0

                flow_ds = flow_f['flow/forward']
                valid_ds = flow_f['flow/valid'] if 'flow/valid' in flow_f else None
                event_start_ds = flow_f['flow/event_start'] if 'flow/event_start' in flow_f else None
                event_end_ds = flow_f['flow/event_end'] if 'flow/event_end' in flow_f else None
                x = event_f['events/x'][:]
                y = event_f['events/y'][:]
                t = event_f['events/t'][:]
                if t.size == 0:
                    return 0

                processed = 0
                num_frames = flow_ds.shape[0]
                # Fallback FPS-based timing when event_start/end metadata is absent
                t_start_us = float(t[0])
                frame_interval_us = 1e6 / frame_fps if frame_fps > 0 else 1e6 / 30.0

                for frame_idx in range(num_frames):
                    frame = flow_ds[frame_idx]  # (H, W, 2)
                    if valid_ds is not None:
                        frame = np.concatenate([frame, valid_ds[frame_idx]], axis=-1)  # (H, W, 3)
                    flow_frame = frame
                    if len(flow_frame.shape) != 3 or flow_frame.shape[2] not in (2, 3):
                        continue

                    flow_u = flow_frame[:, :, 0]
                    flow_v = flow_frame[:, :, 1]
                    h, w = flow_u.shape
                    valid_mask = (flow_frame[:, :, 2] > 0.5) if flow_frame.shape[2] == 3 else np.ones((h, w), dtype=bool)

                    if event_start_ds is not None and event_end_ds is not None:
                        t0 = float(event_start_ds[frame_idx])
                        t1 = float(event_end_ds[frame_idx])
                    else:
                        t1 = t_start_us + (frame_idx + 1) * frame_interval_us
                        t0 = max(t_start_us, t1 - _EVENT_WINDOW_US)
                    start_idx = np.searchsorted(t, t0, side='left')
                    end_idx = np.searchsorted(t, t1, side='left')
                    self.event_stats['events_per_flow_frame'].append(int(end_idx - start_idx))

                    event_mask = np.zeros((h, w), dtype=bool)
                    if end_idx > start_idx:
                        x_slice = x[start_idx:end_idx]
                        y_slice = y[start_idx:end_idx]
                        valid_coords = (x_slice >= 0) & (x_slice < w) & (y_slice >= 0) & (y_slice < h)
                        if np.any(valid_coords):
                            event_mask[y_slice[valid_coords].astype(int), x_slice[valid_coords].astype(int)] = True

                    magnitude_mask = np.sqrt(flow_u**2 + flow_v**2) > _DIRECTION_MAG_THRESHOLD
                    combined_mask = valid_mask & event_mask & magnitude_mask
                    if not combined_mask.any():
                        continue

                    flow_u_valid = flow_u[combined_mask]
                    flow_v_valid = flow_v[combined_mask]
                    magnitude = np.sqrt(flow_u_valid**2 + flow_v_valid**2)
                    angles = np.arctan2(flow_v_valid, flow_u_valid)
                    self._accumulate_event_window_magnitude_histogram(magnitude)
                    self._accumulate_event_window_component_histograms(flow_u_valid, flow_v_valid)

                    n = len(magnitude)
                    self.event_mask_stats['total_pixels'] += h * w
                    self.event_mask_stats['total_valid_pixels'] += n
                    self.event_mask_stats['sample_count'] += 1
                    self.event_mask_stats['mag_sum'] += np.sum(magnitude)
                    self.event_mask_stats['mag_sum_sq'] += np.sum(magnitude**2)
                    self.event_mask_stats['mag_min'] = min(self.event_mask_stats['mag_min'], float(np.min(magnitude)))
                    self.event_mask_stats['mag_max'] = max(self.event_mask_stats['mag_max'], float(np.max(magnitude)))
                    self.event_mask_stats['u_sum'] += np.sum(flow_u_valid)
                    self.event_mask_stats['u_sum_sq'] += np.sum(flow_u_valid**2)
                    self.event_mask_stats['u_min'] = min(self.event_mask_stats['u_min'], float(np.min(flow_u_valid)))
                    self.event_mask_stats['u_max'] = max(self.event_mask_stats['u_max'], float(np.max(flow_u_valid)))
                    self.event_mask_stats['v_sum'] += np.sum(flow_v_valid)
                    self.event_mask_stats['v_sum_sq'] += np.sum(flow_v_valid**2)
                    self.event_mask_stats['v_min'] = min(self.event_mask_stats['v_min'], float(np.min(flow_v_valid)))
                    self.event_mask_stats['v_max'] = max(self.event_mask_stats['v_max'], float(np.max(flow_v_valid)))
                    self.event_mask_stats['per_sample_mean'].append(float(np.mean(magnitude)))
                    self.event_mask_stats['per_sample_max'].append(float(np.max(magnitude)))
                    self.event_mask_stats['per_sample_coverage'].append(float(n) / (h * w))

                    direction_hist, _ = np.histogram(angles, bins=self.flow_stats['direction_bins'])
                    self.event_mask_stats['direction_counts'] += direction_hist

                    nonzero_mask = magnitude > _DIRECTION_MAG_THRESHOLD
                    if nonzero_mask.any():
                        u_nz = flow_u_valid[nonzero_mask]
                        v_nz = flow_v_valid[nonzero_mask]
                        self.event_mask_stats['quadrant_counts']['right'] += np.sum(u_nz > 0)
                        self.event_mask_stats['quadrant_counts']['left'] += np.sum(u_nz < 0)
                        self.event_mask_stats['quadrant_counts']['down'] += np.sum(v_nz > 0)
                        self.event_mask_stats['quadrant_counts']['up'] += np.sum(v_nz < 0)

                    processed += 1

                return processed
        except Exception as e:
            self._log(f"Error processing event-masked flow {flow_h5_path}: {e}")
        return 0
    
    def _initialize_histograms(self):
        """Initialize histogram bins based on observed data range"""
        self._log("\nInitializing histograms based on data range...")
        
        # Use observed min/max with some padding
        mag_range = [0, max(self.flow_stats['mag_max'] * 1.1, 10)]  # Pad by 10% or at least 10
        u_range = [self.flow_stats['u_min'] * 1.1 if self.flow_stats['u_min'] < 0 else self.flow_stats['u_min'] * 0.9,
                   self.flow_stats['u_max'] * 1.1 if self.flow_stats['u_max'] > 0 else self.flow_stats['u_max'] * 0.9]
        v_range = [self.flow_stats['v_min'] * 1.1 if self.flow_stats['v_min'] < 0 else self.flow_stats['v_min'] * 0.9,
                   self.flow_stats['v_max'] * 1.1 if self.flow_stats['v_max'] > 0 else self.flow_stats['v_max'] * 0.9]
        
        # Create bins
        self.histogram_data['magnitude']['bins'] = np.linspace(mag_range[0], mag_range[1], self.histogram_bins + 1)
        self.histogram_data['magnitude']['counts'] = np.zeros(self.histogram_bins, dtype=np.int64)
        self.histogram_data['magnitude']['range'] = mag_range
        
        self.histogram_data['u']['bins'] = np.linspace(u_range[0], u_range[1], self.histogram_bins + 1)
        self.histogram_data['u']['counts'] = np.zeros(self.histogram_bins, dtype=np.int64)
        self.histogram_data['u']['range'] = u_range
        
        self.histogram_data['v']['bins'] = np.linspace(v_range[0], v_range[1], self.histogram_bins + 1)
        self.histogram_data['v']['counts'] = np.zeros(self.histogram_bins, dtype=np.int64)
        self.histogram_data['v']['range'] = v_range
        self.event_window_magnitude_counts = np.zeros(self.histogram_bins, dtype=np.int64)
        self.event_window_u_counts = np.zeros(self.histogram_bins, dtype=np.int64)
        self.event_window_v_counts = np.zeros(self.histogram_bins, dtype=np.int64)
        
        self.histogram_initialized = True
        
        self._log(f"  Magnitude range: [{mag_range[0]:.2f}, {mag_range[1]:.2f}]")
        self._log(f"  U component range: [{u_range[0]:.2f}, {u_range[1]:.2f}]")
        self._log(f"  V component range: [{v_range[0]:.2f}, {v_range[1]:.2f}]")
        self._log(f"  Bins per histogram: {self.histogram_bins}")
    
    def _accumulate_histogram(self, name, data):
        """Accumulate data into histogram bins"""
        bins = self.histogram_data[name]['bins']
        counts = self.histogram_data[name]['counts']
        
        # Compute histogram for this batch and add to cumulative counts
        batch_counts, _ = np.histogram(data, bins=bins)
        counts += batch_counts

    def _accumulate_event_window_magnitude_histogram(self, magnitude):
        """Accumulate event-window filtered magnitudes into the magnitude bins."""
        if not self.histogram_initialized or self.event_window_magnitude_counts is None:
            return
        bins = self.histogram_data['magnitude']['bins']
        batch_counts, _ = np.histogram(magnitude, bins=bins)
        self.event_window_magnitude_counts += batch_counts

    def _accumulate_event_window_component_histograms(self, flow_u_valid, flow_v_valid):
        """Accumulate event-window filtered u and v components into their bins."""
        if not self.histogram_initialized:
            return
        if self.event_window_u_counts is None or self.event_window_v_counts is None:
            return

        u_bins = self.histogram_data['u']['bins']
        v_bins = self.histogram_data['v']['bins']
        u_batch_counts, _ = np.histogram(flow_u_valid, bins=u_bins)
        v_batch_counts, _ = np.histogram(flow_v_valid, bins=v_bins)
        self.event_window_u_counts += u_batch_counts
        self.event_window_v_counts += v_batch_counts
    
    def _compute_percentiles_from_histogram(self, name, percentiles=[25, 50, 75, 95, 99]):
        """Compute percentiles from histogram bins using cumulative distribution"""
        bins = self.histogram_data[name]['bins']
        counts = self.histogram_data[name]['counts']
        
        # Compute cumulative distribution
        cumsum = np.cumsum(counts)
        total = cumsum[-1]
        
        if total == 0:
            return {p: 0.0 for p in percentiles}
        
        # Compute bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Find percentile values
        result = {}
        for p in percentiles:
            target = total * p / 100.0
            idx = np.searchsorted(cumsum, target)
            if idx >= len(bin_centers):
                idx = len(bin_centers) - 1
            result[p] = bin_centers[idx]
        
        return result

    @staticmethod
    def _compute_percentiles_from_counts(bins, counts, percentiles=[25, 50, 75, 95, 99]):
        """Compute percentiles from explicit histogram bins/counts."""
        cumsum = np.cumsum(counts)
        total = cumsum[-1] if len(cumsum) > 0 else 0

        if total == 0:
            return {p: 0.0 for p in percentiles}

        bin_centers = (bins[:-1] + bins[1:]) / 2
        result = {}
        for p in percentiles:
            target = total * p / 100.0
            idx = np.searchsorted(cumsum, target)
            if idx >= len(bin_centers):
                idx = len(bin_centers) - 1
            result[p] = bin_centers[idx]

        return result

    @staticmethod
    def _compute_mean_from_counts(bins, counts):
        """Compute mean from histogram bins/counts."""
        if len(counts) == 0 or np.sum(counts) == 0:
            return 0.0
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean = np.sum(bin_centers * counts) / np.sum(counts)
        return float(mean)
    
    def _log(self, message):
        """Write message to log file or stdout"""
        if self.log_file:
            self.log_file.write(message + '\n')
            self.log_file.flush()
        else:
            print(message)
    
    def scan_dataset(self):
        """Scan the entire dataset directory (two-pass approach)"""
        self._log(f"Scanning dataset at: {self.dataset_path}")
        
        # Find all subdirectories
        subdirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        self._log(f"Found {len(subdirs)} sample directories")
        
        flow_count = 0
        event_count = 0
        
        # PASS 1: Collect basic stats to determine histogram ranges
        self._log("\nPass 1: Computing data ranges...")
        for subdir in tqdm(subdirs, desc="Pass 1"):
            # Process flow files
            flow_h5_file = subdir / 'flow.h5'
            if flow_h5_file.exists():
                flow_count += self.analyze_flow_h5_file(flow_h5_file)

            # Process event files (one sample = one flow-frame event window)
            event_h5_file = subdir / 'events.h5'
            if flow_h5_file.exists() and event_h5_file.exists():
                event_count += self.analyze_event_windows(flow_h5_file, event_h5_file)
        
        # Initialize histograms based on observed range
        self._initialize_histograms()
        
        # PASS 2: Accumulate histogram data
        self._log("\nPass 2: Building histograms from all pixels...")
        for subdir in tqdm(subdirs, desc="Pass 2"):
            flow_h5_file = subdir / 'flow.h5'
            if flow_h5_file.exists():
                self.accumulate_flow_h5_histograms(flow_h5_file)
        
        # PASS 3: Event-based valid mask analysis
        self._log("\nPass 3: Analyzing with event-based valid masks...")
        event_mask_count = 0
        for subdir in tqdm(subdirs, desc="Pass 3"):
            flow_h5_file = subdir / 'flow.h5'
            event_h5_file = subdir / 'events.h5'
            if flow_h5_file.exists() and event_h5_file.exists():
                event_mask_count += self.analyze_flow_h5_with_event_mask(flow_h5_file, event_h5_file)
        
        self._log(f"\nTotal processed: {flow_count} flow files and {event_count} event files")
        self._log(f"Event-masked analysis: {event_mask_count} flow-event pairs")
        self._log(f"All {self.flow_stats['total_pixels']:,} valid pixels included in histograms")
    
    def compute_summary_statistics(self):
        """Compute summary statistics from collected data"""
        self._log("\n" + "="*70)
        self._log("FLOW STATISTICS")
        self._log("="*70)
        
        if self.flow_stats['sample_count'] > 0:
            n = self.flow_stats['total_pixels']
            
            # Compute mean and std from running sums
            mag_mean = self.flow_stats['mag_sum'] / n
            mag_variance = (self.flow_stats['mag_sum_sq'] / n) - (mag_mean ** 2)
            mag_std = np.sqrt(max(0, mag_variance))
            
            u_mean = self.flow_stats['u_sum'] / n
            u_variance = (self.flow_stats['u_sum_sq'] / n) - (u_mean ** 2)
            u_std = np.sqrt(max(0, u_variance))
            
            v_mean = self.flow_stats['v_sum'] / n
            v_variance = (self.flow_stats['v_sum_sq'] / n) - (v_mean ** 2)
            v_std = np.sqrt(max(0, v_variance))
            
            self._log(f"\nTotal flow samples analyzed: {self.flow_stats['sample_count']}")
            self._log(f"Total valid pixels analyzed: {n:,}")
            
            self._log(f"\nFlow Magnitude Statistics:")
            self._log(f"  Mean:               {mag_mean:.4f}")
            self._log(f"  Std Dev:            {mag_std:.4f}")
            self._log(f"  Min:                {self.flow_stats['mag_min']:.4f}")
            self._log(f"  Max:                {self.flow_stats['mag_max']:.4f}")
            
            # Percentiles from full histogram data
            if self.histogram_initialized:
                mag_percentiles = self._compute_percentiles_from_histogram('magnitude')
                self._log(f"  Median:             {mag_percentiles[50]:.4f}")
                self._log(f"  25th percentile:    {mag_percentiles[25]:.4f}")
                self._log(f"  75th percentile:    {mag_percentiles[75]:.4f}")
                self._log(f"  95th percentile:    {mag_percentiles[95]:.4f}")
                self._log(f"  99th percentile:    {mag_percentiles[99]:.4f}")
                
                # Non-zero magnitude statistics
                mag_bins = self.histogram_data['magnitude']['bins']
                mag_counts = self.histogram_data['magnitude']['counts']
                mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2
                
                # Find bins with magnitude > 0 (use small threshold for numerical stability)
                nonzero_mask = mag_centers > 0.01
                nonzero_counts = mag_counts[nonzero_mask]
                nonzero_centers = mag_centers[nonzero_mask]
                
                if nonzero_counts.sum() > 0:
                    # Compute weighted statistics for non-zero magnitudes
                    total_nonzero = nonzero_counts.sum()
                    nonzero_mean = np.sum(nonzero_centers * nonzero_counts) / total_nonzero
                    nonzero_var = np.sum(((nonzero_centers - nonzero_mean) ** 2) * nonzero_counts) / total_nonzero
                    nonzero_std = np.sqrt(nonzero_var)
                    nonzero_min = nonzero_centers[nonzero_counts > 0][0]
                    nonzero_max = nonzero_centers[nonzero_counts > 0][-1]
                    
                    # Compute percentiles for non-zero values
                    cumsum_nonzero = np.cumsum(nonzero_counts)
                    percentiles_nonzero = {}
                    for p in [50, 75, 95, 99]:
                        target = cumsum_nonzero[-1] * p / 100.0
                        idx = np.searchsorted(cumsum_nonzero, target)
                        if idx >= len(nonzero_centers):
                            idx = len(nonzero_centers) - 1
                        percentiles_nonzero[p] = nonzero_centers[idx]
                    
                    self._log(f"\nFlow Magnitude Statistics (Non-Zero Only):")
                    self._log(f"  Total non-zero pixels:  {int(total_nonzero):,} ({100*total_nonzero/n:.2f}% of valid pixels)")
                    self._log(f"  Mean:               {nonzero_mean:.4f}")
                    self._log(f"  Std Dev:            {nonzero_std:.4f}")
                    self._log(f"  Min:                {nonzero_min:.4f}")
                    self._log(f"  Max:                {nonzero_max:.4f}")
                    self._log(f"  Median:             {percentiles_nonzero[50]:.4f}")
                    self._log(f"  75th percentile:    {percentiles_nonzero[75]:.4f}")
                    self._log(f"  95th percentile:    {percentiles_nonzero[95]:.4f}")
                    self._log(f"  99th percentile:    {percentiles_nonzero[99]:.4f}")
            
            self._log(f"\nFlow U Component Statistics:")
            self._log(f"  Mean:               {u_mean:.4f}")
            self._log(f"  Std Dev:            {u_std:.4f}")
            self._log(f"  Min:                {self.flow_stats['u_min']:.4f}")
            self._log(f"  Max:                {self.flow_stats['u_max']:.4f}")
            
            self._log(f"\nFlow V Component Statistics:")
            self._log(f"  Mean:               {v_mean:.4f}")
            self._log(f"  Std Dev:            {v_std:.4f}")
            self._log(f"  Min:                {self.flow_stats['v_min']:.4f}")
            self._log(f"  Max:                {self.flow_stats['v_max']:.4f}")
            
            self._log(f"\nPer-Sample Statistics:")
            self._log(f"  Avg of sample means:    {np.mean(self.flow_stats['per_sample_mean']):.4f}")
            self._log(f"  Avg of sample maxs:     {np.mean(self.flow_stats['per_sample_max']):.4f}")
            self._log(f"  Avg of sample mins:     {np.mean(self.flow_stats['per_sample_min']):.4f}")
            self._log(f"  Avg of sample stds:     {np.mean(self.flow_stats['per_sample_std']):.4f}")
            
            # Flow direction statistics
            self._log(f"\nFlow Direction Statistics:")
            total_directional = self.flow_stats['direction_counts'].sum()
            if total_directional > 0:
                # Find dominant direction bins
                direction_centers = (self.flow_stats['direction_bins'][:-1] + self.flow_stats['direction_bins'][1:]) / 2
                direction_degrees = np.rad2deg(direction_centers)
                top_bins = np.argsort(self.flow_stats['direction_counts'])[-5:][::-1]
                
                self._log(f"  Top 5 Direction Bins (angle, count, %)")
                for i, bin_idx in enumerate(top_bins, 1):
                    angle_deg = direction_degrees[bin_idx]
                    count = self.flow_stats['direction_counts'][bin_idx]
                    percent = 100 * count / total_directional
                    self._log(f"    {i}. {angle_deg:6.1f}°: {int(count):,} ({percent:.2f}%)")
                
            # Quadrant statistics
            self._log(f"\nFlow Quadrant Distribution (non-zero flow only):")
            quad_total = sum(self.flow_stats['quadrant_counts'].values())
            if quad_total > 0:
                for direction in ['right', 'left', 'up', 'down']:
                    count = self.flow_stats['quadrant_counts'][direction]
                    percent = 100 * count / quad_total
                    self._log(f"  {direction.capitalize():>6s}: {int(count):,} ({percent:.2f}%)")
            
            # Display top 10 magnitude samples
            if self.top_magnitude_samples:
                self._log(f"\nTop 10 Flow Magnitude Samples:")
                sorted_samples = sorted(self.top_magnitude_samples, key=lambda x: x[0], reverse=True)[:10]
                for i, (max_mag, file_path) in enumerate(sorted_samples, 1):
                    self._log(f"  {i:2d}. {file_path}: {max_mag:.4f} pixels")
        else:
            self._log("No flow data found!")
        
        self._log("\n" + "="*70)
        self._log("EVENT STATISTICS")
        self._log("="*70)
        
        if self.event_stats['sample_count'] > 0:
            events_per_sample = np.array(self.event_stats['events_per_sample'])
            
            self._log(f"\nTotal event files analyzed: {self.event_stats['sample_count']}")
            self._log(f"Total events: {self.event_stats['total_events']:,}")
            
            self._log(f"\nEvents Per Sample:")
            self._log(f"  Mean:               {np.mean(events_per_sample):.2f}")
            self._log(f"  Median:             {np.median(events_per_sample):.2f}")
            self._log(f"  Std Dev:            {np.std(events_per_sample):.2f}")
            self._log(f"  Min:                {np.min(events_per_sample):.0f}")
            self._log(f"  Max:                {np.max(events_per_sample):.0f}")
            
            if self.event_stats['polarity_counts']:
                self._log(f"\nEvent Polarities:")
                total = sum(self.event_stats['polarity_counts'].values())
                for pol in sorted(self.event_stats['polarity_counts'].keys()):
                    count = self.event_stats['polarity_counts'][pol]
                    self._log(f"  Polarity {pol}: {count:,} ({100*count/total:.2f}%)")
            
            if self.event_stats['time_spans']:
                time_spans = np.array(self.event_stats['time_spans'])
                self._log(f"\nTime Span Per Sample (microseconds):")
                self._log(f"  Mean:               {np.mean(time_spans):.2f}")
                self._log(f"  Median:             {np.median(time_spans):.2f}")
                self._log(f"  Min:                {np.min(time_spans):.2f}")
                self._log(f"  Max:                {np.max(time_spans):.2f}")
        else:
            self._log("No event data found!")
        
        self._log("\n" + "="*70)
        
        # Event-based valid mask statistics
        if self.event_mask_stats['sample_count'] > 0:
            self._log("="*70)
            self._log("EVENT-BASED VALID MASK FLOW STATISTICS")
            self._log("="*70)
            self._log("(Only considers pixels that had events between flow frames)")
            
            n = self.event_mask_stats['total_valid_pixels']
            total = self.event_mask_stats['total_pixels']
            
            self._log(f"\nTotal samples analyzed: {self.event_mask_stats['sample_count']}")
            self._log(f"Total pixels analyzed: {total:,}")
            self._log(f"Total valid pixels (with events): {n:,} ({100*n/total:.2f}% of all pixels)")
            
            if n > 0:
                # Magnitude stats
                mag_mean = self.event_mask_stats['mag_sum'] / n
                mag_std = np.sqrt(max(0, (self.event_mask_stats['mag_sum_sq'] / n) - (mag_mean ** 2)))
                
                self._log(f"\nFlow Magnitude Statistics (Event-Masked):")
                self._log(f"  Mean:               {mag_mean:.4f}")
                self._log(f"  Std Dev:            {mag_std:.4f}")
                self._log(f"  Min:                {self.event_mask_stats['mag_min']:.4f}")
                self._log(f"  Max:                {self.event_mask_stats['mag_max']:.4f}")
                
                # U and V components
                u_mean = self.event_mask_stats['u_sum'] / n
                u_std = np.sqrt(max(0, (self.event_mask_stats['u_sum_sq'] / n) - (u_mean ** 2)))
                
                v_mean = self.event_mask_stats['v_sum'] / n
                v_std = np.sqrt(max(0, (self.event_mask_stats['v_sum_sq'] / n) - (v_mean ** 2)))
                
                self._log(f"\nFlow U Component Statistics (Event-Masked):")
                self._log(f"  Mean:               {u_mean:.4f}")
                self._log(f"  Std Dev:            {u_std:.4f}")
                self._log(f"  Min:                {self.event_mask_stats['u_min']:.4f}")
                self._log(f"  Max:                {self.event_mask_stats['u_max']:.4f}")
                
                self._log(f"\nFlow V Component Statistics (Event-Masked):")
                self._log(f"  Mean:               {v_mean:.4f}")
                self._log(f"  Std Dev:            {v_std:.4f}")
                self._log(f"  Min:                {self.event_mask_stats['v_min']:.4f}")
                self._log(f"  Max:                {self.event_mask_stats['v_max']:.4f}")
                
                # Per-sample stats
                if self.event_mask_stats['per_sample_mean']:
                    self._log(f"\nPer-Sample Statistics (Event-Masked):")
                    self._log(f"  Avg of sample means:    {np.mean(self.event_mask_stats['per_sample_mean']):.4f}")
                    self._log(f"  Avg of sample maxs:     {np.mean(self.event_mask_stats['per_sample_max']):.4f}")
                    self._log(f"  Avg event coverage:     {np.mean(self.event_mask_stats['per_sample_coverage']):.2%}")
                
                # Direction statistics
                self._log(f"\nFlow Direction Statistics (Event-Masked):")
                total_directional = self.event_mask_stats['direction_counts'].sum()
                if total_directional > 0:
                    direction_centers = (self.flow_stats['direction_bins'][:-1] + self.flow_stats['direction_bins'][1:]) / 2
                    direction_degrees = np.rad2deg(direction_centers)
                    top_bins = np.argsort(self.event_mask_stats['direction_counts'])[-5:][::-1]
                    
                    self._log(f"  Top 5 Direction Bins (angle, count, %)")
                    for i, bin_idx in enumerate(top_bins, 1):
                        angle_deg = direction_degrees[bin_idx]
                        count = self.event_mask_stats['direction_counts'][bin_idx]
                        percent = 100 * count / total_directional
                        self._log(f"    {i}. {angle_deg:6.1f}°: {int(count):,} ({percent:.2f}%)")
                
                # Quadrant statistics
                self._log(f"\nFlow Quadrant Distribution (Event-Masked, non-zero):")
                quad_total = sum(self.event_mask_stats['quadrant_counts'].values())
                if quad_total > 0:
                    for direction in ['right', 'left', 'up', 'down']:
                        count = self.event_mask_stats['quadrant_counts'][direction]
                        percent = 100 * count / quad_total
                        self._log(f"  {direction.capitalize():>6s}: {int(count):,} ({percent:.2f}%)")
        
        self._log("\n" + "="*70)
    
    def save_statistics_to_log(self, output_path):
        """Save statistics summary to log file"""
        n = self.flow_stats['total_pixels']
        
        # Compute statistics
        if n > 0:
            mag_mean = self.flow_stats['mag_sum'] / n
            mag_std = np.sqrt(max(0, (self.flow_stats['mag_sum_sq'] / n) - (mag_mean ** 2)))
            u_mean = self.flow_stats['u_sum'] / n
            u_std = np.sqrt(max(0, (self.flow_stats['u_sum_sq'] / n) - (u_mean ** 2)))
            v_mean = self.flow_stats['v_sum'] / n
            v_std = np.sqrt(max(0, (self.flow_stats['v_sum_sq'] / n) - (v_mean ** 2)))
        else:
            mag_mean = mag_std = u_mean = u_std = v_mean = v_std = None
        
        stats_dict = {
            'flow': {
                'sample_count': int(self.flow_stats['sample_count']),
                'total_pixels': int(n),
                'magnitude': {
                    'mean': float(mag_mean) if mag_mean is not None else None,
                    'std': float(mag_std) if mag_std is not None else None,
                    'min': float(self.flow_stats['mag_min']) if self.flow_stats['mag_min'] != float('inf') else None,
                    'max': float(self.flow_stats['mag_max']) if self.flow_stats['mag_max'] != float('-inf') else None,
                },
                'u_component': {
                    'mean': float(u_mean) if u_mean is not None else None,
                    'std': float(u_std) if u_std is not None else None,
                    'min': float(self.flow_stats['u_min']) if self.flow_stats['u_min'] != float('inf') else None,
                    'max': float(self.flow_stats['u_max']) if self.flow_stats['u_max'] != float('-inf') else None,
                },
                'v_component': {
                    'mean': float(v_mean) if v_mean is not None else None,
                    'std': float(v_std) if v_std is not None else None,
                    'min': float(self.flow_stats['v_min']) if self.flow_stats['v_min'] != float('inf') else None,
                    'max': float(self.flow_stats['v_max']) if self.flow_stats['v_max'] != float('-inf') else None,
                },
                'per_sample': {
                    'avg_mean': float(np.mean(self.flow_stats['per_sample_mean'])) if self.flow_stats['per_sample_mean'] else None,
                    'avg_max': float(np.mean(self.flow_stats['per_sample_max'])) if self.flow_stats['per_sample_max'] else None,
                    'avg_min': float(np.mean(self.flow_stats['per_sample_min'])) if self.flow_stats['per_sample_min'] else None,
                    'avg_std': float(np.mean(self.flow_stats['per_sample_std'])) if self.flow_stats['per_sample_std'] else None,
                }
            },
            'events': {
                'sample_count': int(self.event_stats['sample_count']),
                'total_events': int(self.event_stats['total_events']),
                'events_per_sample': {
                    'mean': float(np.mean(self.event_stats['events_per_sample'])) if self.event_stats['events_per_sample'] else None,
                    'median': float(np.median(self.event_stats['events_per_sample'])) if self.event_stats['events_per_sample'] else None,
                    'std': float(np.std(self.event_stats['events_per_sample'])) if self.event_stats['events_per_sample'] else None,
                    'min': float(np.min(self.event_stats['events_per_sample'])) if self.event_stats['events_per_sample'] else None,
                    'max': float(np.max(self.event_stats['events_per_sample'])) if self.event_stats['events_per_sample'] else None,
                },
                'polarity_counts': self.event_stats['polarity_counts']
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"\nStatistics saved to: {output_path}")
    
    def plot_histograms(self, output_dir):
        """Generate histogram plots from all pixels"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if self.histogram_initialized:
            # Combined flow distributions: magnitude, horizontal (u), vertical (v)
            fig, axes = plt.subplots(
                3,
                1,
                figsize=(self.fig_width, _STACKED_SUBPLOT_HEIGHT_IN * 3),
                sharey=True,
            )

            # Flow magnitude histogram
            ax = axes[0]
            mag_bins = self.histogram_data['magnitude']['bins']
            mag_counts = self.event_window_magnitude_counts
            if mag_counts is None or np.sum(mag_counts) == 0:
                mag_counts = self.histogram_data['magnitude']['counts']
            ax.stairs(mag_counts, mag_bins, fill=True, alpha=0.7)
            ax.set_xlabel('Flow Magnitude (px/frame)')
            ax.set_ylabel('Frequency (log scale)')
            ax.set_title('Flow Magnitude Distribution')
            ax.set_yscale('log')
            ax.grid(False)
            mag_mean = self._compute_mean_from_counts(mag_bins, mag_counts)
            ax.axvline(mag_mean, color='red', linestyle='-', linewidth=2, label=f"Mean: {mag_mean:.2f}")
            ax.legend(loc='upper right')

            # U component distribution
            ax = axes[1]
            u_bins = self.histogram_data['u']['bins']
            u_counts = self.event_window_u_counts
            if u_counts is None or np.sum(u_counts) == 0:
                u_counts = self.histogram_data['u']['counts']
            ax.stairs(u_counts, u_bins, fill=True, alpha=0.7, color='blue')
            ax.set_xlabel(r'Flow $u$ Component (px/frame)')
            ax.set_ylabel('Frequency (log scale)')
            ax.set_title('Horizontal Flow Distribution')
            ax.set_yscale('log')
            ax.grid(False)
            u_mean = self._compute_mean_from_counts(u_bins, u_counts)
            ax.axvline(u_mean, color='red', linestyle='-', linewidth=2, label=f"Mean: {u_mean:.2f}")
            ax.legend(loc='upper right')

            # V component distribution
            ax = axes[2]
            v_bins = self.histogram_data['v']['bins']
            v_counts = self.event_window_v_counts
            if v_counts is None or np.sum(v_counts) == 0:
                v_counts = self.histogram_data['v']['counts']
            ax.stairs(v_counts, v_bins, fill=True, alpha=0.7, color='green')
            ax.set_xlabel(r'Flow $v$ Component (px/frame)')
            ax.set_ylabel('Frequency (log scale)')
            ax.set_title('Vertical Flow Distribution')
            ax.set_yscale('log')
            ax.grid(False)
            v_mean = self._compute_mean_from_counts(v_bins, v_counts)
            ax.axvline(v_mean, color='red', linestyle='-', linewidth=2, label=f"Mean: {v_mean:.2f}")
            ax.legend(loc='upper right')
            plt.tight_layout()
            flow_combined_path = output_dir / 'flow_distributions_stacked.pdf'
            plt.savefig(flow_combined_path, dpi=300, bbox_inches='tight', pad_inches=0)
            self._log(f"Combined flow distribution figure saved to: {flow_combined_path}")
            plt.close()
            
            # Flow direction analysis - linear histogram only
            if self.flow_stats['direction_counts'].sum() > 0:
                fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
                
                direction_centers = (self.flow_stats['direction_bins'][:-1] + self.flow_stats['direction_bins'][1:]) / 2
                width = np.diff(self.flow_stats['direction_bins'])[0]
                direction_degrees = np.rad2deg(direction_centers)
                counts_millions = self.flow_stats['direction_counts'] / 1e6
                ax.bar(direction_degrees, counts_millions,
                       width=np.rad2deg(width), alpha=0.7, edgecolor='black')
                ax.set_xlabel(r'Flow Direction ($\degree$)')
                ax.set_ylabel(r'Frequency ($\times 10^6$)')
                ax.set_title('Flow Direction Distribution')
                ax.grid(False)
                ax.set_xlim(-180, 180)
                
                plt.tight_layout()
                direction_path = output_dir / 'flow_direction_analysis.pdf'
                plt.savefig(direction_path, dpi=300, bbox_inches='tight', pad_inches=0)
                self._log(f"Flow direction analysis saved to: {direction_path}")
                plt.close()
            
        if self.event_stats['events_per_sample']:
            if self.event_stats['x_coords']:
                x_coords = np.array(self.event_stats['x_coords'])
                fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
                counts, _, _ = ax.hist(x_coords, bins=50, edgecolor='black', alpha=0.7, color='blue')
                ax.set_xlabel(r'$x$ Coordinate (px)')
                ax.set_ylabel('Event Count (log scale)')
                ax.set_title(r'Event $x$ Coordinate Distribution')
                ax.set_yscale('log')

                ymax = float(np.max(counts)) if counts.size > 0 else 1.0
                ylim_top = max(10.0, 10.0 ** np.ceil(np.log10(max(1.0, ymax))))
                ax.set_ylim(10**3, 10**5)

                ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
                ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.grid(False)
                mean_x = float(np.mean(x_coords)) if len(x_coords) > 0 else 0.0
                ax.axvline(mean_x, color='red', linestyle='-', linewidth=2, label=f"Mean: {mean_x:.0f}")
                ax.legend()
                plt.tight_layout()
                x_coord_path = output_dir / 'event_x_coordinate_histogram.pdf'
                plt.savefig(x_coord_path, dpi=300, bbox_inches='tight', pad_inches=0)
                self._log(f"Event X-coordinate histogram saved to: {x_coord_path}")
                plt.close()

            if self.event_stats['y_coords']:
                y_coords = np.array(self.event_stats['y_coords'])
                fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
                counts, _, _ = ax.hist(y_coords, bins=50, edgecolor='black', alpha=0.7, color='green')
                ax.set_xlabel(r'$y$ Coordinate (px)')
                ax.set_ylabel('Event Count (log scale)')
                ax.set_title(r'Event $y$ Coordinate Distribution')
                ax.set_yscale('log')

                ymax = float(np.max(counts)) if counts.size > 0 else 1.0
                ylim_top = max(10.0, 10.0 ** np.ceil(np.log10(max(1.0, ymax))))
                ax.set_ylim(ylim_top/100, ylim_top)

                ax.grid(False)
                mean_y = float(np.mean(y_coords)) if len(y_coords) > 0 else 0.0
                ax.axvline(mean_y, color='red', linestyle='-', linewidth=2, label=f"Mean: {mean_y:.0f}")
                ax.legend()
                plt.tight_layout()
                y_coord_path = output_dir / 'event_y_coordinate_histogram.pdf'
                plt.savefig(y_coord_path, dpi=300, bbox_inches='tight', pad_inches=0)
                self._log(f"Event Y-coordinate histogram saved to: {y_coord_path}")
                plt.close()

        # Standalone: Event rate (events per second)
        if self.event_stats['event_rate_per_sec']:
            rates = np.array(self.event_stats['event_rate_per_sec']) / 1e6
            fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
            ax.hist(rates, bins=50, edgecolor='black', alpha=0.75, color='teal')
            ax.set_xlabel('Event Rate (Mev/s)')
            ax.set_ylabel('# of Animation Samples (log scale)')
            ax.set_yscale('log')
            ax.set_title('Event Rate Distribution per Animation Sample')
            ax.grid(False)
            mean_rate = np.mean(rates) if len(rates) > 0 else 0.0
            ax.axvline(mean_rate, color='red', linestyle='-', linewidth=1.5,
                       label=f'Mean: {mean_rate:.2f}')
            ax.legend()

            rate_path = output_dir / 'event_rate_per_second.pdf'
            plt.tight_layout()
            plt.savefig(rate_path, dpi=300, bbox_inches='tight', pad_inches=0)
            self._log(f"Event rate plot saved to: {rate_path}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze blink_sim dataset statistics (memory-efficient)')
    parser.add_argument('--dataset-path', type=str, 
                        default='./outputs/hflow320/train',
                        help='Path to the training dataset')
    parser.add_argument('--output-dir', type=str,
                        default='./output/dataset_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--max-histogram-samples', type=int, default=1000000,
                        help='Maximum number of pixels to sample for histograms')
    
    args = parser.parse_args()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create log file
    log_path = output_dir / 'dataset_analysis.log'
    with open(log_path, 'w') as log_file:
        # Initialize analyzer with log file
        analyzer = DatasetAnalyzer(args.dataset_path, max_histogram_samples=args.max_histogram_samples, log_file=log_file)
        
        # Scan dataset
        analyzer.scan_dataset()
        
        # Compute and write statistics to log
        analyzer.compute_summary_statistics()
        
        # Save plots if requested
        if not args.no_plots:
            analyzer.plot_histograms(output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Log file: {log_path}")


if __name__ == '__main__':
    main()
