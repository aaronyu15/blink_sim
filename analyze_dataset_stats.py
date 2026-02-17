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
from tqdm import tqdm
import argparse
import json
from collections import defaultdict


class DatasetAnalyzer:
    def __init__(self, dataset_path, histogram_bins=1000, max_histogram_samples=1000000, log_file=None):
        self.dataset_path = Path(dataset_path)
        self.histogram_bins = histogram_bins
        self.max_histogram_samples = max_histogram_samples
        self.log_file = log_file
        
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
            'polarities': []
        }
        
        # Online histograms - bins will be defined dynamically based on data range
        # We'll use two-pass approach: first pass gets range, second pass builds histogram
        self.histogram_data = {
            'magnitude': {'bins': None, 'counts': None, 'range': [0, 400]},  # Initial guess
            'u': {'bins': None, 'counts': None, 'range': [-250, 250]},
            'v': {'bins': None, 'counts': None, 'range': [-250, 250]}
        }
        self.histogram_initialized = False
        
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
    
    def analyze_flow_file(self, flow_path):
        """Analyze a single flow .npy file with incremental statistics"""
        try:
            flow = np.load(flow_path)
            
            # Flow format is (H, W, 3) with channels: u, v, valid
            if len(flow.shape) == 3 and flow.shape[2] == 3:
                flow_u = flow[:, :, 0]  # u component
                flow_v = flow[:, :, 1]  # v component
                valid = flow[:, :, 2]   # valid mask
                
                # Apply valid mask
                mask = valid > 0.5
                if not mask.any():
                    # No valid flow in this sample
                    return
                
                flow_u_valid = flow_u[mask]
                flow_v_valid = flow_v[mask]
                magnitude = np.sqrt(flow_u_valid**2 + flow_v_valid**2)
                
                # Compute flow directions (angles)
                angles = np.arctan2(flow_v_valid, flow_u_valid)  # Returns angles in [-pi, pi]
                
            elif len(flow.shape) == 3 and flow.shape[2] == 2:
                flow_u = flow[:, :, 0]
                flow_v = flow[:, :, 1]
                magnitude = np.sqrt(flow_u**2 + flow_v**2)
                flow_u_valid = flow_u.flatten()
                flow_v_valid = flow_v.flatten()
                magnitude = magnitude.flatten()
                
                # Compute flow directions
                angles = np.arctan2(flow_v_valid, flow_u_valid)
            else:
                self._log(f"Warning: Unexpected flow shape {flow.shape} in {flow_path}")
                return
            
            # Update running statistics
            n = len(magnitude)
            self.flow_stats['total_pixels'] += n
            self.flow_stats['sample_count'] += 1
            
            # Magnitude stats
            self.flow_stats['mag_sum'] += np.sum(magnitude)
            self.flow_stats['mag_sum_sq'] += np.sum(magnitude**2)
            self.flow_stats['mag_min'] = min(self.flow_stats['mag_min'], float(np.min(magnitude)))
            self.flow_stats['mag_max'] = max(self.flow_stats['mag_max'], float(np.max(magnitude)))
            
            # U component stats
            self.flow_stats['u_sum'] += np.sum(flow_u_valid)
            self.flow_stats['u_sum_sq'] += np.sum(flow_u_valid**2)
            self.flow_stats['u_min'] = min(self.flow_stats['u_min'], float(np.min(flow_u_valid)))
            self.flow_stats['u_max'] = max(self.flow_stats['u_max'], float(np.max(flow_u_valid)))
            
            # V component stats
            self.flow_stats['v_sum'] += np.sum(flow_v_valid)
            self.flow_stats['v_sum_sq'] += np.sum(flow_v_valid**2)
            self.flow_stats['v_min'] = min(self.flow_stats['v_min'], float(np.min(flow_v_valid)))
            self.flow_stats['v_max'] = max(self.flow_stats['v_max'], float(np.max(flow_v_valid)))
            
            # Per-sample statistics
            sample_max_mag = float(np.max(magnitude))
            self.flow_stats['per_sample_mean'].append(float(np.mean(magnitude)))
            self.flow_stats['per_sample_max'].append(sample_max_mag)
            self.flow_stats['per_sample_min'].append(float(np.min(magnitude)))
            self.flow_stats['per_sample_std'].append(float(np.std(magnitude)))
            
            # Track top magnitude samples
            # Extract full relative path from dataset root (e.g., boy1_BaseballHit_0/forward_flow/000005.npy)
            relative_path = flow_path.relative_to(self.dataset_path)
            self.top_magnitude_samples.append((sample_max_mag, str(relative_path)))
            
            # Accumulate into online histograms if initialized
            if self.histogram_initialized:
                self._accumulate_histogram('magnitude', magnitude)
                self._accumulate_histogram('u', flow_u_valid)
                self._accumulate_histogram('v', flow_v_valid)
            
            # Direction statistics
            direction_hist, _ = np.histogram(angles, bins=self.flow_stats['direction_bins'])
            self.flow_stats['direction_counts'] += direction_hist
            
            # Quadrant counts (for non-zero flow only)
            nonzero_mask = magnitude > 0.01
            if nonzero_mask.any():
                u_nz = flow_u_valid[nonzero_mask]
                v_nz = flow_v_valid[nonzero_mask]
                self.flow_stats['quadrant_counts']['right'] += np.sum(u_nz > 0)
                self.flow_stats['quadrant_counts']['left'] += np.sum(u_nz < 0)
                self.flow_stats['quadrant_counts']['down'] += np.sum(v_nz > 0)  # Image coords: positive v is down
                self.flow_stats['quadrant_counts']['up'] += np.sum(v_nz < 0)
            
        except Exception as e:
            self._log(f"Error processing {flow_path}: {e}")
    
    def analyze_event_file(self, event_path):
        """Analyze a single event .h5 file with x,y,t,p format"""
        try:
            with h5py.File(event_path, 'r') as f:
                # Events are stored in separate datasets: events/x, events/y, events/t, events/p
                if 'events/x' in f and 'events/y' in f and 'events/t' in f and 'events/p' in f:
                    x = f['events/x'][:]
                    y = f['events/y'][:]
                    t = f['events/t'][:]
                    p = f['events/p'][:]
                    
                    num_events = len(x)
                    self.event_stats['total_events'] += num_events
                    self.event_stats['events_per_sample'].append(num_events)
                    self.event_stats['sample_count'] += 1
                    
                    # Sample data for histograms (max 10k events per file)
                    if len(self.event_stats['x_coords']) < self.max_histogram_samples:
                        self.event_stats['x_coords'].extend(x.tolist())
                        self.event_stats['y_coords'].extend(y.tolist())
                        self.event_stats['polarities'].extend(p.tolist())
                    
                    # Compute time span for this sample
                    if num_events > 0:
                        time_span = float(t[-1] - t[0])
                        self.event_stats['time_spans'].append(time_span)
                    
                    # Count polarities
                    unique_p, counts = np.unique(p, return_counts=True)
                    for pol, count in zip(unique_p, counts):
                        pol_int = int(pol)
                        self.event_stats['polarity_counts'][pol_int] = \
                            self.event_stats['polarity_counts'].get(pol_int, 0) + int(count)
                else:
                    self._log(f"Warning: Expected format 'events/x,y,t,p' not found in {event_path}")
                    self._log(f"Available keys: {list(f.keys())}")
                
        except Exception as e:
            self._log(f"Error processing {event_path}: {e}")
    
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
            flow_dir = subdir / 'forward_flow'
            if flow_dir.exists():
                flow_files = sorted(flow_dir.glob('*.npy'))
                for flow_file in flow_files:
                    self.analyze_flow_file(flow_file)
                    flow_count += 1
            
            # Process event files
            event_dir = subdir / 'events_left'
            if event_dir.exists():
                event_files = list(event_dir.glob('*.h5'))
                for event_file in event_files:
                    self.analyze_event_file(event_file)
                    event_count += 1
        
        # Initialize histograms based on observed range
        self._initialize_histograms()
        
        # PASS 2: Accumulate histogram data
        self._log("\nPass 2: Building histograms from all pixels...")
        for subdir in tqdm(subdirs, desc="Pass 2"):
            # Process flow files
            flow_dir = subdir / 'forward_flow'
            if flow_dir.exists():
                flow_files = sorted(flow_dir.glob('*.npy'))
                for flow_file in flow_files:
                    self.analyze_flow_file(flow_file)
        
        # PASS 3: Event-based valid mask analysis
        self._log("\nPass 3: Analyzing with event-based valid masks...")
        event_mask_count = 0
        for subdir in tqdm(subdirs, desc="Pass 3"):
            flow_dir = subdir / 'forward_flow'
            event_dir = subdir / 'events_left'
            
            if flow_dir.exists() and event_dir.exists():
                flow_files = sorted(flow_dir.glob('*.npy'))
                event_files = sorted(event_dir.glob('*.h5'))
                
                # Create mapping from frame index to event file
                event_map = {}
                for event_file in event_files:
                    # Extract frame index from filename (e.g., 000001.h5 -> 1)
                    # Skip files that don't have numeric names
                    try:
                        frame_idx = int(event_file.stem)
                        event_map[frame_idx] = event_file
                    except ValueError:
                        # Skip non-numeric filenames
                        continue
                
                # Process each flow file with corresponding events
                for flow_file in flow_files:
                    # Flow file i represents motion from frame i to i+1
                    try:
                        flow_idx = int(flow_file.stem)
                    except ValueError:
                        # Skip non-numeric filenames
                        continue
                    
                    # Get event file for frame i+1 (destination frame)
                    event_curr = event_map.get(flow_idx + 1)
                    event_prev = event_map.get(flow_idx)
                    
                    if event_curr:
                        self.analyze_flow_with_event_mask(flow_file, event_curr, event_prev)
                        event_mask_count += 1
        
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
            # Flow magnitude histogram
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Overall magnitude distribution (from all pixels)
            mag_bins = self.histogram_data['magnitude']['bins']
            mag_counts = self.histogram_data['magnitude']['counts']
            mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2
            axes[0].bar(mag_centers, mag_counts, width=np.diff(mag_bins), edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Flow Magnitude')
            axes[0].set_ylabel('Frequency (log scale)')
            axes[0].set_title(f'Flow Magnitude Distribution ({self.flow_stats["total_pixels"]:,} pixels)')
            axes[0].set_yscale('log')
            axes[0].grid(True, alpha=0.3)
            
            # U component distribution
            u_bins = self.histogram_data['u']['bins']
            u_counts = self.histogram_data['u']['counts']
            u_centers = (u_bins[:-1] + u_bins[1:]) / 2
            axes[1].bar(u_centers, u_counts, width=np.diff(u_bins), edgecolor='black', alpha=0.7, color='blue')
            axes[1].set_xlabel('Flow U Component')
            axes[1].set_ylabel('Frequency (log scale)')
            axes[1].set_title('Flow U Component Distribution')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
            
            # V component distribution
            v_bins = self.histogram_data['v']['bins']
            v_counts = self.histogram_data['v']['counts']
            v_centers = (v_bins[:-1] + v_bins[1:]) / 2
            axes[2].bar(v_centers, v_counts, width=np.diff(v_bins), edgecolor='black', alpha=0.7, color='green')
            axes[2].set_xlabel('Flow V Component')
            axes[2].set_ylabel('Frequency (log scale)')
            axes[2].set_title('Flow V Component Distribution')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            flow_hist_path = output_dir / 'flow_histograms.png'
            plt.savefig(flow_hist_path, dpi=300, bbox_inches='tight')
            self._log(f"Flow histograms saved to: {flow_hist_path}")
            plt.close()
            
            # Flow direction analysis - polar histogram
            if self.flow_stats['direction_counts'].sum() > 0:
                fig = plt.figure(figsize=(12, 10))
                
                # Polar histogram
                ax1 = plt.subplot(2, 2, 1, projection='polar')
                direction_centers = (self.flow_stats['direction_bins'][:-1] + self.flow_stats['direction_bins'][1:]) / 2
                width = np.diff(self.flow_stats['direction_bins'])[0]
                bars = ax1.bar(direction_centers, self.flow_stats['direction_counts'], 
                              width=width, alpha=0.7, edgecolor='black')
                ax1.set_title('Flow Direction Distribution (Polar)', pad=20)
                ax1.set_theta_zero_location('E')  # 0 degrees = right
                ax1.set_theta_direction(1)  # Counter-clockwise
                
                # Linear histogram
                ax2 = plt.subplot(2, 2, 2)
                direction_degrees = np.rad2deg(direction_centers)
                ax2.bar(direction_degrees, self.flow_stats['direction_counts'], 
                       width=np.rad2deg(width), alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Flow Direction (degrees)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Flow Direction Distribution')
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim(-180, 180)
                
                # Quadrant distribution
                ax3 = plt.subplot(2, 2, 3)
                quad_names = ['Right', 'Left', 'Up', 'Down']
                quad_colors = ['red', 'blue', 'green', 'orange']
                quad_counts = [self.flow_stats['quadrant_counts'][k.lower()] for k in quad_names]
                bars = ax3.bar(quad_names, quad_counts, color=quad_colors, alpha=0.7, edgecolor='black')
                ax3.set_ylabel('Pixel Count')
                ax3.set_title('Flow Quadrant Distribution')
                ax3.grid(True, alpha=0.3, axis='y')
                # Add percentage labels
                total = sum(quad_counts)
                if total > 0:
                    for bar, count in zip(bars, quad_counts):
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height,
                               f'{100*count/total:.1f}%',
                               ha='center', va='bottom')
                
                # Event-masked direction comparison (if available)
                ax4 = plt.subplot(2, 2, 4)
                if self.event_mask_stats['direction_counts'].sum() > 0:
                    # Compare all-pixels vs event-masked directions
                    x = np.arange(len(quad_names))
                    width_bar = 0.35
                    
                    all_quad = [self.flow_stats['quadrant_counts'][k.lower()] for k in quad_names]
                    event_quad = [self.event_mask_stats['quadrant_counts'][k.lower()] for k in quad_names]
                    
                    # Normalize to percentages
                    all_total = sum(all_quad)
                    event_total = sum(event_quad)
                    all_pct = [100*c/all_total if all_total > 0 else 0 for c in all_quad]
                    event_pct = [100*c/event_total if event_total > 0 else 0 for c in event_quad]
                    
                    ax4.bar(x - width_bar/2, all_pct, width_bar, label='All Pixels', alpha=0.7)
                    ax4.bar(x + width_bar/2, event_pct, width_bar, label='Event-Masked', alpha=0.7)
                    ax4.set_ylabel('Percentage (%)')
                    ax4.set_title('Quadrant Distribution Comparison')
                    ax4.set_xticks(x)
                    ax4.set_xticklabels(quad_names)
                    ax4.legend()
                    ax4.grid(True, alpha=0.3, axis='y')
                else:
                    ax4.text(0.5, 0.5, 'Event-masked data not available',
                           ha='center', va='center', transform=ax4.transAxes)
                    ax4.axis('off')
                
                plt.tight_layout()
                direction_path = output_dir / 'flow_direction_analysis.png'
                plt.savefig(direction_path, dpi=300, bbox_inches='tight')
                self._log(f"Flow direction analysis saved to: {direction_path}")
                plt.close()
            
            # Per-sample statistics
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            axes[0, 0].hist(self.flow_stats['per_sample_mean'], bins=50, edgecolor='black', alpha=0.7)
            axes[0, 0].set_xlabel('Mean Flow Magnitude')
            axes[0, 0].set_ylabel('Number of Samples (log scale)')
            axes[0, 0].set_title('Distribution of Per-Sample Mean Flow')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].hist(self.flow_stats['per_sample_max'], bins=50, edgecolor='black', alpha=0.7, color='red')
            axes[0, 1].set_xlabel('Max Flow Magnitude')
            axes[0, 1].set_ylabel('Number of Samples (log scale)')
            axes[0, 1].set_title('Distribution of Per-Sample Max Flow')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].hist(self.flow_stats['per_sample_min'], bins=50, edgecolor='black', alpha=0.7, color='green')
            axes[1, 0].set_xlabel('Min Flow Magnitude')
            axes[1, 0].set_ylabel('Number of Samples (log scale)')
            axes[1, 0].set_title('Distribution of Per-Sample Min Flow')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].hist(self.flow_stats['per_sample_std'], bins=50, edgecolor='black', alpha=0.7, color='purple')
            axes[1, 1].set_xlabel('Std Dev Flow Magnitude')
            axes[1, 1].set_ylabel('Number of Samples (log scale)')
            axes[1, 1].set_title('Distribution of Per-Sample Flow Std Dev')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            per_sample_path = output_dir / 'flow_per_sample_stats.png'
            plt.savefig(per_sample_path, dpi=300, bbox_inches='tight')
            self._log(f"Per-sample flow statistics saved to: {per_sample_path}")
            plt.close()
        
        # Event-masked statistics comparison
        if self.event_mask_stats['sample_count'] > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Event coverage distribution
            if self.event_mask_stats['per_sample_coverage']:
                coverage = np.array(self.event_mask_stats['per_sample_coverage']) * 100
                axes[0, 0].hist(coverage, bins=50, edgecolor='black', alpha=0.7, color='cyan')
                axes[0, 0].set_xlabel('Event Coverage (%)')
                axes[0, 0].set_ylabel('Number of Samples')
                axes[0, 0].set_title('Distribution of Event Coverage per Sample')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].axvline(np.mean(coverage), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(coverage):.1f}%')
                axes[0, 0].legend()
            
            # Mean flow comparison
            if self.event_mask_stats['per_sample_mean'] and self.flow_stats['per_sample_mean']:
                axes[0, 1].hist(self.flow_stats['per_sample_mean'], bins=50, alpha=0.5, 
                              label='All Pixels', edgecolor='black')
                axes[0, 1].hist(self.event_mask_stats['per_sample_mean'], bins=50, alpha=0.5,
                              label='Event-Masked', edgecolor='black')
                axes[0, 1].set_xlabel('Mean Flow Magnitude')
                axes[0, 1].set_ylabel('Number of Samples')
                axes[0, 1].set_title('Mean Flow: All vs Event-Masked')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Max flow comparison
            if self.event_mask_stats['per_sample_max'] and self.flow_stats['per_sample_max']:
                axes[1, 0].hist(self.flow_stats['per_sample_max'], bins=50, alpha=0.5,
                              label='All Pixels', edgecolor='black')
                axes[1, 0].hist(self.event_mask_stats['per_sample_max'], bins=50, alpha=0.5,
                              label='Event-Masked', edgecolor='black')
                axes[1, 0].set_xlabel('Max Flow Magnitude')
                axes[1, 0].set_ylabel('Number of Samples')
                axes[1, 0].set_title('Max Flow: All vs Event-Masked')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Scatter: event coverage vs mean flow
            if self.event_mask_stats['per_sample_coverage'] and self.event_mask_stats['per_sample_mean']:
                coverage = np.array(self.event_mask_stats['per_sample_coverage']) * 100
                mean_flow = np.array(self.event_mask_stats['per_sample_mean'])
                axes[1, 1].scatter(coverage, mean_flow, alpha=0.3, s=10)
                axes[1, 1].set_xlabel('Event Coverage (%)')
                axes[1, 1].set_ylabel('Mean Flow Magnitude')
                axes[1, 1].set_title('Event Coverage vs Mean Flow')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            event_mask_path = output_dir / 'event_masked_comparison.png'
            plt.savefig(event_mask_path, dpi=300, bbox_inches='tight')
            self._log(f"Event-masked comparison saved to: {event_mask_path}")
            plt.close()
        
        if self.event_stats['events_per_sample']:
            # Event histograms - 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Events per sample
            events_per_sample = np.array(self.event_stats['events_per_sample'])
            axes[0, 0].hist(events_per_sample, bins=50, edgecolor='black', alpha=0.7)
            axes[0, 0].set_xlabel('Number of Events')
            axes[0, 0].set_ylabel('Number of Samples (log scale)')
            axes[0, 0].set_title('Distribution of Events Per Sample')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
            
            # X coordinate distribution
            if self.event_stats['x_coords']:
                x_coords = np.array(self.event_stats['x_coords'])
                axes[0, 1].hist(x_coords, bins=50, edgecolor='black', alpha=0.7, color='blue')
                axes[0, 1].set_xlabel('X Coordinate (pixels)')
                axes[0, 1].set_ylabel('Frequency (log scale)')
                axes[0, 1].set_title(f'Event X Coordinate Distribution ({len(x_coords):,} sampled)')
                axes[0, 1].set_yscale('log')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Y coordinate distribution
            if self.event_stats['y_coords']:
                y_coords = np.array(self.event_stats['y_coords'])
                axes[1, 0].hist(y_coords, bins=50, edgecolor='black', alpha=0.7, color='green')
                axes[1, 0].set_xlabel('Y Coordinate (pixels)')
                axes[1, 0].set_ylabel('Frequency (log scale)')
                axes[1, 0].set_title(f'Event Y Coordinate Distribution ({len(y_coords):,} sampled)')
                axes[1, 0].set_yscale('log')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Time span distribution
            if self.event_stats['time_spans']:
                time_spans = np.array(self.event_stats['time_spans'])
                axes[1, 1].hist(time_spans, bins=50, edgecolor='black', alpha=0.7, color='orange')
                axes[1, 1].set_xlabel('Time Span (microseconds)')
                axes[1, 1].set_ylabel('Number of Samples (log scale)')
                axes[1, 1].set_title('Distribution of Time Spans Per Sample')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            events_hist_path = output_dir / 'event_statistics_histograms.png'
            plt.savefig(events_hist_path, dpi=300, bbox_inches='tight')
            self._log(f"Event histograms saved to: {events_hist_path}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze blink_sim dataset statistics (memory-efficient)')
    parser.add_argument('--dataset-path', type=str, 
                        default='./output/train_set',
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
