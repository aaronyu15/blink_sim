import os
import h5py
import glob
import argparse
import shutil
import numpy as np
from matplotlib import pyplot as plt
import sys
import json
import cv2
import re
import tqdm
import torch
import torch.nn.functional as F
from src.utils import img2video
from src.flow_viz import flow_to_image

default_rgb_keys = ["blur", "normals", "diffuse", "nocs"]
default_hdr_keys = ["colors"]
default_flow_keys = ["forward_flow", "backward_flow"]
default_segmap_keys = ["segmap", ".*_segmaps"]
default_segcolormap_keys = ["segcolormap"]
default_depth_keys = ["distance", "depth"]
all_default_keys = default_rgb_keys + default_flow_keys + \
    default_segmap_keys + default_segcolormap_keys + default_depth_keys
default_depth_max = 20


def key_matches(key, patterns, return_index=False):
    for p, pattern in enumerate(patterns):
        if re.fullmatch(pattern, key):
            return (True, p) if return_index else True

    return (False, None) if return_index else False


def vis_data(
        key, data, full_hdf5_data=None, file_label="", rgb_keys=None,
        hdr_keys=None, flow_keys=None, segmap_keys=None, segcolormap_keys=None,
        depth_keys=None, depth_max=default_depth_max, save_to_file=None):
    if rgb_keys is None:
        rgb_keys = default_rgb_keys[:]
    if hdr_keys is None:
        hdr_keys = default_hdr_keys[:]
    if flow_keys is None:
        flow_keys = default_flow_keys[:]
    if segmap_keys is None:
        segmap_keys = default_segmap_keys[:]
    if segcolormap_keys is None:
        segcolormap_keys = default_segcolormap_keys[:]
    if depth_keys is None:
        depth_keys = default_depth_keys[:]

    # If key is valid and does not contain segmentation data, create figure and add title
    if key_matches(key, flow_keys + rgb_keys + hdr_keys + depth_keys):
        plt.figure()
        plt.title("{} in {}".format(key, file_label))

    try:
        if key_matches(key, flow_keys):
            try:
                # This import here is ugly, but else everytime someone uses this script it demands opencv and the progressbar
                sys.path.append(os.path.join(os.path.dirname(__file__)))
                # from utils import flow_to_image
            except ImportError:
                raise ImportError(
                    "Using .hdf5 containers, which contain flow images needs opencv-python and progressbar "
                    "to be installed!")

            # Visualize optical flow
            if save_to_file is None:
                plt.imshow(flow_to_image(data))
            else:
                flow_data = flow_to_image(data)
                plt.imsave(save_to_file, flow_data)
                # try:
                #     plt.imsave(save_to_file, flow_to_image(data), cmap='jet')
                # except:
                #     import pdb; pdb.set_trace()
        elif key_matches(key, segmap_keys):
            # Try to find labels for each channel in the segcolormap
            channel_labels = {}
            _, key_index = key_matches(key, segmap_keys, return_index=True)
            if key_index < len(segcolormap_keys):
                # Check if segcolormap_key for the current segmap key is configured and exists
                segcolormap_key = segcolormap_keys[key_index]
                if full_hdf5_data is not None and segcolormap_key in full_hdf5_data:
                    # Extract segcolormap data
                    segcolormap = json.loads(
                        np.array(full_hdf5_data[segcolormap_key]).tostring())
                    if len(segcolormap) > 0:
                        # Go though all columns, we are looking for channel_* ones
                        for colormap_key, colormap_value in segcolormap[0].items():
                            if colormap_key.startswith("channel_") and colormap_value.isdigit():
                                channel_labels[int(
                                    colormap_value)] = colormap_key[len("channel_"):]

            # Make sure we have three dimensions
            if len(data.shape) == 2:
                data = data[:, :, None]
            # Go through all channels
            for i in range(data.shape[2]):
                # Try to determine label
                if i in channel_labels:
                    channel_label = channel_labels[i]
                else:
                    channel_label = i

                # Visualize channel
                if save_to_file is None:
                    plt.figure()
                    plt.title("{} / {} in {}".format(key,
                              channel_label, file_label))
                    plt.imshow(data[:, :, i], cmap='jet')
                else:
                    if data.shape[2] > 1:
                        filename = save_to_file.replace(
                            ".png", "_" + str(channel_label) + ".png")
                    else:
                        filename = save_to_file
                    plt.imsave(filename, data[:, :, i], cmap='jet')

        elif key_matches(key, depth_keys):
            # Make sure the data has only one channel, otherwise matplotlib will treat it as an rgb image
            if len(data.shape) == 3:
                if data.shape[2] != 1:
                    print(
                        "Warning: The data with key '" + key +
                        "' has more than one channel which would not allow using a jet color map. Therefore only the first channel is visualized.")
                data = data[:, :, 0]

            if save_to_file is None:
                im = plt.imshow(data, cmap='summer', vmax=depth_max)
                plt.colorbar()
            else:
                plt.imsave(save_to_file, data, cmap='summer', vmax=depth_max)
        elif key_matches(key, rgb_keys):
            if save_to_file is None:
                plt.imshow(data)
            else:
                data = np.clip(data, 0, 1)
                plt.imsave(save_to_file, data)
        elif key_matches(key, hdr_keys):
            import imageio
            if save_to_file is None:
                import pdb; pdb.set_trace()
                # plt.imshow(data)
            else:
                save_to_file = save_to_file.replace('png', 'exr')
                imageio.imwrite(save_to_file, data)
        else:
            if save_to_file is None:
                plt.imshow(data)
            else:
                plt.imsave(save_to_file, data)
        plt.close()
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        import pdb
        pdb.set_trace()


def vis_file(path, keys_to_visualize=None, rgb_keys=None, hdr_keys=None,
             flow_keys=None, segmap_keys=None, segcolormap_keys=None,
             depth_keys=None, depth_max=default_depth_max, save_to_path=None):
    if save_to_path is not None and not os.path.exists(save_to_path):
        os.makedirs(save_to_path)

    # Check if file exists
    if os.path.exists(path):
        if os.path.isfile(path):
            with h5py.File(path, 'r') as data:
                # print(path + ": ")

                # Select only a subset of keys if args.keys is given
                if keys_to_visualize is not None:
                    keys = [key for key in data.keys()
                            if key in keys_to_visualize]
                else:
                    keys = [key for key in data.keys()]

                # Visualize every key
                res = []
                for key in keys:
                    value = np.array(data[key])

                    if sum([ele for ele in value.shape]) < 5 or "version" in key:
                        if value.dtype == "|S5":
                            res.append((key, str(value).replace("[", "").replace(
                                "]", "").replace("b'", "").replace("'", "")))
                        else:
                            res.append((key, value))
                    else:
                        res.append((key, value.shape))

                if res:
                    res = ["'{}': {}".format(key, key_res)
                           for key, key_res in res]
                    # print("Keys: " + ', '.join(res))

                for key in keys:
                    value = np.array(data[key])
                    if save_to_path is not None:
                        save_to_file = os.path.join(
                            save_to_path, str(os.path.basename(path)).split('.')
                            [0] + "_" + key + ".png")
                    else:
                        save_to_file = None
                    # Check if it is a stereo image
                    if len(value.shape) >= 3 and value.shape[0] == 2:
                        # Visualize both eyes separately
                        for i, img in enumerate(value):
                            vis_data(
                                key, img, data, os.path.basename(path) +
                                (" (left)" if i == 0 else " (right)"),
                                rgb_keys, hdr_keys, flow_keys, segmap_keys,
                                segcolormap_keys, depth_keys, depth_max,
                                save_to_file)
                    else:
                        vis_data(
                            key, value, data, os.path.basename(path),
                            rgb_keys, hdr_keys, flow_keys, segmap_keys,
                            segcolormap_keys, depth_keys, depth_max,
                            save_to_file)
        else:
            print("The path is not a file")
    else:
        print("The file does not exist: {}".format(path))


def parse_hdf5_to_img_video3(output_dir, mode, size, num_frame, save_hdr_mp4=True):
    if not save_hdr_mp4:
        return
    hdf5_paths = sorted(glob.glob(f"{output_dir}/hdf5/{mode}/*.hdf5"))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(f'{output_dir}/hdr.mp4', fourcc, 10, (size[1], size[0]))
    for i in range(0, num_frame):
        with h5py.File(f'{output_dir}/hdf5/{mode}/{i}.hdf5', 'r') as data:
            img = data['hdr'][:]
            img = (np.clip(img, 0, 1)*255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img)

    video.release()
    cv2.destroyAllWindows()

def parse_hdf5_to_img_video(output_dir):
    hdf5_paths = sorted(glob.glob(f"{output_dir}/hdf5/*.hdf5"))
    for hdf5_path in tqdm.tqdm(hdf5_paths):
        vis_file(
            path=hdf5_path,
            keys_to_visualize=["colors", "blur", "hdr", "forward_flow",
                               "backward_flow", "depth"],
            rgb_keys=["blur", "hdr", "normals", "diffuse", "nocs"],
            hdr_keys=["colors"],
            flow_keys=["forward_flow", "backward_flow"],
            segmap_keys=["segmap", ".*_segmaps", "instance_segmaps"],
            segcolormap_keys=["segcolormap"],
            depth_keys=["distance", "depth"],
            depth_max=80, save_to_path=f"{output_dir}/frames")

    num_f = len(hdf5_paths)
    dir_id = output_dir.split('_')[-1]
    in_dir = f"{output_dir}/frames"

    img2video(in_dir, output_dir, num_f, '_blur', 'blur.mp4')
    # img2video(in_dir, output_dir, num_f, '_depth', 'depth.mp4', dir_id)
    # img2video(in_dir, output_dir, num_f, '_normals', 'normals.mp4', dir_id)
    img2video(in_dir, output_dir, num_f, '_forward_flow', 'forward_flow.mp4')
    # img2video(in_dir, output_dir, num_f, '_backward_flow', 'backward_flow.mp4', dir_id)
    # img2video(in_dir, output_dir, num_f, '_instance_segmaps', 'instance_segmaps.mp4', dir_id)

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def flow_consistency(forward, backward, device="cpu"):
    """
    forward/backward: HxWx2 numpy arrays
    device: "cpu", "cuda", or torch.device(...)
    """
    device = torch.device(device)

    # Optional safety: if someone passes "cuda" but it's not available, fall back.
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    H, W, _ = forward.shape

    coords = coords_grid(1, H, W).to(device=device, dtype=torch.float32).contiguous()

    forward_t = (
        torch.from_numpy(forward)
        .unsqueeze(0)                 # [1,H,W,2]
        .permute(0, 3, 1, 2)          # [1,2,H,W]
        .to(device=device, dtype=torch.float32)
    )

    backward_t = (
        torch.from_numpy(backward)
        .unsqueeze(0)
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
    )

    # Build sampling grid in normalized coords
    grid = (forward_t + coords).permute(0, 2, 3, 1).contiguous()  # [1,H,W,2]
    grid[..., 0] = (grid[..., 0] * 2 - W + 1) / (W - 1)
    grid[..., 1] = (grid[..., 1] * 2 - H + 1) / (H - 1)

    backward_warped = F.grid_sample(
        backward_t, grid, padding_mode="zeros", align_corners=False
    )

    consistency = forward_t + backward_warped                   # [1,2,H,W]
    consistency = consistency[0].permute(1, 2, 0)               # [H,W,2]

    valid = (torch.norm(consistency, dim=2) < 1.0)              # [H,W]
    valid = valid.unsqueeze(2).to(dtype=torch.float32)          # [H,W,1]

    return valid.cpu().numpy()



def _find_event_h5_path(output_dir, event_h5_path=None):
    """
    Try to find the event HDF5 file produced for this sequence.

    If found, frame_event_start/frame_event_end can be written as indices into
    the actual event arrays instead of just high-FPS frame numbers.
    """
    if event_h5_path is not None:
        return event_h5_path if os.path.exists(event_h5_path) else None

    candidates = [
        f"{output_dir}/events.h5",
        f"{output_dir}/events.hdf5",
        f"{output_dir}/event.h5",
        f"{output_dir}/event.hdf5",
        f"{output_dir}/dvs_events.h5",
        f"{output_dir}/dvs_events.hdf5",
        f"{output_dir}/dvs_events/events.h5",
        f"{output_dir}/dvs_events/events.hdf5",
        f"{output_dir}/dvs_events/event.h5",
        f"{output_dir}/dvs_events/event.hdf5",
    ]

    for pattern in [
        f"{output_dir}/dvs_events/*.h5",
        f"{output_dir}/dvs_events/*.hdf5",
        f"{output_dir}/events/*.h5",
        f"{output_dir}/events/*.hdf5",
    ]:
        candidates.extend(sorted(glob.glob(pattern)))

    for p in candidates:
        if os.path.exists(p) and os.path.basename(p) != "flow.h5":
            return p
    return None


def _read_event_timestamps_from_h5(event_h5_path):
    """
    Return the event timestamp vector from an event HDF5 file.

    Supports common layouts such as:
      /t, /ts, /timestamp, /timestamps
      /events/t, /events/ts, /events/timestamp, /events/timestamps
      a compound /events dataset with a t/ts/timestamp field
    """
    preferred_paths = [
        "events/t", "events/ts", "events/time", "events/timestamp", "events/timestamps",
        "event/t", "event/ts", "event/time", "event/timestamp", "event/timestamps",
        "t", "ts", "time", "timestamp", "timestamps",
    ]

    with h5py.File(event_h5_path, "r") as hf:
        for key in preferred_paths:
            if key in hf:
                arr = np.asarray(hf[key])
                if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                    return arr

        found = []

        def visitor(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            base = name.split("/")[-1].lower()
            if base in {"t", "ts", "time", "timestamp", "timestamps"}:
                arr = np.asarray(obj)
                if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                    found.append(arr)
                    return

            # Compound dataset case, for example /events with fields x,y,t,p.
            if obj.dtype.names:
                lower_fields = {field.lower(): field for field in obj.dtype.names}
                for alias in ("t", "ts", "time", "timestamp", "timestamps"):
                    if alias in lower_fields:
                        arr = np.asarray(obj[lower_fields[alias]])
                        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                            found.append(arr)
                            return

        hf.visititems(visitor)

    if not found:
        return None
    return found[0]


def _event_timestamp_scale_to_us(event_t, expected_duration_s):
    """
    Infer how to convert event-file timestamps to microseconds.

    The returned value is the multiplier such that:
        timestamp_us = event_t * scale_to_us

    Float event timestamps are usually seconds. Integer event timestamps are
    commonly microseconds, but this also handles millisecond/nanosecond-looking
    magnitudes.
    """
    if event_t is None or len(event_t) == 0:
        return None

    arr = np.asarray(event_t)
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    if finite.size == 0:
        return None

    t0 = float(finite[0])
    t1 = float(finite[-1])
    span = max(0.0, t1 - t0)
    expected_duration_s = max(float(expected_duration_s), 1e-9)

    if np.issubdtype(arr.dtype, np.floating):
        # Most DVS/event simulators store floating timestamps in seconds.
        return 1e6

    # Integer timestamps: infer from the approximate sequence duration.
    # Use loose thresholds because events may start after 0 or end before the last frame.
    if span <= expected_duration_s * 20.0:
        return 1e6       # seconds stored as integer, rare but harmless
    if span <= expected_duration_s * 20.0 * 1e3:
        return 1e3       # milliseconds
    if span <= expected_duration_s * 20.0 * 1e6:
        return 1.0       # microseconds
    return 1e-3          # nanoseconds


def _event_indices_from_timestamps(event_t, event_start_us, event_end_us, expected_duration_s):
    """
    Convert flow interval timestamps in microseconds to [start,end) event indices.
    Returns None if the event timestamp vector cannot be used safely.
    """
    if event_t is None or len(event_t) == 0:
        return None

    event_t = np.asarray(event_t)
    if event_t.ndim != 1 or not np.issubdtype(event_t.dtype, np.number):
        return None
    if len(event_t) > 1 and np.any(np.diff(event_t) < 0):
        print("[flow.h5] Warning: event timestamps are not sorted; falling back to event-frame indices.")
        return None

    scale_to_us = _event_timestamp_scale_to_us(event_t, expected_duration_s)
    if scale_to_us is None:
        return None

    # Convert stored event timestamps into microseconds, then search with the
    # absolute flow interval timestamps. This keeps flow.h5 independent of any
    # trim/offset metadata.
    event_t_us = event_t.astype(np.float64) * float(scale_to_us)
    start_idx = np.searchsorted(event_t_us, event_start_us.astype(np.float64), side="left")
    end_idx = np.searchsorted(event_t_us, event_end_us.astype(np.float64), side="left")

    return start_idx.astype(np.uint64), end_idx.astype(np.uint64), scale_to_us

def parse_hdf5_to_flow_dataset(
        output_dir, nFrames, width, height, save_hdr=True,
        rgb_fps=30, event_fps=300, trim_initial_frames=0,
        event_h5_path=None):
    """
    Export a flow.h5 file whose metadata is directly aligned to the event stream.

    The generated flow.h5 contains root-level datasets:

      forward_flow          [N,H,W,2] float32
      valid                 [N,H,W,1] float32
      frame_event_start     [N] uint64
      frame_event_end       [N] uint64
      event_start           [N] uint64, absolute timestamp in microseconds
      event_end             [N] uint64, absolute timestamp in microseconds

    For flow sample k exported from original RGB frame i, the interval is:
        [i / rgb_fps, (i + 1) / rgb_fps)

    This means trim_initial_frames is already baked into event_start/event_end
    and frame_event_start/frame_event_end. No separate offset field is required.
    """
    num = len(glob.glob(f"{output_dir}/hdf5/rgb_and_flow/*.hdf5"))
    assert nFrames <= num

    if save_hdr:
        os.system(f'mkdir -p {output_dir}/hdr')
    os.system(f'mkdir -p {output_dir}/forward_flow')

    exported_frames = max(1, int(nFrames) - int(trim_initial_frames))
    flow_count = max(0, exported_frames - 1)

    flow_forward = np.zeros((flow_count, height, width, 2), dtype=np.float32)
    flow_valid = np.zeros((flow_count, height, width, 1), dtype=np.float32)

    # Absolute timestamps in the event-stream timebase, expressed in microseconds.
    # These already include trim_initial_frames, so the first exported flow sample
    # does NOT start at zero unless trim_initial_frames == 0.
    event_start = np.zeros((flow_count,), dtype=np.uint64)
    event_end = np.zeros((flow_count,), dtype=np.uint64)

    # These will become true event-array indices if an event HDF5 timestamp array
    # is available. Otherwise, they fall back to absolute high-FPS event-frame
    # indices, which are still offset-free with respect to trim_initial_frames.
    frame_event_start = np.zeros((flow_count,), dtype=np.uint64)
    frame_event_end = np.zeros((flow_count,), dtype=np.uint64)

    for i in range(int(trim_initial_frames), int(nFrames) - 1):
        export_idx = i - int(trim_initial_frames)

        hdf5_path = f"{output_dir}/hdf5/rgb_and_flow/{i}.hdf5"
        hdf5_path_next = f"{output_dir}/hdf5/rgb_and_flow/{i + 1}.hdf5"

        with h5py.File(hdf5_path, 'r') as data:
            forward = data['forward_flow'][:].astype(np.float32)

            if save_hdr:
                hdr = data['blur'][:]
                np.save(f'{output_dir}/hdr/{i:06d}.npy', hdr)

        with h5py.File(hdf5_path_next, 'r') as data_next:
            backward = data_next['backward_flow'][:].astype(np.float32)

        valid = flow_consistency(forward, backward)

        # Keep the old per-frame .npy side output for compatibility with any
        # existing visualization/debug code, but the final dataset is flow.h5.
        flow_image = np.concatenate([forward, valid], axis=2)
        np.save(f'{output_dir}/forward_flow/{export_idx:06d}.npy', flow_image)

        flow_forward[export_idx] = forward
        flow_valid[export_idx] = valid.astype(np.float32)

        # Absolute, untrimmed timeline. No later offset correction needed.
        event_start[export_idx] = int(round((i / float(rgb_fps)) * 1e6))
        event_end[export_idx] = int(round(((i + 1) / float(rgb_fps)) * 1e6))

        # Offset-free event-input frame indices. These are overwritten below with
        # actual event-array indices if an event HDF5 timestamp vector is found.
        frame_event_start[export_idx] = int(round(i * float(event_fps) / float(rgb_fps)))
        frame_event_end[export_idx] = int(round((i + 1) * float(event_fps) / float(rgb_fps)))

    # If the event HDF5 file already exists, make frame_event_start/end true
    # indices into the event arrays by searching the event timestamp vector.
    found_event_h5 = _find_event_h5_path(output_dir, event_h5_path=event_h5_path)
    event_timestamp_scale_to_us = None
    if found_event_h5 is not None and flow_count > 0:
        event_t = _read_event_timestamps_from_h5(found_event_h5)
        expected_duration_s = float(nFrames) / float(rgb_fps)
        idx_info = _event_indices_from_timestamps(
            event_t, event_start, event_end, expected_duration_s
        )
        if idx_info is not None:
            frame_event_start, frame_event_end, event_timestamp_scale_to_us = idx_info
            print(f"[flow.h5] Using event indices from: {found_event_h5}")
        else:
            print("[flow.h5] Warning: could not read usable event timestamps; "
                  "frame_event_start/end are high-FPS event-frame indices.")
    elif flow_count > 0:
        print("[flow.h5] Warning: event HDF5 file not found yet; "
              "frame_event_start/end are high-FPS event-frame indices.")

    with h5py.File(f'{output_dir}/flow.h5', 'w') as hf:
        hf.create_dataset('flow/forward', data=flow_forward, compression='gzip', compression_opts=4)
        hf.create_dataset('flow/valid', data=flow_valid, compression='gzip', compression_opts=4)
        hf.create_dataset('flow/frame_event_start', data=frame_event_start, compression='gzip', compression_opts=4)
        hf.create_dataset('flow/frame_event_end', data=frame_event_end, compression='gzip', compression_opts=4)
        hf.create_dataset('flow/event_start', data=event_start, compression='gzip', compression_opts=4)
        hf.create_dataset('flow/event_end', data=event_end, compression='gzip', compression_opts=4)

        hf['flow/forward'].attrs['description'] = 'Forward optical flow from RGB frame i to i+1.'
        hf['flow/valid'].attrs['description'] = 'Forward/backward consistency mask.'
        hf['flow/frame_event_start'].attrs['description'] = (
            'Start index into the event HDF5 event arrays when event timestamps were found; '
            'otherwise absolute high-FPS event-frame index.'
        )
        hf['flow/frame_event_end'].attrs['description'] = (
            'Exclusive end index into the event HDF5 event arrays when event timestamps were found; '
            'otherwise absolute high-FPS event-frame index.'
        )
        hf['flow/event_start'].attrs['unit'] = 'microseconds'
        hf['flow/event_end'].attrs['unit'] = 'microseconds'
        hf.attrs['rgb_fps'] = float(rgb_fps)
        hf.attrs['event_fps'] = float(event_fps)


    forward_flow_dir = f'{output_dir}/forward_flow'
    if os.path.isdir(forward_flow_dir):
        shutil.rmtree(forward_flow_dir)

def cli():
    parser = argparse.ArgumentParser("Script to visualize hdf5 files")

    parser.add_argument('hdf5_paths', nargs='+', help='Path to hdf5 file/s')
    parser.add_argument(
        '--keys', nargs='+',
        help='Keys that should be visualized. If none is given, all keys are visualized.',
        default=all_default_keys)
    parser.add_argument(
        '--rgb_keys', nargs='+',
        help='Keys that should be interpreted as rgb data.',
        default=default_rgb_keys)
    parser.add_argument(
        '--hdr_keys', nargs='+',
        help='Keys that should be interpreted as hdr data.',
        default=default_hdr_keys)
    parser.add_argument(
        '--flow_keys', nargs='+',
        help='Keys that should be interpreted as optical flow data.',
        default=default_flow_keys)
    parser.add_argument(
        '--segmap_keys', nargs='+',
        help='Keys that should be interpreted as segmentation data.',
        default=default_segmap_keys)
    parser.add_argument(
        '--segcolormap_keys', nargs='+',
        help='Keys that point to the segmentation color maps corresponding to the configured segmap_keys.',
        default=default_segcolormap_keys)
    parser.add_argument(
        '--depth_keys', nargs='+',
        help='Keys that contain additional non-RGB data which should be visualized using a jet color map.',
        default=default_depth_keys)
    parser.add_argument('--depth_max', type=float, default=default_depth_max)
    parser.add_argument('--save', default=None, type=str,
                        help='Saves visualizations to file.')

    args = parser.parse_args()

    # Visualize all given files
    for path in args.hdf5_paths:
        vis_file(
            path=path,
            keys_to_visualize=args.keys,
            rgb_keys=args.rgb_keys,
            hdr_keys=args.hdr_keys,
            flow_keys=args.flow_keys,
            segmap_keys=args.segmap_keys,
            segcolormap_keys=args.segcolormap_keys,
            depth_keys=args.depth_keys,
            depth_max=args.depth_max,
            save_to_path=args.save
        )
    if args.save is None:
        plt.show()


if __name__ == "__main__":
    cli()
 