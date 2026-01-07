# python main.py

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"]="1"
from numpy import require
import yaml
import numpy as np
import random
import glob
from pathlib import Path
from collections import defaultdict
from src.utils import clean_tmp_files, check_blender_result, clean_unfinished
from src.blender.launcher import blender_generate_images_v2
from src.blender.visHdf5Files import parse_hdf5_to_flow_dataset, parse_hdf5_to_img_video3
from src.video2event import make_events

def _filter_by_names(paths, allowed_names):
    if not allowed_names:
        return paths
    allowed = set(allowed_names)
    return [p for p in paths if Path(p).stem in allowed]


def build_jobs(config):
    human_model_dir = config.get('human_model_dir', 'data/human_models')
    human_anim_dir = config.get('human_anim_dir', 'data/human_animations')
    model_files = sorted(glob.glob(f"{human_model_dir}/*.fbx"))
    anim_files = sorted(glob.glob(f"{human_anim_dir}/*.fbx"))
    if len(model_files) == 0:
        raise ValueError(f"No FBX models found in {human_model_dir}")
    if len(anim_files) == 0:
        raise ValueError(f"No FBX animations found in {human_anim_dir}")

    active_models = _filter_by_names(model_files, config.get('active_human_models'))
    active_anims = _filter_by_names(anim_files, config.get('active_human_anims'))
    if len(active_models) == 0:
        raise ValueError("No active models after filtering; check active_human_models")
    if len(active_anims) == 0:
        raise ValueError("No active animations after filtering; check active_human_anims")

    seq_range = config.get('seq_range', [0, 1])
    clips_per_character = config.get('clips_per_character', seq_range[1] - seq_range[0])
    if clips_per_character <= 0:
        raise ValueError("clips_per_character must be > 0")

    jobs = []
    pair_version = defaultdict(int)
    for model_path in active_models:
        for clip_idx in range(clips_per_character):
            anim_path = active_anims[clip_idx % len(active_anims)]
            model_base = Path(model_path).stem
            anim_base = Path(anim_path).stem
            pair_key = (model_base, anim_base)
            version = pair_version[pair_key]
            pair_version[pair_key] += 1
            folder_name = f"{model_base}_{anim_base}_{version}"
            jobs.append({
                'model_path': model_path,
                'anim_path': anim_path,
                'folder_name': folder_name,
            })
    return jobs


def main(config):
    rgb_fps = config['rgb_image_fps']
    event_fps = config['event_image_fps']
    duration = config['duration']
    duration_from_anim = config.get('duration_from_animation', False)
    base_rgb_frames = int(round(duration * rgb_fps))
    base_event_frames = int(round(duration * event_fps))
    train_ratio = config['train_split_ratio']
    size = (config['image_height'], config['image_width'])

    save_dir = "output/"
    jobs = build_jobs(config)
    num_seq = len(jobs)
    train_cut = int(num_seq * train_ratio)

    for idx, job in enumerate(jobs):
        np.random.seed(idx)
        random.seed(idx)
        mode = 'train' if idx < train_cut else 'test'
        output_dir = f"{save_dir}/{mode}/{job['folder_name']}"
        os.makedirs(output_dir, exist_ok=True)

        # Write a per-job config with forced model/animation selection and output dir
        job_config = dict(config)
        job_config['forced_model_path'] = job['model_path']
        job_config['forced_animation_path'] = job['anim_path']
        job_config['output_dir'] = output_dir
        job_config['sequence_label'] = job['folder_name']
        job_config_file = os.path.join(output_dir, 'config_job.yaml')
        with open(job_config_file, 'w') as jf:
            yaml.safe_dump(job_config, jf)

        blender_generate_images_v2(job_config_file, output_dir, mode)
        status = check_blender_result(output_dir)
        if not status:
            clean_unfinished(output_dir)
            continue

        # If requested, derive frame counts from rendered outputs (animation length)
        rgb_frames = base_rgb_frames
        event_frames = base_event_frames
        if duration_from_anim:
            rgb_h5_dir = os.path.join(output_dir, 'hdf5', 'rgb_and_flow')
            evt_h5_dir = os.path.join(output_dir, 'hdf5', 'event_input')
            rgb_found = len(os.listdir(rgb_h5_dir)) if os.path.exists(rgb_h5_dir) else 0
            evt_found = len(os.listdir(evt_h5_dir)) if os.path.exists(evt_h5_dir) else 0
            if rgb_found == 0 or evt_found == 0:
                print(f"duration_from_animation enabled but could not count frames (rgb: {rgb_found}, event: {evt_found}). Falling back to config duration.")
            else:
                rgb_frames = min(rgb_found, base_rgb_frames)
                event_frames = min(evt_found, base_event_frames)
                capped_duration = rgb_frames / float(rgb_fps)
                print(f"Derived duration from animation output (capped by config): {rgb_frames} rgb frames @ {rgb_fps} fps -> {capped_duration:.3f}s (max {duration:.3f}s)")

        parse_hdf5_to_img_video3(
            output_dir,
            'event_input',
            size,
            event_frames,
            save_hdr_mp4=config.get('save_hdr_mp4', False)
        )
        parse_hdf5_to_flow_dataset(
            output_dir,
            rgb_frames,
            config['image_width'],
            config['image_height'],
            save_hdr=config.get('save_hdr', False)
        )
        evt_np = make_events(output_dir, size, event_frames, event_fps, True, False, num_bins=15)
        clean_tmp_files(output_dir)

        print(f'seq#{idx} ok')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.add_argument('--seq_range', nargs="+", type=int, required=False)
    parser.add_argument('--config', type=str, required=False, default='configs/blinkflow_v1.yaml')
    args = parser.parse_args()

    config_file = args.config
    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if args.seq_range:
        config['seq_range'] = [args.seq_range[0], args.seq_range[1]]
    print('seq_range:', config['seq_range'])

    main(config)
