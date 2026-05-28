# python main.py

import os
import yaml
import random
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict
import time


def _apply_runtime_thread_settings(config):
    num_cpu_threads = int(config.get('num_cpu_threads', 1))
    for key in ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_MAX_THREADS"]:
        os.environ[key] = str(num_cpu_threads)

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
    from src.utils import clean_tmp_files, check_blender_result, clean_unfinished
    from src.blender.launcher import blender_generate_images_v2
    from src.blender.visHdf5Files import parse_hdf5_to_flow_dataset, parse_hdf5_to_img_video3
    from src.video2event import make_events

    rgb_fps = config['rgb_image_fps']
    event_fps = config['event_image_fps']
    duration = config['duration']
    clamp_to_anim = config.get('clamp_duration_to_animation', True)
    trim_initial_frames = config.get('trim_initial_rgb_frames', 0)
    trim_initial_event_frames = int(round(trim_initial_frames * float(event_fps) / float(rgb_fps)))
    base_rgb_frames = int(round(duration * rgb_fps))
    base_event_frames = int(round(duration * event_fps))
    train_ratio = config['train_split_ratio']
    size = (config['image_height'], config['image_width'])

    save_dir = "output/"
    jobs = build_jobs(config)
    num_seq = len(jobs)
    train_cut = int(num_seq * train_ratio)

    # --- Resume logic ---
    resume_path = "resume_state.yaml"
    if os.path.exists(resume_path):
        with open(resume_path, 'r') as f:
            resume_state = yaml.safe_load(f) or {}
    else:
        resume_state = {}
    completed_jobs = set(resume_state.get('resume_state', {}).get('completed_jobs', []))
    in_progress_job = resume_state.get('resume_state', {}).get('in_progress_job', None)

    # Clean up any unfinished job from previous crash
    if in_progress_job:
        unfinished_dir = None
        for job in jobs:
            if job['folder_name'] == in_progress_job:
                mode = 'train' if jobs.index(job) < train_cut else 'test'
                unfinished_dir = f"{save_dir}/{mode}/{job['folder_name']}"
                break
        if unfinished_dir and os.path.exists(unfinished_dir):
            print(f"Cleaning up unfinished job: {unfinished_dir}")
            from shutil import rmtree
            rmtree(unfinished_dir, ignore_errors=True)
        resume_state['resume_state']['in_progress_job'] = None
        with open(resume_path, 'w') as f:
            yaml.safe_dump(resume_state, f)

    # Save config info for reference
    resume_state.setdefault('resume_state', {})
    resume_state['resume_state']['active_human_models'] = config.get('active_human_models', [])
    resume_state['resume_state']['active_human_anims'] = config.get('active_human_anims', [])
    resume_state['resume_state']['clips_per_character'] = config.get('clips_per_character', 0)
    resume_state['resume_state']['seq_range'] = config.get('seq_range', [])
    resume_state['resume_state']['last_update'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(resume_path, 'w') as f:
        yaml.safe_dump(resume_state, f)

    for idx, job in enumerate(jobs):
        if job['folder_name'] in completed_jobs:
            print(f"Skipping completed job: {job['folder_name']}")
            continue
        # Mark as in-progress
        resume_state['resume_state']['in_progress_job'] = job['folder_name']
        resume_state['resume_state']['last_update'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(resume_path, 'w') as f:
            yaml.safe_dump(resume_state, f)
        #np.random.seed(idx)
        #random.seed(idx)
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
            # Clear in-progress if failed
            resume_state['resume_state']['in_progress_job'] = None
            with open(resume_path, 'w') as f:
                yaml.safe_dump(resume_state, f)
            continue

        # Derive frame counts from rendered outputs and keep rgb/event streams synchronized.
        rgb_frames = base_rgb_frames
        event_frames = base_event_frames

        rgb_h5_dir = os.path.join(output_dir, 'hdf5', 'rgb_and_flow')
        evt_h5_dir = os.path.join(output_dir, 'hdf5', 'event_input')
        rgb_found = len(glob.glob(os.path.join(rgb_h5_dir, '*.hdf5'))) if os.path.exists(rgb_h5_dir) else 0
        evt_found = len(glob.glob(os.path.join(evt_h5_dir, '*.hdf5'))) if os.path.exists(evt_h5_dir) else 0

        if rgb_found > 0 and evt_found > 0:
            # Start from what was actually rendered.
            rgb_frames = rgb_found
            event_frames = evt_found

            # Optionally cap to config duration for legacy behavior.
            if clamp_to_anim:
                rgb_frames = min(rgb_frames, base_rgb_frames)
                event_frames = min(event_frames, base_event_frames)

            # Enforce paired timing: keep only the common duration across rgb/event fps.
            rgb_from_event = int(np.floor(float(event_frames) * float(rgb_fps) / float(event_fps)))
            rgb_frames = max(1, min(rgb_frames, rgb_from_event if rgb_from_event > 0 else rgb_frames))
            event_from_rgb = int(round(float(rgb_frames) * float(event_fps) / float(rgb_fps)))
            event_frames = max(1, min(event_frames, event_from_rgb if event_from_rgb > 0 else event_frames))

            used_duration = rgb_frames / float(rgb_fps)
            print(
                f"Using synchronized rendered counts: rgb={rgb_frames} (found {rgb_found}), "
                f"event={event_frames} (found {evt_found}), duration={used_duration:.3f}s"
            )
        else:
            print(
                f"Could not infer rendered frame counts (rgb: {rgb_found}, event: {evt_found}); "
                f"falling back to config duration counts rgb={rgb_frames}, event={event_frames}."
            )

        parse_hdf5_to_img_video3(
            output_dir,
            'event_input',
            size,
            event_frames,
            save_hdr_mp4=config.get('save_hdr_mp4', False)
        )
        evt_np = make_events(
            output_dir,
            size,
            event_frames,
            event_fps,
            True,
            False,
            num_bins=15,
            noise_enabled=config.get('event_noise_enabled', False),
            noise_rate=config.get('event_noise_rate', 1000),  # events per second
            trim_initial_frames=trim_initial_event_frames,
        )
        parse_hdf5_to_flow_dataset(
            output_dir,
            rgb_frames,
            config['image_width'],
            config['image_height'],
            save_hdr=config.get('save_hdr', False),
            rgb_fps=rgb_fps,
            event_fps=event_fps,
            trim_initial_frames=trim_initial_frames,
        )
        clean_tmp_files(output_dir)

        # Mark as completed
        completed_jobs.add(job['folder_name'])
        resume_state['resume_state']['completed_jobs'] = sorted(list(completed_jobs))
        resume_state['resume_state']['in_progress_job'] = None
        resume_state['resume_state']['last_update'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(resume_path, 'w') as f:
            yaml.safe_dump(resume_state, f)

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

    _apply_runtime_thread_settings(config)
    main(config)
