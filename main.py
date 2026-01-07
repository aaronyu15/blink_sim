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
from src.utils import clean_tmp_files, check_blender_result, clean_unfinished
from src.blender.launcher import blender_generate_images_v2
from src.blender.visHdf5Files import parse_hdf5_to_flow_dataset, parse_hdf5_to_img_video3
from src.video2event import make_events

def main(config):
    # num_frames = config['num_frame']
    rgb_fps = config['rgb_image_fps']
    event_fps = config['event_image_fps']
    duration = config['duration']
    duration_from_anim = config.get('duration_from_animation', False)
    base_rgb_frames = int(round(duration * rgb_fps))
    base_event_frames = int(round(duration * event_fps))
    seq_range = config['seq_range']
    train_ratio = config['train_split_ratio']
    size = (config['image_height'], config['image_width'])
   
    mode = 'train'
    save_dir = "output/"
    num_seq = seq_range[1] - seq_range[0]
    for i in range(seq_range[0], seq_range[1]):
        np.random.seed(i)
        random.seed(i)
        if i > num_seq * train_ratio+seq_range[0]:
            mode = 'test'
        output_dir = f'{save_dir}/{mode}/{i:06d}'
        blender_generate_images_v2(config_file, output_dir, mode)
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
            rgb_frames = len(os.listdir(rgb_h5_dir)) if os.path.exists(rgb_h5_dir) else 0
            event_frames = len(os.listdir(evt_h5_dir)) if os.path.exists(evt_h5_dir) else 0
            if rgb_frames == 0 or event_frames == 0:
                print(f"duration_from_animation enabled but could not count frames (rgb: {rgb_frames}, event: {event_frames}). Falling back to config duration.")
                rgb_frames = base_rgb_frames
                event_frames = base_event_frames
            else:
                duration = rgb_frames / float(rgb_fps)
                print(f"Derived duration from animation output: {rgb_frames} rgb frames @ {rgb_fps} fps -> {duration:.3f}s")

        parse_hdf5_to_img_video3(output_dir, 'event_input', size, event_frames)
        parse_hdf5_to_flow_dataset(output_dir, rgb_frames, config['image_width'], config['image_height'])
        evt_np = make_events(output_dir, size, event_frames, event_fps, True, False, num_bins=15)
        clean_tmp_files(output_dir)

        print(f'seq#{i} ok')


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
