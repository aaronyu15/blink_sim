import os
import subprocess, signal, pickle
import numpy as np
import yaml

def blender_generate_images_v2(config_file, output_dir, mode):
    command = f'blenderproc run src/blender/blender_script.py -config_file {config_file} -output_dir {output_dir} -mode {mode}'
    command_list = command.split(' ')

    env = os.environ.copy()
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    num_cpu_threads = int(config.get('num_cpu_threads', 1))
    for key in ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_MAX_THREADS"]:
        env[key] = str(num_cpu_threads)

    p = subprocess.Popen(command_list, env=env)
    p.wait()


