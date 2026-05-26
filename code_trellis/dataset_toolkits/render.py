import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
import math
from utils import sphere_hammersley_sequence


BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'
print (f'BLENDER_PATH: {BLENDER_PATH}')

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        # os.system('apt-get update')
        # os.system('apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system('apt update && apt install -y libx11-6 libxi6 libxrender1 libsm6 libxext6 libxfixes3 libxkbcommon-x11-0 libxxf86vm1 libgl1')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def _render(file_path, sha256, output_dir, num_views, renders_dir_name='renders', eval_mode=None, save_depth=False):
    output_folder = os.path.join(output_dir, renders_dir_name, sha256)
    
    # Build camera {yaw, pitch, radius, fov}
    if eval_mode == "appearance":
        # 4 views: yaw = 0°, 90°, 180°, 270°, pitch = 30°
        yaws_deg = [0, 90, 180, 270]
        pitch_deg = 30

        yaws = [math.radians(y) for y in yaws_deg]
        pitchs = [math.radians(pitch_deg)] * len(yaws)

    elif eval_mode == "prompt":
        # 8 views: yaw = every 45°, pitch = 30°
        yaws_deg = list(range(0, 360, 45))
        pitch_deg = 30

        yaws = [math.radians(y) for y in yaws_deg]
        pitchs = [math.radians(pitch_deg)] * len(yaws)

    else:
        # Default: Hammersley sampling
        yaws = []
        pitchs = []
        offset = (np.random.rand(), np.random.rand())
        for i in range(num_views):
            y, p = sphere_hammersley_sequence(i, num_views, offset)
            yaws.append(y)
            pitchs.append(p)

    radius = [2] * len(yaws)
    fov = [40 / 180 * np.pi] * len(yaws)
    # radius = [2.0] * len(yaws)
    # fov = [math.radians(40)] * len(yaws)

    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    args = [
        BLENDER_PATH, '-b', '-P', 'dataset_toolkits/blender_script/render.py',
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        '--save_mesh',
    ]

    if save_depth:
        args.append('--save_depth')

    # if eval_mode is not None:
    #     args.append('--eval_mode')

    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    # GLB files are handled by the --object parameter, no need to insert them
    
    call(args, stdout=DEVNULL, stderr=DEVNULL)
    # call(args)  # Enable output for debugging
    
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'sha256': sha256, 'rendered': True}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--renders_dir_name', type=str, default='renders',
                        help='Name of the directory to save renders')
    parser.add_argument('--raw_assets_dir', type=str, default='datasets/Toys4k',
                        help='Root directory for raw assets')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    # added by me for rendering evaluation sets
    parser.add_argument('--eval_mode', type=str, default=None,
                    choices=["appearance", "prompt"],
                    help="Use fixed views for evaluation rendering.")
    parser.add_argument('--save_depth', action='store_true', help='Save the depth maps.')


    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, opt.renders_dir_name), exist_ok=True)
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' in metadata.columns:
            metadata = metadata[metadata['rendered'] == False]
    else:
        if os.path.exists(opt.instances):
            if opt.instances.endswith('.csv'):
                instances_df = pd.read_csv(opt.instances)
                if 'sha256' not in instances_df.columns:
                    raise ValueError(f"CSV must contain a 'sha256' column. Found columns: {list(instances_df.columns)}")
                instances = instances_df['sha256'].astype(str).tolist()
            else:
                with open(opt.instances, 'r') as f:
                    instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, opt.renders_dir_name, sha256, 'transforms.json')):
            records.append({'sha256': sha256, 'rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')
    
    # process objects
    func = partial(_render, output_dir=opt.output_dir, num_views=opt.num_views, renders_dir_name=opt.renders_dir_name, eval_mode=opt.eval_mode, save_depth=opt.save_depth)
    # rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    rendered = dataset_utils.foreach_instance(metadata, opt.raw_assets_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    rendered.to_csv(os.path.join(opt.output_dir, f'rendered_{opt.rank}.csv'), index=False)
