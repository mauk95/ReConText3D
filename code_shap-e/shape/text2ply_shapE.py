# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: September 20, 2023
#
# This code is licensed under the MIT License.
# ==============================================================================
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.models.configs import model_from_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget, decode_latent_mesh
import os
import time
import argparse
import random
from IPython import embed
import pandas as pd
import ast
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, type=str, help="path to finetuned model checkpoint (.pth)")
parser.add_argument(
    '--metadata_csv',
    default='',
    type=str,
    help="CSV with columns including 'sha256' and 'captions'"
)
parser.add_argument('--output_dir', default=None, type=str, help="Directory to save generated .ply meshes")
parser.add_argument('--save_name', default='Cap3D_test1_meshes', type=str, help="Fallback run name used if output_dir is not provided")
parser.add_argument('--guidance_scale', default=15.0, type=float, help="Guidance scale for sampling")
parser.add_argument('--karras_steps', default=64, type=int, help="Number of Karras steps for sampling")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
# Load finetuned checkpoint
model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict'])

diffusion = diffusion_from_config(load_config('diffusion'))

batch_size = 1
guidance_scale = float(args.guidance_scale)


df = pd.read_csv(args.metadata_csv)

# Parse 'captions' field which is a stringified list; use the first caption
def parse_first_caption(captions_str):
    try:
        if isinstance(captions_str, str):
            caps = ast.literal_eval(captions_str)
        else:
            caps = captions_str
        if isinstance(caps, list) and len(caps) > 0:
            return caps[0]
    except Exception:
        pass
    return "A 3D model"

df['first_caption'] = df['captions'].apply(parse_first_caption)

outdir = args.output_dir if args.output_dir is not None else './shapE_inference/%s'%(args.save_name)
os.makedirs(outdir, exist_ok=True)

print('start generation')
uids = list(df['sha256'].values)
for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc='Generating meshes')):
    uid = getattr(row, 'sha256')
    prompt = getattr(row, 'first_caption')
    out_path = os.path.join(outdir, '%s.ply'%uid)
    if os.path.exists(out_path):
        continue

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=int(args.karras_steps),
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    with torch.no_grad():
        size = 512

        gen_mesh = decode_latent_mesh(xm, latents).tri_mesh()
        with open(out_path, 'wb') as f:
            gen_mesh.write_ply(f)