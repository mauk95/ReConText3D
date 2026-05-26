import os
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def load_camera_params(transform_json_path):
    with open(transform_json_path, 'r') as f:
        meta = json.load(f)

    fov_x = meta['frames'][0]['camera_angle_x']
    width = 512
    focal_length = width / (2 * np.tan(fov_x / 2))

    intrinsics = {
        'fx': focal_length,
        'fy': focal_length,
        'cx': width / 2,
        'cy': width / 2
    }
    offset = np.array(meta.get('offset', [0.0, 0.0, 0.0]))
    scale = meta.get('scale', 1.0)

    return intrinsics, offset, scale

def load_depth_image(path):
    depth = Image.open(path)
    depth = np.array(depth).astype(np.float32)

    if depth.max() > 10:
        depth = depth / 65535.0 if depth.max() > 255 else depth / 255.0

    return depth

def depth_to_pointcloud(depth, intrinsics):
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    # valid = (z > 0) & (z < 10) & np.isfinite(z)
    valid = (z > 0) & (z < 1) & np.isfinite(z)  # Adjusted for normalized depth
    u, v, z = u[valid], v[valid], z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=1)
    return points

def farthest_point_sampling(points, num_samples=4000):
    N = len(points)
    if N <= num_samples:
        repeats = num_samples // N + 1
        return np.tile(points, (repeats, 1))[:num_samples]

    fps = [np.random.randint(N)]
    dists = np.full(N, np.inf)

    for _ in range(num_samples - 1):
        current = points[fps[-1]]
        dist = np.linalg.norm(points - current, axis=1)
        dists = np.minimum(dists, dist)
        next_idx = np.argmax(dists)
        fps.append(next_idx)
    return points[fps]

def extract_pointcloud_from_views(args):
    renders_dir, sha256, num_views, output_dir = args
    obj_dir = os.path.join(renders_dir, sha256)
    transform_path = os.path.join(obj_dir, 'transforms.json')
    if not os.path.exists(transform_path):
        print(f"Warning: Transform file not found for {sha256}")
        return

    try:
        intrinsics, offset, scale = load_camera_params(transform_path)
        if output_dir:
            save_dir = os.path.join(output_dir, sha256)
            os.makedirs(save_dir, exist_ok=True)

        for i in range(num_views):
            depth_path = os.path.join(obj_dir, f"{i:03d}_depth.png")
            if not os.path.exists(depth_path):
                continue
            if os.path.isfile(os.path.join(save_dir, f"{i:03d}.npy")):
                continue
            depth = load_depth_image(depth_path)
            pts = depth_to_pointcloud(depth, intrinsics)
            pts = pts * scale + offset
            sampled = farthest_point_sampling(pts, num_samples=4000)

            if output_dir:
                np.save(os.path.join(save_dir, f"{i:03d}.npy"), sampled)
    except:
        print(f"Error processing {sha256}")
        return

def batch_process(renders_dir, eval_split_file, output_dir, num_views=4, num_workers=None):
    df = pd.read_csv(eval_split_file)

    if 'sha256' not in df.columns:
        raise ValueError(
            f"CSV must contain a 'sha256' column. Found columns: {list(df.columns)}"
        )

    sha_list = df['sha256'].tolist()
    print(f"Processing {len(sha_list)} assets...")

    args_list = [
        (renders_dir, sha, num_views, output_dir)
        for sha in sha_list
    ]

    with Pool(processes=num_workers or cpu_count()) as pool:
        list(tqdm(
            pool.imap(extract_pointcloud_from_views, args_list),
            total=len(args_list)
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--renders_dir", type=str, required=True)
    parser.add_argument(
        "--eval_split_file",
        type=str,
        required=True,
        help="CSV file containing evaluation split SHA256s and classes (columns: sha256, class)"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=None)

    args = parser.parse_args()

    batch_process(
        args.renders_dir,
        args.eval_split_file,
        args.output_dir,
        num_views=args.num_views,
        num_workers=args.num_workers
    )