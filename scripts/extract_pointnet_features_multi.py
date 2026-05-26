import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.multiprocessing import Pool, set_start_method
from pointnet_model import get_model

def load_point_cloud(pc_path):
    pc = np.load(pc_path)
    pc = pc.T
    return torch.from_numpy(pc).float().unsqueeze(0)

def run_on_device(device_id, asset_dirs, pc_dir, out_dir, ckpt_path, aggregate=True):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_class=40, normal_channel=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    for sha in tqdm(asset_dirs, desc="Extracting PointNet++ features"):
        sha_dir = os.path.join(pc_dir, sha)
        view_files = sorted([f for f in os.listdir(sha_dir) if f.endswith('.npy')])

        view_feats = []

        for view_file in view_files:
            pc_path = os.path.join(sha_dir, view_file)
            pc_tensor = load_point_cloud(pc_path).to(device)

            with torch.no_grad():
                _, features = model(pc_tensor)  # [1, 1024, 1]
                feat_np = features.squeeze().cpu().numpy()  # shape: (1024,)

                view_feats.append(feat_np)

                if not aggregate:
                    # Optional: save per-view feature
                    np.save(os.path.join(out_dir, f"{sha}_{view_file[:-4]}.npy"), feat_np)

        if aggregate and view_feats:
            agg_feat = np.mean(np.stack(view_feats), axis=0)
            
            np.save(os.path.join(out_dir, f"{sha}.npy"), agg_feat)

def run_on_device_multi(device_id, asset_dirs, pc_dir, out_dir, ckpt_path, aggregate=True):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_class=40, normal_channel=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    for sha in tqdm(asset_dirs, desc="Extracting PointNet++ features"):
        sha_dir = os.path.join(pc_dir, sha)
        view_files = sorted([f for f in os.listdir(sha_dir) if f.endswith('.npy')])
        
        if len(view_files) == 0:
            continue

        if os.path.exists(os.path.join(out_dir, f"{sha}.npy")) and aggregate:
            continue

        pcs = []
        for view_file in view_files:
            pc_path = os.path.join(sha_dir, view_file)
            pc_tensor = load_point_cloud(pc_path).squeeze(0)  # shape: [3, 4000]
            pcs.append(pc_tensor)
        
        pc_batch = torch.stack(pcs, dim=0).to(device)  # shape: [N_views, 3, 4000]

        with torch.no_grad():
            _, features = model(pc_batch)  # [N_views, 1024, 1]
            features = features.squeeze(-1).cpu().numpy()  # shape: [N_views, 1024]

        if not aggregate:
            for i, view_file in enumerate(view_files):
                view_name = view_file[:-4]
                np.save(os.path.join(out_dir, f"{sha}_{view_name}.npy"), features[i])
        else:
            agg_feat = np.mean(features, axis=0)  # shape: [1024]
            np.save(os.path.join(out_dir, f"{sha}.npy"), agg_feat)


def split_work(asset_dirs, num_chunks):
    return [asset_dirs[i::num_chunks] for i in range(num_chunks)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pc_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--no_aggregate', action='store_true',
                        help="If set, only saves per-view features, not aggregated")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    asset_dirs = [d for d in os.listdir(args.pc_dir) if os.path.isdir(os.path.join(args.pc_dir, d))]

    chunks = split_work(asset_dirs, args.gpus)
    print(f"Splitting work into {len(chunks)} chunks for {args.gpus} GPUs")

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    with Pool(processes=args.gpus) as pool:
        pool.starmap(run_on_device_multi, [(i, chunks[i], args.pc_dir, args.output_dir, args.ckpt, not args.no_aggregate) for i in range(args.gpus)])

if __name__ == '__main__':
    main()