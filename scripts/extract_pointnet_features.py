import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from scripts_eval.pointnet_model import get_model

def load_point_cloud(pc_path):
    pc = np.load(pc_path)  # shape: [4000, 3]
    if pc.shape[0] != 4000:
        raise ValueError(f"Expected 4000 points, got {pc.shape}")
    pc = pc.T  # shape: [3, 4000]
    return torch.from_numpy(pc).float().unsqueeze(0)  # shape: [1, 3, 4000]

def extract_features(pc_dir, out_dir, model, device, aggregate=True):
    os.makedirs(out_dir, exist_ok=True)
    asset_dirs = [d for d in os.listdir(pc_dir) if os.path.isdir(os.path.join(pc_dir, d))]

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
                print(f"Extracted feature for {sha} from {view_file}: {feat_np.shape}")

                view_feats.append(feat_np)

                if not aggregate:
                    # Optional: save per-view feature
                    np.save(os.path.join(out_dir, f"{sha}_{view_file[:-4]}.npy"), feat_np)

        if aggregate and view_feats:
            agg_feat = np.mean(np.stack(view_feats), axis=0)
            print(f"Aggregated feature for {sha}: {agg_feat.shape}")
            
            np.save(os.path.join(out_dir, f"{sha}.npy"), agg_feat)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pc_dir', type=str, required=True,
                        help="Directory with per-asset subfolders containing view point clouds")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save extracted features")
    parser.add_argument('--ckpt', type=str, required=True,
                        help="Path to pretrained PointNet++ checkpoint")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--no_aggregate', action='store_true',
                        help="If set, only saves per-view features, not aggregated")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = get_model(num_class=40, normal_channel=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    extract_features(args.pc_dir, args.output_dir, model, device, aggregate=not args.no_aggregate)

if __name__ == '__main__':
    main()
