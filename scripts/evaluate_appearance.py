#!/usr/bin/env python3
"""
Complete TRELLIS appearance evaluation pipeline in a single script.
Performs feature extraction and FD metric calculation with unified logging.

Model Input Specifications:
- Inception-v3: 299x299 input size
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import json
from datetime import datetime
from collections import Counter
from torchvision.models import inception_v3
from torchvision.transforms import InterpolationMode

from metrics import calculate_frechet_distance

CLASSES_BASE = ["airplane", "bicycle", "bottle", "bowl", "cake", "candy", "car", "cat", "cells_battery", "chair", "chicken", "cookie", "cupcake", "deer_moose", "donut", "dragon", "elephant", "fish", "flower", "grapes", "hammer", "helicopter", "helmet", "horse", "ice_cream", "mouse", "motorcycle", "mushroom", "orange", "panda", "pear", "penguin", "piano", "sandwich", "screwdriver", "shark", "sheep", "shoe", "snake", "sofa", "trashcan", "tractor", "train", "tree", "violin"]
CLASSES_NOVEL = ["apple", "ball", "banana", "boat", "bread", "bunny", "bus", "butterfly", "chess_piece", "coin", "cow", "crab", "cup", "dinosaur", "dog", "dolphin", "drum", "fan", "fox", "fridge", "fries", "frog", "glass", "guitar", "hamburger", "hat", "key", "knife", "laptop", "lizard", "monkey", "mug", "octopus", "pencil", "phone", "pig", "pizza", "plate", "radio", "robot", "sink", "spade", "stove", "truck", "whale"]


def collect_rendered_images_with_classes(renders_base_dir, sha256_to_class_dict, num_views=4):
    """Collect paths to all rendered images for given SHA256 list with class information."""
    
    image_paths = []
    image_classes = []
    image_sha256s = []
    missing_sha256 = set()
    available_sha256 = set()
    
    for sha256, class_name in sha256_to_class_dict.items():
        sha256_dir = os.path.join(renders_base_dir, sha256)
        if not os.path.exists(sha256_dir):
            missing_sha256.add(sha256)
            continue
            
        sha256_images = []
        for view_idx in range(num_views):
            img_path = os.path.join(sha256_dir, f"{view_idx:03d}.png")
            if os.path.exists(img_path):
                sha256_images.append(img_path)
            else:
                print(f"Warning: Image not found: {img_path}")
        
        if len(sha256_images) == num_views:
            available_sha256.add(sha256)
            for img_path in sha256_images:
                image_paths.append(img_path)
                image_classes.append(class_name)
                image_sha256s.append(sha256)
        else:
            missing_sha256.add(sha256)
            print(f"Warning: Incomplete views for {sha256}, skipping")

    return image_paths, image_classes, image_sha256s, list(missing_sha256), available_sha256


def extract_features_from_paths(image_paths, model_type, model, transform, device, batch_size=32, type='real'):
    """Extract features from a list of image paths."""
    
    if len(image_paths) == 0:
        print("No image paths provided")
        return None
    
    features = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Extracting {model_type} features"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                image = Image.open(img_path)

                if image.mode == 'RGBA':
                    black_bg = Image.new('RGB', image.size, (0, 0, 0))
                    image = Image.alpha_composite(black_bg.convert('RGBA'), image).convert('RGB')
                else:
                    image = image.convert('RGB')
                
                image = transform(image)
                batch_images.append(image)

            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if len(batch_images) == 0:
            continue
            
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            if model_type == 'inception':
                batch_features = model(batch_tensor)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")
            
            features.append(batch_features.cpu().numpy())
    
    if len(features) == 0:
        return None
        
    return np.concatenate(features, axis=0)


def subsample_features(features, max_samples=5000, seed=42):
    """Subsample features if there are too many."""
    if features.shape[0] > max_samples:
        np.random.seed(seed)
        indices = np.random.choice(features.shape[0], max_samples, replace=False)
        features = features[indices]
        print(f"Subsampled to {max_samples} features")
    return features


def calculate_metrics_from_features_by_class_and_stage(
    real_features,
    generated_features,
    real_classes,
    generated_classes,
    model_type,
    max_samples=5000,
    seed=42
):
    """Calculate FD metrics from feature arrays overall, by stage, and per class."""
    
    print(f"\nCalculating metrics for {model_type} features...")
    print(f"Real features shape: {real_features.shape}")
    print(f"Generated features shape: {generated_features.shape}")
    
    results = {}
    
    print("Calculating overall metrics...")
    real_features_sub = subsample_features(real_features.copy(), max_samples, seed)
    generated_features_sub = subsample_features(generated_features.copy(), max_samples, seed)
    
    mu1, sigma1 = real_features_sub.mean(axis=0), np.cov(real_features_sub, rowvar=False)
    mu2, sigma2 = generated_features_sub.mean(axis=0), np.cov(generated_features_sub, rowvar=False)

    fd_overall = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    results['overall'] = {
        'frechet_distance': fd_overall,
        'real_samples': real_features.shape[0],
        'generated_samples': generated_features.shape[0]
    }
    
    print(f"Overall - FD: {fd_overall:.4f}")
    
    print("\nCalculating stage-specific metrics...")
    stage_results = {}
    
    for stage_name, stage_classes in [('base', CLASSES_BASE), ('novel', CLASSES_NOVEL)]:
        print(f"Processing {stage_name} classes: {len(stage_classes)} classes")
        
        real_stage_indices = [i for i, c in enumerate(real_classes) if c in stage_classes]
        gen_stage_indices = [i for i, c in enumerate(generated_classes) if c in stage_classes]
        
        if len(real_stage_indices) == 0 or len(gen_stage_indices) == 0:
            print(f"  Warning: No samples found for {stage_name}")
            continue
            
        real_stage_features = real_features[real_stage_indices]
        gen_stage_features = generated_features[gen_stage_indices]
        
        print(f"  {stage_name} - Real samples: {len(real_stage_features)}, Generated samples: {len(gen_stage_features)}")
        
        real_stage_sub = subsample_features(real_stage_features.copy(), max_samples // 2, seed)
        gen_stage_sub = subsample_features(gen_stage_features.copy(), max_samples // 2, seed)
        
        mu1_stage, sigma1_stage = real_stage_sub.mean(axis=0), np.cov(real_stage_sub, rowvar=False)
        mu2_stage, sigma2_stage = gen_stage_sub.mean(axis=0), np.cov(gen_stage_sub, rowvar=False)

        try:
            fd_stage = calculate_frechet_distance(mu1_stage, sigma1_stage, mu2_stage, sigma2_stage)

            stage_results[stage_name] = {
                'frechet_distance': fd_stage,
                'real_samples': len(real_stage_features),
                'generated_samples': len(gen_stage_features),
                'classes': sorted([c for c in stage_classes if c in set(real_classes) & set(generated_classes)])
            }
            
            print(f"  {stage_name} - FD: {fd_stage:.4f}")
            
        except Exception as e:
            print(f"  Error calculating metrics for {stage_name}: {e}")
            continue
    
    results['by_stage'] = stage_results
    
    unique_classes = sorted(set(real_classes) & set(generated_classes))
    print(f"\nComputing class-wise metrics for {len(unique_classes)} classes...")
    
    class_results = {}
    
    for class_name in unique_classes:
        print(f"Processing class: {class_name}")
        
        real_class_indices = [i for i, c in enumerate(real_classes) if c == class_name]
        gen_class_indices = [i for i, c in enumerate(generated_classes) if c == class_name]
        
        if len(real_class_indices) == 0 or len(gen_class_indices) == 0:
            print(f"  Warning: No samples found for class {class_name}")
            continue
            
        real_class_features = real_features[real_class_indices]
        gen_class_features = generated_features[gen_class_indices]
        
        print(f"  Real samples: {len(real_class_features)}, Generated samples: {len(gen_class_features)}")
        
        mu1_class, sigma1_class = real_class_features.mean(axis=0), np.cov(real_class_features, rowvar=False)
        mu2_class, sigma2_class = gen_class_features.mean(axis=0), np.cov(gen_class_features, rowvar=False)

        try:
            fd_class = calculate_frechet_distance(mu1_class, sigma1_class, mu2_class, sigma2_class)

            stage = 'base' if class_name in CLASSES_BASE else 'novel'

            class_results[class_name] = {
                'frechet_distance': fd_class,
                'real_samples': len(real_class_features),
                'generated_samples': len(gen_class_features),
                'stage': stage
            }
            
            print(f"  {class_name} ({stage}) - FD: {fd_class:.4f}")
            
        except Exception as e:
            print(f"  Error calculating metrics for class {class_name}: {e}")
            continue
    
    results['per_class'] = class_results
    
    return results


def setup_model_and_transforms(model_type, device):
    """Setup model and transforms for feature extraction."""
    
    if model_type == 'inception':
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval().to(device)
        
        transform = transforms.Compose([
            transforms.Resize((299, 299), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        return model, transform
    
    raise ValueError(f"Unsupported model_type: {model_type}")


def save_results_with_classes_and_stages(
    results_dir,
    model_type,
    results_dict,
    run_info,
    real_features,
    generated_features,
    timestamp
):
    """Save evaluation results with class-wise and stage-specific FD results."""
    
    results = {
        "timestamp": timestamp,
        "model_type": model_type,
        "metrics": results_dict,
        "dataset_info": {
            "real_features_shape": real_features.shape,
            "generated_features_shape": generated_features.shape,
        },
        "run_info": run_info
    }
    
    results_json_path = os.path.join(results_dir, f"evaluation_results_{model_type}.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved JSON results to: {results_json_path}")
    
    results_txt_path = os.path.join(results_dir, f"evaluation_results_{model_type}.txt")
    with open(results_txt_path, 'w') as f:
        f.write(f"TRELLIS Evaluation Results ({model_type} features)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model Type: {model_type}\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"  Real features shape: {real_features.shape}\n")
        f.write(f"  Generated features shape: {generated_features.shape}\n")
        if "dataset_filtering" in run_info:
            f.write(f"  Dataset filtering applied: {run_info['dataset_filtering']['filtering_applied']}\n")
            f.write(f"  Common SHA256s used: {run_info['dataset_filtering']['common_sha256s']}\n")
            f.write(f"  Total eval assets: {run_info['dataset_filtering']['total_eval_assets']}\n")
        f.write("\n")
        
        f.write("Overall Evaluation Metrics:\n")
        f.write(f"  Fréchet Distance (FD): {results_dict['overall']['frechet_distance']:.4f}\n")
        f.write(f"  Real samples: {results_dict['overall']['real_samples']}\n")
        f.write(f"  Generated samples: {results_dict['overall']['generated_samples']}\n\n")
        
        if 'by_stage' in results_dict:
            f.write("Stage-Specific Evaluation Metrics:\n")
            for stage_name, stage_metrics in results_dict['by_stage'].items():
                f.write(f"  {stage_name.upper()}:\n")
                f.write(f"    Fréchet Distance (FD): {stage_metrics['frechet_distance']:.4f}\n")
                f.write(f"    Real samples: {stage_metrics['real_samples']}\n")
                f.write(f"    Generated samples: {stage_metrics['generated_samples']}\n")
                f.write(f"    Classes ({len(stage_metrics['classes'])}): {', '.join(stage_metrics['classes'])}\n\n")
        
        if 'per_class' in results_dict:
            f.write("Per-Class Evaluation Metrics:\n")
            base_classes = [c for c, m in results_dict['per_class'].items() if m.get('stage') == 'base']
            novel_classes = [c for c, m in results_dict['per_class'].items() if m.get('stage') == 'novel']
            
            if base_classes:
                f.write("  Base Classes:\n")
                for class_name in sorted(base_classes):
                    class_metrics = results_dict['per_class'][class_name]
                    f.write(f"    {class_name}:\n")
                    f.write(f"      Fréchet Distance (FD): {class_metrics['frechet_distance']:.4f}\n")
                    f.write(f"      Real samples: {class_metrics['real_samples']}\n")
                    f.write(f"      Generated samples: {class_metrics['generated_samples']}\n")
                f.write("\n")
            
            if novel_classes:
                f.write("  Novel Classes:\n")
                for class_name in sorted(novel_classes):
                    class_metrics = results_dict['per_class'][class_name]
                    f.write(f"    {class_name}:\n")
                    f.write(f"      Fréchet Distance (FD): {class_metrics['frechet_distance']:.4f}\n")
                    f.write(f"      Real samples: {class_metrics['real_samples']}\n")
                    f.write(f"      Generated samples: {class_metrics['generated_samples']}\n")
                f.write("\n")
    
    print(f"Saved text results to: {results_txt_path}")
    
    results_csv_path = os.path.join(results_dir, f"evaluation_results_{model_type}.csv")
    csv_data = []
    
    csv_data.append({
        'timestamp': timestamp,
        'model_type': model_type,
        'category': 'overall',
        'class_or_stage': 'overall',
        'stage': 'both',
        'frechet_distance': results_dict['overall']['frechet_distance'],
        'real_samples': results_dict['overall']['real_samples'],
        'generated_samples': results_dict['overall']['generated_samples']
    })
    
    if 'by_stage' in results_dict:
        for stage_name, stage_metrics in results_dict['by_stage'].items():
            csv_data.append({
                'timestamp': timestamp,
                'model_type': model_type,
                'category': 'by_stage',
                'class_or_stage': stage_name,
                'stage': stage_name,
                'frechet_distance': stage_metrics['frechet_distance'],
                'real_samples': stage_metrics['real_samples'],
                'generated_samples': stage_metrics['generated_samples']
            })
    
    if 'per_class' in results_dict:
        for class_name, class_metrics in results_dict['per_class'].items():
            csv_data.append({
                'timestamp': timestamp,
                'model_type': model_type,
                'category': 'per_class',
                'class_or_stage': class_name,
                'stage': class_metrics.get('stage', 'unknown'),
                'frechet_distance': class_metrics['frechet_distance'],
                'real_samples': class_metrics['real_samples'],
                'generated_samples': class_metrics['generated_samples']
            })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(results_csv_path, index=False)
    print(f"Saved CSV results to: {results_csv_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Complete TRELLIS appearance evaluation pipeline')
    parser.add_argument(
        '--real_renders_dir',
        type=str,
        default='datasets/Toys4k_eval_appearance_real/renders',
        help='Directory containing rendered views of real assets'
    )
    parser.add_argument(
        '--generated_renders_dir',
        type=str,
        default='datasets/Toys4k_eval_appearance_generated/renders',
        help='Directory containing rendered views of generated assets'
    )
    parser.add_argument(
        '--eval_split_file',
        type=str,
        default='toys4k_eval_split_1250.csv',
        help='CSV file containing evaluation split SHA256s and classes'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Base directory to save all evaluation results'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=['inception'],
        default=['inception'],
        help='Feature extraction models to use'
    )
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--num_views', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_name', type=str, default=None)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_dir_name = f"{timestamp}_{args.run_name}_evaluation"
    else:
        run_dir_name = f"{timestamp}_appearance_evaluation"
    
    exp_folder_name = os.path.basename(os.path.dirname(args.generated_renders_dir))
    results_dir = os.path.join(args.output_dir, run_dir_name + "_" + exp_folder_name)
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 80)
    print("TRELLIS APPEARANCE EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"Models to evaluate: {args.models}")
    print(f"Device: {args.device}")
    
    eval_df = pd.read_csv(args.eval_split_file, usecols=['sha256', 'class'])
    
    sha256_to_class = dict(zip(eval_df['sha256'], eval_df['class']))
    eval_sha256s = list(eval_df['sha256'])
    
    print(f"Loaded {len(eval_sha256s)} SHA256s from evaluation split")
    print(f"Found {len(eval_df['class'].unique())} unique classes: {sorted(eval_df['class'].unique())}")
    
    class_counts = eval_df['class'].value_counts()
    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} samples")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "=" * 50)
    print("COLLECTING RENDERED IMAGES")
    print("=" * 50)
    
    print("Collecting real asset image paths...")
    real_image_paths, real_image_classes, real_image_sha256s, missing_real_sha256, available_real_sha256 = collect_rendered_images_with_classes(
        args.real_renders_dir,
        sha256_to_class,
        args.num_views
    )
    
    print("Collecting generated asset image paths...")
    generated_image_paths, generated_image_classes, generated_image_sha256s, missing_generated_sha256, available_generated_sha256 = collect_rendered_images_with_classes(
        args.generated_renders_dir,
        sha256_to_class,
        args.num_views
    )
    
    common_sha256s = available_real_sha256 & available_generated_sha256
    only_real_sha256s = available_real_sha256 - available_generated_sha256
    only_generated_sha256s = available_generated_sha256 - available_real_sha256
    
    print(f"\nDataset overlap analysis:")
    print(f"  Total SHA256s in eval split: {len(eval_sha256s)}")
    print(f"  Available in real dataset: {len(available_real_sha256)}")
    print(f"  Available in generated dataset: {len(available_generated_sha256)}")
    print(f"  Common SHA256s (for fair comparison): {len(common_sha256s)}")
    print(f"  Only in real: {len(only_real_sha256s)}")
    print(f"  Only in generated: {len(only_generated_sha256s)}")
    
    if len(common_sha256s) == 0:
        print("ERROR: No common SHA256s found between real and generated datasets!")
        return
    
    if len(common_sha256s) < len(eval_sha256s) * 0.8:
        print(
            f"WARNING: Only {len(common_sha256s)}/{len(eval_sha256s)} "
            f"({len(common_sha256s) / len(eval_sha256s) * 100:.1f}%) assets available in both datasets"
        )
    
    print("\nFiltering to common SHA256s for fair evaluation...")
    
    filtered_real_paths = []
    filtered_real_classes = []
    for i, sha256 in enumerate(real_image_sha256s):
        if sha256 in common_sha256s:
            filtered_real_paths.append(real_image_paths[i])
            filtered_real_classes.append(real_image_classes[i])
    
    filtered_generated_paths = []
    filtered_generated_classes = []
    for i, sha256 in enumerate(generated_image_sha256s):
        if sha256 in common_sha256s:
            filtered_generated_paths.append(generated_image_paths[i])
            filtered_generated_classes.append(generated_image_classes[i])
    
    real_image_paths, real_image_classes = filtered_real_paths, filtered_real_classes
    generated_image_paths, generated_image_classes = filtered_generated_paths, filtered_generated_classes
    
    if missing_real_sha256:
        print(f"\nWarning: {len(missing_real_sha256)} real SHA256s missing rendered directories")
    if missing_generated_sha256:
        print(f"\nWarning: {len(missing_generated_sha256)} generated SHA256s missing rendered directories")
        for sha256 in missing_generated_sha256:
            print(f"  - {sha256}")

    print(f"\nAfter filtering to common SHA256s:")
    print(f"  Final real images: {len(real_image_paths)} (from {len(common_sha256s)} assets)")
    print(f"  Final generated images: {len(generated_image_paths)} (from {len(common_sha256s)} assets)")
    print(f"  Images per asset: {args.num_views}")
    
    expected_images_per_set = len(common_sha256s) * args.num_views
    if len(real_image_paths) != expected_images_per_set:
        print(f"  WARNING: Expected {expected_images_per_set} real images, got {len(real_image_paths)}")
    if len(generated_image_paths) != expected_images_per_set:
        print(f"  WARNING: Expected {expected_images_per_set} generated images, got {len(generated_image_paths)}")
    
    print(f"\nClass distribution after filtering:")
    real_class_counts = Counter(real_image_classes)
    for class_name, count in sorted(real_class_counts.items()):
        print(f"  {class_name}: {count} images ({count // args.num_views} assets)")
    
    run_info = {
        "timestamp": timestamp,
        "args": vars(args),
        "device": str(device),
        "dataset_filtering": {
            "total_eval_assets": len(eval_sha256s),
            "available_real_sha256s": len(available_real_sha256),
            "available_generated_sha256s": len(available_generated_sha256),
            "common_sha256s": len(common_sha256s),
            "only_real_sha256s": len(only_real_sha256s),
            "only_generated_sha256s": len(only_generated_sha256s),
            "filtering_applied": True
        },
        "num_real_images": len(real_image_paths),
        "num_generated_images": len(generated_image_paths),
        "eval_split_file": args.eval_split_file
    }
    
    run_info_path = os.path.join(results_dir, "run_info.json")
    with open(run_info_path, 'w') as f:
        json.dump(run_info, f, indent=2, default=str)
    print(f"Saved run info to: {run_info_path}")
    
    all_results = {}
    
    for model_type in args.models:
        print("\n" + "=" * 50)
        print(f"PROCESSING {model_type.upper()} MODEL")
        print("=" * 50)
        
        print(f"Loading {model_type} model...")
        model, transform = setup_model_and_transforms(model_type, device)
        
        print("Extracting features from real assets...")
        real_features = extract_features_from_paths(
            real_image_paths,
            model_type,
            model,
            transform,
            device,
            args.batch_size,
            type='real'
        )
        
        if real_features is None:
            print(f"Error: Failed to extract real features for {model_type}")
            continue
            
        print(f"Extracted real features shape: {real_features.shape}")

        print("Extracting features from generated assets...")
        generated_features = extract_features_from_paths(
            generated_image_paths,
            model_type,
            model,
            transform,
            device,
            args.batch_size,
            type='generated'
        )

        if generated_features is None:
            print(f"Error: Failed to extract generated features for {model_type}")
            continue
            
        print(f"Extracted generated features shape: {generated_features.shape}")

        features_dir = os.path.join(results_dir, "extracted_features")
        os.makedirs(features_dir, exist_ok=True)
        
        real_features_path = os.path.join(features_dir, f'real_{model_type}_features.npy')
        np.save(real_features_path, real_features)
        print(f"Saved real features to: {real_features_path}")
        
        generated_features_path = os.path.join(features_dir, f'generated_{model_type}_features.npy')
        np.save(generated_features_path, generated_features)
        print(f"Saved generated features to: {generated_features_path}")

        results_dict = calculate_metrics_from_features_by_class_and_stage(
            real_features,
            generated_features,
            real_image_classes,
            generated_image_classes,
            model_type,
            args.max_samples,
            args.seed
        )
        
        print(f"\nResults for {model_type} features:")
        print(f"Overall - Fréchet Distance (FD): {results_dict['overall']['frechet_distance']:.4f}")
        
        if 'by_stage' in results_dict:
            print("Stage-specific results:")
            for stage_name, stage_metrics in results_dict['by_stage'].items():
                print(f"  {stage_name.upper()} - FD: {stage_metrics['frechet_distance']:.4f}")
        
        if 'per_class' in results_dict:
            print("Per-class results:")
            for class_name, class_metrics in results_dict['per_class'].items():
                stage = class_metrics.get('stage', '')
                print(f"  {class_name} ({stage}) - FD: {class_metrics['frechet_distance']:.4f}")
        
        save_results_with_classes_and_stages(
            results_dir,
            model_type,
            results_dict,
            run_info,
            real_features,
            generated_features,
            timestamp
        )
        
        all_results[model_type] = results_dict
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE - FINAL SUMMARY")
    print("=" * 50)
    
    summary = {
        "timestamp": timestamp,
        "run_info": run_info,
        "results_by_model": all_results,
        "paper_metrics": {}
    }
    
    for model_type, results in all_results.items():
        if model_type == 'inception':
            summary["paper_metrics"]["FDincep"] = results["overall"]["frechet_distance"]

            if "by_stage" in results:
                for stage, stage_results in results["by_stage"].items():
                    summary["paper_metrics"][f"FDincep_{stage}"] = stage_results["frechet_distance"]
    
    summary_path = os.path.join(results_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved comprehensive summary to: {summary_path}")
    
    print(f"\nFinal Results (saved to {results_dir}):")
    print("-" * 40)

    for model_type, results in all_results.items():
        print(f"{model_type.upper()}:")
        print(f"  Overall - Fréchet Distance: {results['overall']['frechet_distance']:.4f}")
        
        if 'by_stage' in results:
            print(f"  Stage-specific results:")
            for stage_name, stage_metrics in results['by_stage'].items():
                print(f"    {stage_name.upper()} - FD: {stage_metrics['frechet_distance']:.4f}")
        
        if 'per_class' in results:
            print(f"  Per-class results:")
            base_classes = {c: m for c, m in results['per_class'].items() if m.get('stage') == 'base'}
            novel_classes = {c: m for c, m in results['per_class'].items() if m.get('stage') == 'novel'}
            
            if base_classes:
                print(f"    Base:")
                for class_name, class_metrics in sorted(base_classes.items()):
                    print(f"      {class_name} - FD: {class_metrics['frechet_distance']:.4f}")
            
            if novel_classes:
                print(f"    Novel:")
                for class_name, class_metrics in sorted(novel_classes.items()):
                    print(f"      {class_name} - FD: {class_metrics['frechet_distance']:.4f}")
        print()
    
    if 'FDincep' in summary["paper_metrics"]:
        print(f"Paper Metrics (Overall):")
        print(f"  FDincep: {summary['paper_metrics']['FDincep']:.4f}")

        if 'FDincep_base' in summary["paper_metrics"]:
            print(f"  FDincep_base: {summary['paper_metrics']['FDincep_base']:.4f}")

        if 'FDincep_novel' in summary["paper_metrics"]:
            print(f"  FDincep_novel: {summary['paper_metrics']['FDincep_novel']:.4f}")
    
    print(f"\nAll results and logs saved to: {results_dir}")
    print("Evaluation pipeline completed successfully!")


if __name__ == '__main__':
    main()