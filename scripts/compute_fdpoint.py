from datetime import datetime
import os
from unittest import result
import numpy as np
from scipy.linalg import sqrtm
import argparse
import pandas as pd
import json
from collections import defaultdict

from metrics import calculate_frechet_distance

CLASSES_BASE = ["airplane", "bicycle", "bottle", "bowl", "cake", "candy", "car", "cat", "cells_battery", "chair", "chicken", "cookie", "cupcake", "deer_moose", "donut", "dragon", "elephant", "fish", "flower", "grapes", "hammer", "helicopter", "helmet", "horse", "ice_cream", "mouse", "motorcycle", "mushroom", "orange", "panda", "pear", "penguin", "piano", "sandwich", "screwdriver", "shark", "sheep", "shoe", "snake", "sofa", "trashcan", "tractor", "train", "tree", "violin"]
CLASSES_NOVEL = ["apple", "ball", "banana", "boat", "bread", "bunny", "bus", "butterfly", "chess_piece", "coin", "cow", "crab", "cup", "dinosaur", "dog", "dolphin", "drum", "fan", "fox", "fridge", "fries", "frog", "glass", "guitar", "hamburger", "hat", "key", "knife", "laptop", "lizard", "monkey", "mug", "octopus", "pencil", "phone", "pig", "pizza", "plate", "radio", "robot", "sink", "spade", "stove", "truck", "whale"]

def load_features_with_classes(dir_path, sha256_to_class_dict):
    """Load .npy feature vectors from a directory for specified SHA256s with class info."""
    features = []
    shas = []
    classes = []
    
    for sha256, class_name in sha256_to_class_dict.items():
        fpath = os.path.join(dir_path, f"{sha256}.npy")
        if os.path.exists(fpath):
            vec = np.load(fpath)
            features.append(vec)
            shas.append(sha256)
            classes.append(class_name)
        else:
            print(f"Warning: Feature file not found for {sha256}")
    
    if len(features) == 0:
        return None, [], []
        
    return np.stack(features), shas, classes


def compute_fd(real_feats, gen_feats):
    """Calculate single FD metric between real and generated features."""
    mu_real = np.mean(real_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)

    mu_gen = np.mean(gen_feats, axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)

    fd = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    return fd

def compute_fd_by_class_and_stage(real_feats, gen_feats, real_classes, gen_classes):
    """Calculate FD metrics overall, by stage, and per class."""
    
    results = {}
    
    # Overall FD
    print("Calculating overall FD...")
    fd_overall = compute_fd(real_feats, gen_feats)
    
    results['overall'] = {
        'frechet_distance': fd_overall,
        'real_samples': real_feats.shape[0],
        'generated_samples': gen_feats.shape[0]
    }
    
    print(f"Overall FD: {fd_overall:.6f}")
    
    print("\nCalculating stage-specific FD...")
    stage_results = {}
    
    for stage_name, stage_classes in [('base', CLASSES_BASE), ('novel', CLASSES_NOVEL)]:
        print(f"Processing {stage_name} classes: {len(stage_classes)} classes")
        
        real_stage_indices = [i for i, c in enumerate(real_classes) if c in stage_classes]
        gen_stage_indices = [i for i, c in enumerate(gen_classes) if c in stage_classes]
        
        if len(real_stage_indices) == 0 or len(gen_stage_indices) == 0:
            print(f"  Warning: No samples found for {stage_name}")
            continue
            
        real_stage_features = real_feats[real_stage_indices]
        gen_stage_features = gen_feats[gen_stage_indices]
        
        print(f"  {stage_name} - Real samples: {len(real_stage_features)}, Generated samples: {len(gen_stage_features)}")
        
        if len(real_stage_features) < 3 or len(gen_stage_features) < 3:
            print(f"  Warning: Too few samples for {stage_name}, skipping...")
            continue
        
        try:
            fd_stage = compute_fd(real_stage_features, gen_stage_features)
            
            stage_results[stage_name] = {
                'frechet_distance': fd_stage,
                'real_samples': len(real_stage_features),
                'generated_samples': len(gen_stage_features),
                'classes': sorted([c for c in stage_classes if c in set(real_classes) & set(gen_classes)])
            }
            
            print(f"  {stage_name} FD: {fd_stage:.6f}")
            
        except Exception as e:
            print(f"  Error calculating FD for {stage_name}: {e}")
            continue
    
    results['by_stage'] = stage_results
    
    # Per-class FD
    unique_classes = sorted(set(real_classes) & set(gen_classes))
    print(f"\nComputing class-wise FD for {len(unique_classes)} classes...")
    
    class_results = {}
    
    for class_name in unique_classes:
        print(f"Processing class: {class_name}")
        
        # Get indices for this class
        real_class_indices = [i for i, c in enumerate(real_classes) if c == class_name]
        gen_class_indices = [i for i, c in enumerate(gen_classes) if c == class_name]
        
        if len(real_class_indices) == 0 or len(gen_class_indices) == 0:
            print(f"  Warning: No samples found for class {class_name}")
            continue
            
        # Extract features for this class
        real_class_features = real_feats[real_class_indices]
        gen_class_features = gen_feats[gen_class_indices]
        
        print(f"  Real samples: {len(real_class_features)}, Generated samples: {len(gen_class_features)}")
        
        # Skip if too few samples
        if len(real_class_features) < 3 or len(gen_class_features) < 3:
            print(f"  Warning: Too few samples for class {class_name}, skipping...")
            continue
        
        # Calculate class-specific FD
        try:
            fd_class = compute_fd(real_class_features, gen_class_features)
            
            # Determine which stage this class belongs to
            stage = 'base' if class_name in CLASSES_BASE else 'novel'
            
            class_results[class_name] = {
                'frechet_distance': fd_class,
                'real_samples': len(real_class_features),
                'generated_samples': len(gen_class_features),
                'stage': stage
            }
            
            print(f"  {class_name} ({stage}) FD: {fd_class:.6f}")
            
        except Exception as e:
            print(f"  Error calculating FD for class {class_name}: {e}")
            continue
    
    results['per_class'] = class_results
    
    return results


def save_results_with_classes_and_stages(results_dir, results_dict, timestamp, real_shape, gen_shape, missing_in_real, missing_in_generated, missing_from_split):
    """Save FD results with class-wise breakdown and stage-specific results in multiple formats."""
    
    # Create comprehensive results
    results = {
        "timestamp": timestamp,
        "metrics": results_dict,
        "dataset_info": {
            "real_features_shape": real_shape,
            "generated_features_shape": gen_shape,
        }
    }
    
    # 1. JSON format
    results_json_path = os.path.join(results_dir, "geometric_evaluation_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved JSON results to: {results_json_path}")
    
    # 2. Text format (human readable)
    results_txt_path = os.path.join(results_dir, "fdpoint_result.txt")
    with open(results_txt_path, 'w') as f:
        f.write(f"TRELLIS Geometric Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("\n")
        f.write("Dataset Information:\n")
        f.write(f"  Real features shape: {real_shape}\n")
        f.write(f"  Generated features shape: {gen_shape}\n")
        f.write(f"  Missing in real features: {missing_in_real}\n")
        f.write(f"  Missing in generated features: {missing_in_generated}\n")
        f.write(f"  Missing from evaluation split: {missing_from_split}\n")
        f.write("\n")
        
        # Overall metrics
        f.write("Overall Evaluation Metrics:\n")
        f.write(f"  Fréchet Distance (FDpoint): {results_dict['overall']['frechet_distance']:.6f}\n")
        f.write(f"  Real samples: {results_dict['overall']['real_samples']}\n")
        f.write(f"  Generated samples: {results_dict['overall']['generated_samples']}\n")
        f.write("\n")
        
        # Stage-specific metrics
        if 'by_stage' in results_dict:
            f.write("Stage-Specific Evaluation Metrics:\n")
            for stage_name, stage_metrics in results_dict['by_stage'].items():
                f.write(f"  {stage_name.upper()}:\n")
                f.write(f"    Fréchet Distance (FDpoint): {stage_metrics['frechet_distance']:.6f}\n")
                f.write(f"    Real samples: {stage_metrics['real_samples']}\n")
                f.write(f"    Generated samples: {stage_metrics['generated_samples']}\n")
                f.write(f"    Classes ({len(stage_metrics['classes'])}): {', '.join(stage_metrics['classes'])}\n")
                f.write("\n")
        
        # Per-class metrics
        if 'per_class' in results_dict:
            f.write("Per-Class Evaluation Metrics:\n")
            # Group by stage for better organization
            base_classes = [c for c, m in results_dict['per_class'].items() if m.get('stage') == 'base']
            novel_classes = [c for c, m in results_dict['per_class'].items() if m.get('stage') == 'novel']
            
            if base_classes:
                f.write("  Base Classes:\n")
                for class_name in sorted(base_classes):
                    class_metrics = results_dict['per_class'][class_name]
                    f.write(f"    {class_name}:\n")
                    f.write(f"      Fréchet Distance (FDpoint): {class_metrics['frechet_distance']:.6f}\n")
                    f.write(f"      Real samples: {class_metrics['real_samples']}\n")
                    f.write(f"      Generated samples: {class_metrics['generated_samples']}\n")
                f.write("\n")
            
            if novel_classes:
                f.write("  Novel Classes:\n")
                for class_name in sorted(novel_classes):
                    class_metrics = results_dict['per_class'][class_name]
                    f.write(f"    {class_name}:\n")
                    f.write(f"      Fréchet Distance (FDpoint): {class_metrics['frechet_distance']:.6f}\n")
                    f.write(f"      Real samples: {class_metrics['real_samples']}\n")
                    f.write(f"      Generated samples: {class_metrics['generated_samples']}\n")
                f.write("\n")
    
    print(f"Saved text results to: {results_txt_path}")
    
    results_csv_path = os.path.join(results_dir, "geometric_evaluation_results.csv")
    csv_data = []
    
    # Overall row
    csv_data.append({
        'timestamp': timestamp,
        'category': 'overall',
        'class_or_stage': 'overall',
        'stage': 'both',
        'frechet_distance': results_dict['overall']['frechet_distance'],
        'real_samples': results_dict['overall']['real_samples'],
        'generated_samples': results_dict['overall']['generated_samples']
    })
    
    # Stage-specific rows
    if 'by_stage' in results_dict:
        for stage_name, stage_metrics in results_dict['by_stage'].items():
            csv_data.append({
                'timestamp': timestamp,
                'category': 'by_stage',
                'class_or_stage': stage_name,
                'stage': stage_name,
                'frechet_distance': stage_metrics['frechet_distance'],
                'real_samples': stage_metrics['real_samples'],
                'generated_samples': stage_metrics['generated_samples']
            })
    
    # Per-class rows
    if 'per_class' in results_dict:
        for class_name, class_metrics in results_dict['per_class'].items():
            csv_data.append({
                'timestamp': timestamp,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True,
                        help="Directory with real asset features (.npy files)")
    parser.add_argument("--gen_dir", type=str, required=True,
                        help="Directory with generated asset features (.npy files)")
    parser.add_argument("--eval_split_file", type=str, required=True,
                        help="CSV file containing evaluation split SHA256s and classes (columns: sha256, class)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save computed FD results")
    parser.add_argument('--run_name', type=str, default=None,
                       help='Custom name for this run (if not provided, uses timestamp)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_dir_name = f"{timestamp}_{args.run_name}_evaluation"
    else:
        run_dir_name = f"{timestamp}_geometric_evaluation"

    exp_folder_name = os.path.basename(os.path.dirname(args.gen_dir))

    results_dir = os.path.join(args.output_dir, run_dir_name + "_" + exp_folder_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    eval_df = pd.read_csv(args.eval_split_file, usecols=['sha256', 'class'])
        
    # Create mapping from SHA256 to class
    sha256_to_class = dict(zip(eval_df['sha256'], eval_df['class']))
    
    print(f"Loaded {len(eval_df)} SHA256s from evaluation split")
    print(f"Found {len(eval_df['class'].unique())} unique classes: {sorted(eval_df['class'].unique())}")
    
    class_counts = eval_df['class'].value_counts()
    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} samples")

    print("\nLoading features...")
    real_feats, real_shas, real_classes = load_features_with_classes(args.real_dir, sha256_to_class)
    gen_feats, gen_shas, gen_classes = load_features_with_classes(args.gen_dir, sha256_to_class)
    
    if real_feats is None or gen_feats is None:
        print("Error: Failed to load features")
        return
    
    print(f"Loaded {len(real_shas)} real features and {len(gen_shas)} generated features.")

    common_shas = list(set(real_shas) & set(gen_shas))
    missing_from_split = len(eval_df) - len(common_shas)
    missing_in_real = set(gen_shas) - set(real_shas)
    missing_in_generated = set(real_shas) - set(gen_shas)
    list_missing_real = list(missing_in_real)
    list_missing_generated = list(missing_in_generated)

    if len(common_shas) == 0:
        print("Error: No common SHA256s found between real and generated features")
        return

    real_indices = [i for i, sha in enumerate(real_shas) if sha in common_shas]
    gen_indices = [i for i, sha in enumerate(gen_shas) if sha in common_shas]

    real_feats_filtered = real_feats[real_indices]
    gen_feats_filtered = gen_feats[gen_indices]
    real_classes_filtered = [real_classes[i] for i in real_indices]
    gen_classes_filtered = [gen_classes[i] for i in gen_indices]

    print(f'Shapes - Real: {real_feats_filtered.shape}, Generated: {gen_feats_filtered.shape}')

    print(f"Computing FD over {len(common_shas)} shared assets...")
    print (f"Missing ids in real: {list_missing_real}")
    print (f"Missing ids in generated: {list_missing_generated}")

    # Calculate metrics (overall, by stage, and per-class)
    results_dict = compute_fd_by_class_and_stage(
        real_feats_filtered, gen_feats_filtered, 
        real_classes_filtered, gen_classes_filtered
    )
    
    print(f"\nFinal Results:")
    print(f"Overall Fréchet Distance (FDpoint): {results_dict['overall']['frechet_distance']:.6f}")
    
    if 'by_stage' in results_dict:
        print("Stage-specific results:")
        for stage_name, stage_metrics in results_dict['by_stage'].items():
            print(f"  {stage_name.upper()} FDpoint: {stage_metrics['frechet_distance']:.6f}")
    
    if 'per_class' in results_dict:
        print("Per-class results:")
        base_classes = {c: m for c, m in results_dict['per_class'].items() if m.get('stage') == 'base'}
        novel_classes = {c: m for c, m in results_dict['per_class'].items() if m.get('stage') == 'novel'}
        
        if base_classes:
            print("  Base:")
            for class_name, class_metrics in sorted(base_classes.items()):
                print(f"    {class_name} FDpoint: {class_metrics['frechet_distance']:.6f}")
        
        if novel_classes:
            print("  Novel:")
            for class_name, class_metrics in sorted(novel_classes.items()):
                print(f"    {class_name} FDpoint: {class_metrics['frechet_distance']:.6f}")

    save_results_with_classes_and_stages(
        results_dir, results_dict, timestamp, 
        real_feats_filtered.shape, gen_feats_filtered.shape, list_missing_real, list_missing_generated, missing_from_split
    )
    
    print(f"\nAll results saved to: {results_dir}")

if __name__ == "__main__":
    main()
