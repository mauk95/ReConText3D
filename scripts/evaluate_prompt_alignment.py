#!/usr/bin/env python3
"""
CLIP Score Evaluation for TRELLIS Prompt Alignment Assessment.
Calculates CLIP scores between rendered views of generated 3D assets and their corresponding text captions.

The eval_split_file should be a metadata CSV containing:
- sha256
- captions
- class
"""

import os
import argparse
import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime
from collections import defaultdict

CLASSES_STAGE_BASE = ["airplane", "bicycle", "bottle", "bowl", "cake", "candy", "car", "cat", "cells_battery", "chair", "chicken", "cookie", "cupcake", "deer_moose", "donut", "dragon", "elephant", "fish", "flower", "grapes", "hammer", "helicopter", "helmet", "horse", "ice_cream", "mouse", "motorcycle", "mushroom", "orange", "panda", "pear", "penguin", "piano", "sandwich", "screwdriver", "shark", "sheep", "shoe", "snake", "sofa", "trashcan", "tractor", "train", "tree", "violin"]
CLASSES_STAGE_NOVEL = ["apple", "ball", "banana", "boat", "bread", "bunny", "bus", "butterfly", "chess_piece", "coin", "cow", "crab", "cup", "dinosaur", "dog", "dolphin", "drum", "fan", "fox", "fridge", "fries", "frog", "glass", "guitar", "hamburger", "hat", "key", "knife", "laptop", "lizard", "monkey", "mug", "octopus", "pencil", "phone", "pig", "pizza", "plate", "radio", "robot", "sink", "spade", "stove", "truck", "whale"]


def load_eval_metadata(eval_split_file):
    """Load SHA256, captions, and class info from a metadata CSV."""
    print(f"Loading eval metadata from: {eval_split_file}")

    df = pd.read_csv(eval_split_file)

    required_cols = ["sha256", "captions", "class"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV must contain columns {required_cols}. Missing: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    captions_dict = {}
    class_dict = {}

    successful_count = 0
    skipped_too_long = 0
    total_count = len(df)

    for _, row in df.iterrows():
        sha256 = str(row["sha256"]).strip()
        asset_class = row["class"]

        try:
            captions = json.loads(row["captions"])

            if isinstance(captions, list) and len(captions) > 0:
                caption = captions[0]

                if caption:
                    try:
                        _ = clip.tokenize([caption], truncate=True)
                        captions_dict[sha256] = caption
                        class_dict[sha256] = asset_class
                        successful_count += 1
                    except RuntimeError:
                        print(f"Warning: Caption too long for CLIP, skipping sha {sha256}")
                        skipped_too_long += 1
                        continue

        except Exception as e:
            print(f"Warning: Failed to parse captions for {sha256}: {e}")
            continue

    print(f"Loaded eval metadata with {total_count} total assets")
    print(f"Successfully loaded {successful_count} captions")
    print(f"Skipped {skipped_too_long} captions")
    print(f"Assets with valid captions: {successful_count}/{total_count} ({successful_count / total_count * 100:.1f}%)")

    class_counts = pd.Series(class_dict).value_counts()
    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} assets")

    return captions_dict, class_dict


def collect_rendered_images(renders_dir, sha256_list, num_views=8):
    """Collect paths to rendered images for prompt alignment evaluation."""
    
    image_caption_pairs = []
    missing_assets = set()
    
    for sha256 in sha256_list:
        sha256_dir = os.path.join(renders_dir, sha256)

        if not os.path.exists(sha256_dir):
            missing_assets.add(sha256)
            continue
            
        asset_images = []

        for view_idx in range(num_views):
            img_path = os.path.join(sha256_dir, f"{view_idx:03d}.png")

            if os.path.exists(img_path):
                asset_images.append(img_path)
            else:
                print(f"Warning: Image not found: {img_path}")
                missing_assets.add(sha256)

        if asset_images:
            image_caption_pairs.append({
                'sha256': sha256,
                'image_paths': asset_images
            })
    
    if missing_assets:
        print(f"Warning: {len(missing_assets)} assets have missing rendered images")
        print(f"First few missing: {list(missing_assets)[:5]}")

    return image_caption_pairs, list(missing_assets)


def calculate_clip_scores_by_class_and_stage(
    image_caption_pairs,
    captions_dict,
    class_dict,
    model,
    preprocess,
    device,
    batch_size=32
):
    """Calculate CLIP scores between images and captions, with class-wise and stage-wise breakdown."""
    
    all_scores = []
    asset_results = {}
    class_scores = defaultdict(list)
    stage_scores = defaultdict(list)
    
    print(f"Calculating CLIP scores for {len(image_caption_pairs)} assets...")
    
    for pair in tqdm(image_caption_pairs, desc="Processing assets"):
        sha256 = pair['sha256']
        image_paths = pair['image_paths']
        
        if sha256 not in captions_dict:
            print(f"Warning: No caption found for {sha256}")
            continue
            
        caption = captions_dict[sha256]

        if not caption:
            print(f"Warning: Empty caption for {sha256}")
            continue
        
        asset_class = class_dict.get(sha256)
        
        if asset_class in CLASSES_STAGE_BASE:
            stage = 'base'
        elif asset_class in CLASSES_STAGE_NOVEL:
            stage = 'novel'
        else:
            stage = 'unknown'
        
        view_scores = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image = preprocess(image)
                    batch_images.append(image)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            images_tensor = torch.stack(batch_images).to(device)
            text_tokens = clip.tokenize([caption], truncate=True).to(device)

            with torch.no_grad():
                image_features = model.encode_image(images_tensor)
                text_features = model.encode_text(text_tokens)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                clip_scores = (image_features @ text_features.T).squeeze(-1)
                scores_np = clip_scores.cpu().numpy()

                view_scores.extend(scores_np.tolist())
        
        if view_scores:
            mean_score = np.mean(view_scores)
            max_score = np.max(view_scores)
            min_score = np.min(view_scores)
            std_score = np.std(view_scores)
            
            asset_results[sha256] = {
                'caption': caption,
                'class': asset_class,
                'stage': stage,
                'num_views': len(view_scores),
                'view_scores': view_scores,
                'mean_score': mean_score,
                'max_score': max_score,
                'min_score': min_score,
                'std_score': std_score
            }
            
            all_scores.extend(view_scores)
            class_scores[asset_class].extend(view_scores)
            stage_scores[stage].extend(view_scores)
    
    return asset_results, all_scores, class_scores, stage_scores


def save_results_with_classes_and_stages(
    results_dir,
    asset_results,
    all_scores,
    class_scores,
    stage_scores,
    run_info,
    timestamp,
    missing_assets
):
    """Save CLIP evaluation results with class-wise and stage-wise breakdown."""
    
    overall_stats = {
        'num_assets': len(asset_results),
        'num_missing_assets': len(missing_assets),
        'missing_assets': missing_assets,
        'num_views_total': len(all_scores),
        'mean_clip_score': float(np.mean(all_scores)),
        'std_clip_score': float(np.std(all_scores)),
        'median_clip_score': float(np.median(all_scores)),
        'min_clip_score': float(np.min(all_scores)),
        'max_clip_score': float(np.max(all_scores))
    }
    
    stage_stats = {}
    for stage_name, scores in stage_scores.items():
        if len(scores) > 0:
            stage_stats[stage_name] = {
                'num_views': len(scores),
                'num_assets': len([a for a in asset_results.values() if a.get('stage') == stage_name]),
                'mean_clip_score': float(np.mean(scores)),
                'std_clip_score': float(np.std(scores)),
                'median_clip_score': float(np.median(scores)),
                'min_clip_score': float(np.min(scores)),
                'max_clip_score': float(np.max(scores)),
                'classes': sorted(list(set([
                    a['class'] for a in asset_results.values()
                    if a.get('stage') == stage_name
                ])))
            }
    
    class_stats = {}
    for class_name, scores in class_scores.items():
        if len(scores) > 0:
            if class_name in CLASSES_STAGE_BASE:
                stage = 'base'
            elif class_name in CLASSES_STAGE_NOVEL:
                stage = 'novel'
            else:
                stage = 'unknown'
            
            class_stats[class_name] = {
                'num_views': len(scores),
                'num_assets': len([a for a in asset_results.values() if a['class'] == class_name]),
                'mean_clip_score': float(np.mean(scores)),
                'std_clip_score': float(np.std(scores)),
                'median_clip_score': float(np.median(scores)),
                'min_clip_score': float(np.min(scores)),
                'max_clip_score': float(np.max(scores)),
                'stage': stage
            }
    
    results = {
        "timestamp": timestamp,
        "evaluation_type": "prompt_alignment_clip_score",
        "overall_statistics": overall_stats,
        "by_stage_statistics": stage_stats,
        "per_class_statistics": class_stats,
        "run_info": run_info,
        "asset_results": asset_results
    }
    
    results_json_path = os.path.join(results_dir, "clip_score_evaluation_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved complete JSON results to: {results_json_path}")
    
    results_txt_path = os.path.join(results_dir, "clip_score_evaluation_summary.txt")
    with open(results_txt_path, 'w') as f:
        f.write("TRELLIS Prompt Alignment Evaluation - CLIP Scores\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("Evaluation Type: Prompt Alignment (CLIP Score)\n\n")

        f.write("Overall Statistics:\n")
        f.write(f"  Number of assets evaluated: {overall_stats['num_assets']}\n")
        f.write(f"  Number of missing assets: {overall_stats['num_missing_assets']}\n")
        f.write(f"  Missing assets: {overall_stats['missing_assets']}\n")
        f.write(f"  Total number of views: {overall_stats['num_views_total']}\n")
        f.write(f"  Mean CLIP Score: {overall_stats['mean_clip_score']:.4f}\n")
        f.write(f"  Standard Deviation: {overall_stats['std_clip_score']:.4f}\n")
        f.write(f"  Median CLIP Score: {overall_stats['median_clip_score']:.4f}\n")
        f.write(f"  Min CLIP Score: {overall_stats['min_clip_score']:.4f}\n")
        f.write(f"  Max CLIP Score: {overall_stats['max_clip_score']:.4f}\n\n")
        
        if stage_stats:
            f.write("Stage-Specific Statistics:\n")
            f.write("-" * 40 + "\n")
            for stage_name, stats in stage_stats.items():
                f.write(f"Stage: {stage_name.upper()}\n")
                f.write(f"  Assets: {stats['num_assets']}, Views: {stats['num_views']}\n")
                f.write(f"  Mean CLIP Score: {stats['mean_clip_score']:.4f}\n")
                f.write(f"  Std CLIP Score: {stats['std_clip_score']:.4f}\n")
                f.write(f"  Median CLIP Score: {stats['median_clip_score']:.4f}\n")
                f.write(f"  Min CLIP Score: {stats['min_clip_score']:.4f}\n")
                f.write(f"  Max CLIP Score: {stats['max_clip_score']:.4f}\n")
                f.write(f"  Classes ({len(stats['classes'])}): {', '.join(stats['classes'])}\n\n")
        
        if class_stats:
            f.write("Per-Class Statistics:\n")
            f.write("-" * 40 + "\n")
            
            base_classes = {c: s for c, s in class_stats.items() if s.get('stage') == 'base'}
            novel_classes = {c: s for c, s in class_stats.items() if s.get('stage') == 'novel'}
            unknown_classes = {c: s for c, s in class_stats.items() if s.get('stage') == 'unknown'}
            
            if base_classes:
                f.write("Base Classes:\n")
                for class_name, stats in sorted(base_classes.items()):
                    f.write(f"  {class_name}:\n")
                    f.write(f"    Assets: {stats['num_assets']}, Views: {stats['num_views']}\n")
                    f.write(f"    Mean CLIP Score: {stats['mean_clip_score']:.4f}\n")
                    f.write(f"    Std CLIP Score: {stats['std_clip_score']:.4f}\n")
                    f.write(f"    Median CLIP Score: {stats['median_clip_score']:.4f}\n")
                    f.write(f"    Min CLIP Score: {stats['min_clip_score']:.4f}\n")
                    f.write(f"    Max CLIP Score: {stats['max_clip_score']:.4f}\n\n")
            
            if novel_classes:
                f.write("Novel Classes:\n")
                for class_name, stats in sorted(novel_classes.items()):
                    f.write(f"  {class_name}:\n")
                    f.write(f"    Assets: {stats['num_assets']}, Views: {stats['num_views']}\n")
                    f.write(f"    Mean CLIP Score: {stats['mean_clip_score']:.4f}\n")
                    f.write(f"    Std CLIP Score: {stats['std_clip_score']:.4f}\n")
                    f.write(f"    Median CLIP Score: {stats['median_clip_score']:.4f}\n")
                    f.write(f"    Min CLIP Score: {stats['min_clip_score']:.4f}\n")
                    f.write(f"    Max CLIP Score: {stats['max_clip_score']:.4f}\n\n")

            if unknown_classes:
                f.write("Unknown Classes:\n")
                for class_name, stats in sorted(unknown_classes.items()):
                    f.write(f"  {class_name}:\n")
                    f.write(f"    Assets: {stats['num_assets']}, Views: {stats['num_views']}\n")
                    f.write(f"    Mean CLIP Score: {stats['mean_clip_score']:.4f}\n")
                    f.write(f"    Std CLIP Score: {stats['std_clip_score']:.4f}\n")
                    f.write(f"    Median CLIP Score: {stats['median_clip_score']:.4f}\n")
                    f.write(f"    Min CLIP Score: {stats['min_clip_score']:.4f}\n")
                    f.write(f"    Max CLIP Score: {stats['max_clip_score']:.4f}\n\n")
        
        f.write("Per-Asset Statistics (Top 10 by mean score):\n")
        f.write("-" * 40 + "\n")
        
        sorted_assets = sorted(
            asset_results.items(),
            key=lambda x: x[1]['mean_score'],
            reverse=True
        )
        
        for i, (sha256, result) in enumerate(sorted_assets[:10]):
            f.write(
                f"{i + 1:2d}. {sha256[:12]}... "
                f"Mean: {result['mean_score']:.4f} "
                f"Class: {result['class']} ({result.get('stage', 'unknown')}) "
                f"({result['num_views']} views)\n"
            )
            f.write(f"     Caption: {result['caption'][:80]}...\n\n")
    
    print(f"Saved text summary to: {results_txt_path}")
    
    csv_data = []
    
    csv_data.append({
        'category': 'overall',
        'class_or_stage': 'overall',
        'stage': 'both',
        'num_assets': overall_stats['num_assets'],
        'num_views': overall_stats['num_views_total'],
        'mean_clip_score': overall_stats['mean_clip_score'],
        'std_clip_score': overall_stats['std_clip_score'],
        'median_clip_score': overall_stats['median_clip_score'],
        'min_clip_score': overall_stats['min_clip_score'],
        'max_clip_score': overall_stats['max_clip_score']
    })
    
    for stage_name, stats in stage_stats.items():
        csv_data.append({
            'category': 'by_stage',
            'class_or_stage': stage_name,
            'stage': stage_name,
            'num_assets': stats['num_assets'],
            'num_views': stats['num_views'],
            'mean_clip_score': stats['mean_clip_score'],
            'std_clip_score': stats['std_clip_score'],
            'median_clip_score': stats['median_clip_score'],
            'min_clip_score': stats['min_clip_score'],
            'max_clip_score': stats['max_clip_score']
        })
    
    for class_name, stats in class_stats.items():
        csv_data.append({
            'category': 'per_class',
            'class_or_stage': class_name,
            'stage': stats.get('stage', 'unknown'),
            'num_assets': stats['num_assets'],
            'num_views': stats['num_views'],
            'mean_clip_score': stats['mean_clip_score'],
            'std_clip_score': stats['std_clip_score'],
            'median_clip_score': stats['median_clip_score'],
            'min_clip_score': stats['min_clip_score'],
            'max_clip_score': stats['max_clip_score']
        })
    
    results_csv_path = os.path.join(results_dir, "clip_score_by_class_and_stage.csv")
    df = pd.DataFrame(csv_data)
    df.to_csv(results_csv_path, index=False)
    print(f"Saved class-wise and stage-wise CSV to: {results_csv_path}")
    
    asset_csv_data = []
    for sha256, result in asset_results.items():
        asset_csv_data.append({
            'sha256': sha256,
            'class': result['class'],
            'stage': result.get('stage', 'unknown'),
            'caption': result['caption'],
            'num_views': result['num_views'],
            'mean_clip_score': result['mean_score'],
            'max_clip_score': result['max_score'],
            'min_clip_score': result['min_score'],
            'std_clip_score': result['std_score']
        })
    
    asset_results_csv_path = os.path.join(results_dir, "clip_score_per_asset.csv")
    df_assets = pd.DataFrame(asset_csv_data)
    df_assets.to_csv(asset_results_csv_path, index=False)
    print(f"Saved per-asset CSV to: {asset_results_csv_path}")
    
    view_data = []
    for sha256, result in asset_results.items():
        for i, score in enumerate(result['view_scores']):
            view_data.append({
                'sha256': sha256,
                'class': result['class'],
                'stage': result.get('stage', 'unknown'),
                'view_index': i,
                'clip_score': score,
                'caption': result['caption']
            })
    
    view_csv_path = os.path.join(results_dir, "clip_score_per_view.csv")
    df_views = pd.DataFrame(view_data)
    df_views.to_csv(view_csv_path, index=False)
    print(f"Saved per-view CSV to: {view_csv_path}")


def main():
    parser = argparse.ArgumentParser(description='CLIP Score Evaluation for Prompt Alignment')

    parser.add_argument(
        '--generated_renders_dir',
        type=str,
        default='datasets/Toys4k_eval_prompt_alignment_generated/renders',
        help='Directory containing rendered views of generated assets'
    )
    parser.add_argument(
        '--eval_split_file',
        type=str,
        required=True,
        help='Metadata CSV containing assets to evaluate with columns: sha256, captions, class'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Base directory to save evaluation results'
    )
    parser.add_argument(
        '--clip_model',
        type=str,
        default='ViT-B/32',
        choices=[
            'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
            'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        ],
        help='CLIP model variant to use'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for CLIP computation'
    )
    parser.add_argument(
        '--num_views',
        type=int,
        default=8,
        help='Number of rendered views per asset'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for CLIP computation'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Custom name for this run'
    )

    args = parser.parse_args()
    
    if not os.path.exists(args.eval_split_file):
        print(f"Error: Eval metadata file not found: {args.eval_split_file}")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.run_name:
        run_dir_name = f"{timestamp}_{args.run_name}_clip_evaluation"
    else:
        run_dir_name = f"{timestamp}_clip_prompt_alignment"
    
    exp_folder_name = os.path.basename(os.path.dirname(args.generated_renders_dir))

    results_dir = os.path.join(
        args.output_dir,
        run_dir_name + "_" + exp_folder_name + "_" + args.clip_model.replace("/", "-")
    )
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 80)
    print("TRELLIS PROMPT ALIGNMENT EVALUATION - CLIP SCORES")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"CLIP model: {args.clip_model}")
    print(f"Device: {args.device}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading CLIP model: {args.clip_model}")
    model, preprocess = clip.load(args.clip_model, device=device)
    print("CLIP model loaded successfully")
    
    eval_captions, eval_classes = load_eval_metadata(args.eval_split_file)

    if not eval_captions:
        print("Error: No valid captions loaded from eval metadata CSV.")
        return

    eval_sha256s = list(eval_captions.keys())

    print(f"Assets in eval metadata: {len(eval_sha256s)}")
    print(f"Assets with captions: {len(eval_captions)}")
    print(f"Assets with class info: {len(eval_classes)}")
    
    if eval_classes:
        class_counts = defaultdict(int)
        for class_name in eval_classes.values():
            class_counts[class_name] += 1

        print("Class distribution in evaluated assets:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} assets")
    
    print("\n" + "=" * 50)
    print("COLLECTING RENDERED IMAGES")
    print("=" * 50)
    
    image_caption_pairs, missing_assets = collect_rendered_images(
        args.generated_renders_dir,
        eval_sha256s,
        args.num_views
    )
    
    print(f"Found rendered images for {len(image_caption_pairs)} assets")
    
    pairs_with_captions = []

    for pair in image_caption_pairs:
        if pair['sha256'] in eval_captions and eval_captions[pair['sha256']]:
            pairs_with_captions.append(pair)
    
    print(f"Assets with both renders and captions: {len(pairs_with_captions)}")
    
    if not pairs_with_captions:
        print("Error: No assets found with both rendered images and captions")
        return
    
    run_info = {
        "timestamp": timestamp,
        "args": vars(args),
        "clip_model": args.clip_model,
        "device": str(device),
        "num_assets_with_renders": len(image_caption_pairs),
        "num_assets_with_captions": len(pairs_with_captions),
        "total_captions_available": len(eval_captions),
        "eval_metadata_size": len(eval_sha256s)
    }
    
    run_info_path = os.path.join(results_dir, "run_info.json")
    with open(run_info_path, 'w') as f:
        json.dump(run_info, f, indent=2, default=str)
    print(f"Saved run info to: {run_info_path}")
    
    print("\n" + "=" * 50)
    print("CALCULATING CLIP SCORES")
    print("=" * 50)
    
    asset_results, all_scores, class_scores, stage_scores = calculate_clip_scores_by_class_and_stage(
        pairs_with_captions,
        eval_captions,
        eval_classes,
        model,
        preprocess,
        device,
        args.batch_size
    )
    
    if not all_scores:
        print("Error: No CLIP scores calculated")
        return
    
    print(f"\nCalculated CLIP scores for {len(asset_results)} assets")
    print(f"Total views processed: {len(all_scores)}")
    print(f"Mean CLIP score: {np.mean(all_scores):.4f}")
    print(f"Std CLIP score: {np.std(all_scores):.4f}")
    
    if stage_scores:
        print("\nPer-stage CLIP scores:")
        for stage_name, scores in sorted(stage_scores.items()):
            if len(scores) > 0:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                num_assets = len([
                    a for a in asset_results.values()
                    if a.get('stage') == stage_name
                ])
                print(
                    f"  {stage_name.upper()}: {mean_score:.4f} ± {std_score:.4f} "
                    f"({num_assets} assets, {len(scores)} views)"
                )
    
    if class_scores:
        print("\nPer-class CLIP scores:")

        base_classes = {c: s for c, s in class_scores.items() if c in CLASSES_STAGE_BASE}
        novel_classes = {c: s for c, s in class_scores.items() if c in CLASSES_STAGE_NOVEL}
        unknown_classes = {
            c: s for c, s in class_scores.items()
            if c not in CLASSES_STAGE_BASE and c not in CLASSES_STAGE_NOVEL
        }
        
        if base_classes:
            print("  Base:")
            for class_name, scores in sorted(base_classes.items()):
                if len(scores) > 0:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    num_assets = len([
                        a for a in asset_results.values()
                        if a['class'] == class_name
                    ])
                    print(
                        f"    {class_name}: {mean_score:.4f} ± {std_score:.4f} "
                        f"({num_assets} assets, {len(scores)} views)"
                    )
        
        if novel_classes:
            print("  Novel:")
            for class_name, scores in sorted(novel_classes.items()):
                if len(scores) > 0:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    num_assets = len([
                        a for a in asset_results.values()
                        if a['class'] == class_name
                    ])
                    print(
                        f"    {class_name}: {mean_score:.4f} ± {std_score:.4f} "
                        f"({num_assets} assets, {len(scores)} views)"
                    )

        if unknown_classes:
            print("  Unknown:")
            for class_name, scores in sorted(unknown_classes.items()):
                if len(scores) > 0:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    num_assets = len([
                        a for a in asset_results.values()
                        if a['class'] == class_name
                    ])
                    print(
                        f"    {class_name}: {mean_score:.4f} ± {std_score:.4f} "
                        f"({num_assets} assets, {len(scores)} views)"
                    )
    
    print("\n" + "=" * 50)
    print("SAVING RESULTS")
    print("=" * 50)
    
    save_results_with_classes_and_stages(
        results_dir,
        asset_results,
        all_scores,
        class_scores,
        stage_scores,
        run_info,
        timestamp,
        missing_assets
    )
    
    print("\nPrompt alignment evaluation complete!")
    print(f"Mean CLIP Score: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
    print(f"All results saved to: {results_dir}")


if __name__ == '__main__':
    main()