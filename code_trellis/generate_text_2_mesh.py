#!/usr/bin/env python3
"""
Script to generate 3D meshes for all items in the Toys4k evaluation split using TRELLIS text-to-3D pipeline.
This script processes all captions from the Toys4k dataset evaluation split and generates corresponding 3D assets.
"""

import os

# Set environment variables before importing TRELLIS
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto'

import imageio
import numpy as np
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

import csv
import json
import argparse
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Optional
import logging
from functools import partial
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_eval_split(split_file: str) -> List[str]:
    """Load the list of SHA256 hashes from the evaluation split file."""
    with open(split_file, 'r') as f:
        sha256_list = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(sha256_list)} items from evaluation split")
    return sha256_list


def load_metadata(metadata_file: str) -> Dict[str, Dict]:
    """Load metadata from CSV file and return a dictionary mapping SHA256 to metadata."""
    metadata = {}
    with open(metadata_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sha256 = row['sha256']
            # Parse captions from string representation of list
            try:
                captions = eval(row['captions'])  # Safe since we trust the source
                if isinstance(captions, list) and len(captions) > 0:
                    metadata[sha256] = {
                        'captions': captions,
                        'file_identifier': row['file_identifier'],
                        'aesthetic_score': float(row['aesthetic_score']),
                        'local_path': row['local_path']
                    }
            except Exception as e:
                logger.warning(f"Failed to parse captions for {sha256}: {e}")
                continue
    
    logger.info(f"Loaded metadata for {len(metadata)} items")
    return metadata


def setup_output_directories(output_dir: str) -> Dict[str, str]:
    """Create output directories for different asset types."""
    output_paths = {
        'meshes': os.path.join(output_dir, 'meshes'),
        'gaussians': os.path.join(output_dir, 'gaussians'),
        'radiance_fields': os.path.join(output_dir, 'radiance_fields'),
        'videos': os.path.join(output_dir, 'videos'),
        'glb': os.path.join(output_dir, 'glb'),
        'logs': os.path.join(output_dir, 'logs')
    }
    
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)
    
    return output_paths


def generate_asset(pipeline, caption: str, sha256: str, output_paths: Dict[str, str], 
                  seed: int = 42, save_videos: bool = False, save_glb: bool = True, 
                  save_gaussians: bool = False, save_radiance_fields: bool = False, 
                  save_meshes: bool = False, render_gaussians: bool = False, render_dir: Optional[str] = None) -> bool:
    """Generate 3D asset from caption and save all outputs."""
    try:
        logger.info(f"Generating asset for {sha256} with caption: '{caption}'")
        
        outputs = pipeline.run(
            caption,
            seed=seed,
            # Optional parameters - can be tuned for quality vs speed
            # sparse_structure_sampler_params={
            #     "steps": 12,
            #     "cfg_strength": 7.5,
            # },
            # slat_sampler_params={
            #     "steps": 12,
            #     "cfg_strength": 7.5,
            # },
        )

        if save_gaussians:
            gaussian_path = os.path.join(output_paths['gaussians'], f"{sha256}.ply")
            outputs['gaussian'][0].save_ply(gaussian_path)
            logger.info(f"Saved Gaussian to {gaussian_path}")
        
        if render_gaussians:
            assert render_dir is not None, "render_dir must be specified if render_gaussians is True"

            sha_dir = os.path.join(render_dir, sha256)
            os.makedirs(sha_dir, exist_ok=True)
            
            # Render with transparent background (like Blender)
            render_result = render_utils.render_appearance_views(outputs['gaussian'][0], transparent_bg=False)
            images = render_result['color']
            alphas = render_result.get('alpha', None)
            
            for i, img in enumerate(images):
                if alphas is not None:
                    # Create RGBA image with transparency (like Blender's film_transparent)
                    alpha = alphas[i]
                    rgba_img = np.concatenate([img, alpha[..., np.newaxis]], axis=2)
                    img_path = os.path.join(sha_dir, f'{i:03d}.png')
                    imageio.imwrite(img_path, rgba_img)
                else:
                    # Fallback to RGB if alpha is not available
                    img_path = os.path.join(sha_dir, f'{i:03d}.png')
                    imageio.imwrite(img_path, img)

        # Save meshes as PLY files if requested
        if save_meshes:
            mesh_path = os.path.join(output_paths['meshes'], f"{sha256}.ply")
            outputs['mesh'][0].save_ply(mesh_path)
            logger.info(f"Saved mesh to {mesh_path}")
        
        # Save videos if requested
        if save_videos:
            # Gaussian video (only if gaussians are saved)
            if save_gaussians:
                video = render_utils.render_video(outputs['gaussian'][0])['color']
                gs_video_path = os.path.join(output_paths['videos'], f"{sha256}_gaussian.mp4")
                imageio.mimsave(gs_video_path, video, fps=30)
            
            # Radiance field video (only if radiance fields are saved)
            if save_radiance_fields:
                video = render_utils.render_video(outputs['radiance_field'][0])['color']
                rf_video_path = os.path.join(output_paths['videos'], f"{sha256}_radiance_field.mp4")
                imageio.mimsave(rf_video_path, video, fps=30)
            
            # Mesh video
            video = render_utils.render_video(outputs['mesh'][0])['normal']
            mesh_video_path = os.path.join(output_paths['videos'], f"{sha256}_mesh.mp4")
            imageio.mimsave(mesh_video_path, video, fps=30)
            
            logger.info(f"Saved videos for {sha256}")
        
        # Save GLB files if requested
        if save_glb:
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.95,          # Ratio of triangles to remove
                texture_size=1024,      # Texture size for GLB
            )
            glb_path = os.path.join(output_paths['glb'], f"{sha256}.glb")
            glb.export(glb_path)
            logger.info(f"Saved GLB to {glb_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate asset for {sha256}: {e}")
        return False


def generate_single_asset(file_path: str, sha256: str, pipeline, metadata_dict: Dict, args, output_paths: Dict[str, str]) -> Dict:
    """Generate a single 3D asset - compatible with foreach_instance pattern."""
    try:
        # Get metadata for this item
        item_metadata = metadata_dict[sha256]
        captions = item_metadata['captions']
        
        # Select caption based on index
        if args.caption_index < len(captions):
            caption = captions[args.caption_index]
        else:
            caption = captions[0]  # Fallback to first caption
            logger.warning(f"Caption index {args.caption_index} not available for {sha256}, using index 0")
        
        logger.info(f"Generating asset for {sha256} with caption: '{caption[:100]}...'")

        glb_path = os.path.join(output_paths['glb'], f"{sha256}.glb")
        if os.path.exists(glb_path):
            logger.info(f"GLB for {sha256} already exists at {glb_path}, skipping generation")
            return {
                'sha256': sha256,
                'caption': caption,
                'file_identifier': item_metadata['file_identifier'],
                'aesthetic_score': item_metadata['aesthetic_score'],
                'success': True
            }   
        
        # Generate the asset
        success = generate_asset(
            pipeline=pipeline,
            caption=caption,
            sha256=sha256,
            output_paths=output_paths,
            seed=args.seed,
            save_videos=args.save_videos,
            save_glb=not args.no_glb,
            save_gaussians=args.save_gaussians,
            save_radiance_fields=args.save_radiance_fields,
            save_meshes=args.save_meshes,
            render_gaussians=args.render_gaussians,
            render_dir=args.render_dir
        )

        return {
            'sha256': sha256,
            'caption': caption,
            'file_identifier': item_metadata['file_identifier'],
            'aesthetic_score': item_metadata['aesthetic_score'],
            'success': success
        }
        
    except Exception as e:
        logger.error(f"Failed to generate asset for {sha256}: {e}")
        return {
            'sha256': sha256,
            'caption': '',
            'file_identifier': '',
            'aesthetic_score': 0.0,
            'success': False
        }


def save_generation_log(output_paths: Dict[str, str], results: List[Dict], args):
    """Save a log of all generation results."""
    log_data = {
        'args': vars(args),
        'total_items': len(results),
        'successful': len([r for r in results if r['success']]),
        'failed': len([r for r in results if not r['success']]),
        'results': results
    }
    
    log_path = os.path.join(output_paths['logs'], 'generation_log.json')
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    logger.info(f"Saved generation log to {log_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate 3D meshes for Toys4k evaluation split')
    parser.add_argument('--model', type=str, default='microsoft/TRELLIS-text-xlarge',
                       help='TRELLIS model to use (default: microsoft/TRELLIS-text-xlarge)')
    parser.add_argument('--metadata', type=str, default='datasets/Toys4k/metadata.csv',
                       help='Path to metadata CSV file')
    parser.add_argument('--output-dir', type=str, default='toys4k_generated_assets',
                       help='Output directory for generated assets')
    parser.add_argument('--caption-index', type=int, default=0,
                       help='Which caption to use from the list (default: 0 - most detailed)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for generation')
    parser.add_argument('--save-videos', action='store_true',
                       help='Generate and save video renderings')
    parser.add_argument('--save-gaussians', action='store_true',
                       help='Generate and save Gaussian PLY files')
    parser.add_argument('--save-meshes', action='store_true',
                       help='Generate and save mesh PLY files')
    parser.add_argument('--save-radiance-fields', action='store_true',
                       help='Generate and save radiance field outputs')
    parser.add_argument('--no-glb', action='store_true',
                       help='Skip GLB file generation (mesh GLB files are generated by default)')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Start index for processing (useful for resuming)')
    parser.add_argument('--end-idx', type=int, default=None,
                       help='End index for processing (useful for batch processing)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for generation')
    parser.add_argument('--rank', type=int, default=0,
                       help='Rank of current process for multi-job processing')
    parser.add_argument('--world-size', type=int, default=1,
                       help='Total number of processes for multi-job processing')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='Maximum number of worker threads per process')
    parser.add_argument('--render-gaussians', action='store_true',
                       help='Render Gaussian images')
    parser.add_argument('--render-dir', type=str, default=None,
                       help='Output directory for saving rendered Gaussian images')
    args = parser.parse_args()
    
    # Import dataset utilities
    sys.path.append('dataset_toolkits')
    dataset_utils = importlib.import_module('datasets.Toys4k')
    
    # Load metadata and use all SHA256 IDs from metadata
    metadata = load_metadata(args.metadata)
    valid_sha256s = list(metadata.keys())

    logger.info(f"Using {len(valid_sha256s)} SHA256 IDs from metadata CSV")
    
    # Apply start/end index filtering if not using rank-based splitting
    if args.world_size == 1:
        if args.end_idx is not None:
            valid_sha256s = valid_sha256s[args.start_idx:args.end_idx]
        else:
            valid_sha256s = valid_sha256s[args.start_idx:]
    else:
        # Use rank-based splitting for multi-job processing
        start = len(valid_sha256s) * args.rank // args.world_size
        end = len(valid_sha256s) * (args.rank + 1) // args.world_size
        valid_sha256s = valid_sha256s[start:end]
        logger.info(f"Rank {args.rank}/{args.world_size}: Processing items {start} to {end}")
    
    logger.info(f"Processing {len(valid_sha256s)} items")
    
    output_paths = setup_output_directories(args.output_dir)
    
    if args.render_gaussians:
        assert args.render_dir is not None, "render_dir must be specified if render_gaussians is True"
        os.makedirs(args.render_dir, exist_ok=True)
    
    logger.info(f"Loading TRELLIS pipeline: {args.model}")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model)
    if args.device == 'cuda':
        pipeline.cuda()
    
    # Create metadata for the items to process
    metadata_subset = []
    for sha256 in valid_sha256s:
        if sha256 in metadata:
            # Create a dummy local_path since foreach_instance expects it
            metadata_subset.append({
                'sha256': sha256,
                'local_path': f'dummy/{sha256}',  # Not used in our case
                'file_identifier': metadata[sha256]['file_identifier'],
                'aesthetic_score': metadata[sha256]['aesthetic_score'],
                'captions': metadata[sha256]['captions']
            })
    
    metadata_df = pd.DataFrame(metadata_subset)
    
    generate_func = partial(
        generate_single_asset,
        pipeline=pipeline,
        metadata_dict=metadata,
        args=args,
        output_paths=output_paths
    )
    
    logger.info(f"Starting generation with {args.max_workers} workers")
    results_df = dataset_utils.foreach_instance(
        metadata_df, 
        args.output_dir, 
        generate_func, 
        max_workers=args.max_workers,
        desc=f'Generating 3D assets (rank {args.rank})'
    )
    
    results = results_df.to_dict('records') if not results_df.empty else []
    
    # Save results for this rank
    rank_suffix = f"_{args.rank}" if args.world_size > 1 else ""
    save_generation_log(output_paths, results, args)
    
    # Save rank-specific results
    if args.world_size > 1:
        rank_results_path = os.path.join(output_paths['logs'], f'generation_results_rank_{args.rank}.csv')
        results_df.to_csv(rank_results_path, index=False)
        logger.info(f"Saved rank {args.rank} results to {rank_results_path}")
    
    # Print summary
    successful = len([r for r in results if r['success']])
    failed = len([r for r in results if not r['success']])
    logger.info(f"Rank {args.rank} complete! Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        failed_items = [r['sha256'] for r in results if not r['success']]
        logger.info(f"Failed items: {failed_items}")


if __name__ == "__main__":
    main()