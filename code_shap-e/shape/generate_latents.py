#!/usr/bin/env python3
"""
Generate Shap-E latent codes for Toys4k dataset
Usage: python generate_latents.py --category airplane --output_dir ./toys4k_latents
"""

import os
import sys
import argparse
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import json

# Add shap-e to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch


def load_metadata(metadata_path):
    """Load the metadata CSV file"""
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} total entries")
    return df


def get_all_categories(df):
    """Get all available categories from the dataset"""
    categories = df['file_identifier'].str.split('/').str[0].unique()
    return sorted(categories)


def filter_category(df, category):
    """Filter dataframe for specific category"""
    # Filter for the specific category
    category_df = df[df['file_identifier'].str.startswith(f"{category}/")]
    print(f"Found {len(category_df)} entries for category '{category}'")
    return category_df


def generate_latent_for_file(blend_path, device, xm, cache_dir, output_images_dir=None, verbose=False):
    """Generate latent code for a single .blend file"""
    try:
        if verbose:
            print(f"Processing: {blend_path}")
        
        # Create multimodal batch (this will render the model)
        batch = load_or_create_multimodal_batch(
            device,
            model_path=blend_path,
            mv_light_mode="basic",
            mv_image_size=256,
            cache_dir=cache_dir,
            verbose=verbose
        )
        
        # Save rendered images if output directory is provided
        if output_images_dir is not None:
            save_rendered_images(batch, blend_path, output_images_dir, verbose)
        
        # Encode to latent
        with torch.no_grad():
            latent = xm.encoder.encode_to_bottleneck(batch)
        
        return latent
        
    except Exception as e:
        print(f"Error processing {blend_path}: {str(e)}")
        return None


def save_rendered_images(batch, blend_path, output_images_dir, verbose=False):
    """Save the rendered images from the batch"""
    try:
        import os
        from PIL import Image
        import numpy as np
        
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(blend_path))[0]
        
        # Create subdirectory for this model
        model_images_dir = os.path.join(output_images_dir, base_name)
        os.makedirs(model_images_dir, exist_ok=True)
        
        # Save multiview images if available
        if 'views' in batch and batch['views']:
            views = batch['views'][0]  # Get first batch
            for i, view in enumerate(views):
                if isinstance(view, np.ndarray):
                    # Convert to PIL Image if it's a numpy array
                    if view.dtype != np.uint8:
                        view = (view * 255).astype(np.uint8)
                    img = Image.fromarray(view)
                else:
                    img = view
                
                img_path = os.path.join(model_images_dir, f"view_{i:02d}.png")
                img.save(img_path)
                
                if verbose:
                    print(f"Saved view {i}: {img_path}")
        
        # Save point cloud visualization if available
        if 'points' in batch:
            points = batch['points'][0].cpu().numpy()  # Get first batch and convert to numpy
            if points.shape[0] >= 3:  # Check if we have at least x,y,z coordinates
                # Create a simple point cloud visualization
                save_point_cloud_visualization(points, model_images_dir, base_name, verbose)
        
        if verbose:
            print(f"Saved rendered images to: {model_images_dir}")
            
    except Exception as e:
        print(f"Error saving rendered images: {str(e)}")


def save_point_cloud_visualization(points, output_dir, base_name, verbose=False):
    """Create a simple point cloud visualization"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Extract coordinates (first 3 dimensions)
        coords = points[:3, :].T  # Shape: (N, 3)
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color points based on z-coordinate
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                           c=coords[:, 2], cmap='viridis', s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Point Cloud: {base_name}')
        
        # Save the plot
        pc_path = os.path.join(output_dir, f"{base_name}_pointcloud.png")
        plt.savefig(pc_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved point cloud visualization: {pc_path}")
            
    except Exception as e:
        print(f"Error creating point cloud visualization: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Generate Shap-E latent codes for Toys4k dataset')
    parser.add_argument('--category', type=str, default='all', 
                       help='Category to process (default: all). Use "all" for all categories, specify a single category like "airplane", or multiple categories separated by commas like "airplane,car,truck"')
    parser.add_argument('--metadata_path', type=str, 
                       default='',
                       help='Path to metadata CSV file')
    parser.add_argument('--dataset_root', type=str,
                       default='',
                       help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default='./test_toys4k_latents',
                       help='Output directory for latent codes')
    parser.add_argument('--images_dir', type=str, default=None,
                       help='Output directory for rendered images (optional)')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                       help='Cache directory for intermediate files')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--max_files_per_category', type=int, default=None,
                       help='Maximum number of files per category (for testing)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Create images directory if specified
    if args.images_dir:
        os.makedirs(args.images_dir, exist_ok=True)
        print(f"Rendered images will be saved to: {args.images_dir}")
    
    # Load metadata
    df = load_metadata(args.metadata_path)
    
    # Determine categories to process
    if args.category == 'all':
        categories = get_all_categories(df)
        print(f"Processing all {len(categories)} categories: {', '.join(categories)}")
    else:
        # Handle multiple categories separated by commas
        categories = [cat.strip() for cat in args.category.split(',')]
        print(f"Processing {len(categories)} categories: {', '.join(categories)}")
    
    # Load Shap-E model
    print("Loading Shap-E model...")
    xm = load_model('transmitter', device=device)
    print("Shap-E model loaded successfully!")
    
    # Process each category
    total_successful = 0
    total_failed = 0
    
    for category in categories:
        print(f"\n{'='*60}")
        print(f"Processing category: {category}")
        print(f"{'='*60}")
        
        # Filter for this category
        category_df = filter_category(df, category)
        
        if len(category_df) == 0:
            print(f"No entries found for category '{category}', skipping...")
            continue
        
        # Limit files if specified
        if args.max_files_per_category:
            category_df = category_df.head(args.max_files_per_category)
            print(f"Limited to {len(category_df)} files for testing")
        elif args.max_files:
            category_df = category_df.head(args.max_files)
            print(f"Limited to {len(category_df)} files for testing")
        
        # Process files in this category
        successful = 0
        failed = 0
        
        print(f"Processing {len(category_df)} files in {category}...")
        
        for idx, row in tqdm(category_df.iterrows(), total=len(category_df), desc=f"Generating latents for {category}"):
            # Get file info
            file_identifier = row['file_identifier']
            sha256_id = row['sha256']
            
            # Construct full path to .blend file
            blend_path = os.path.join(args.dataset_root, file_identifier)
            
            if not os.path.exists(blend_path):
                print(f"File not found: {blend_path}")
                failed += 1
                continue
            
            # Generate latent
            cache_subdir = os.path.join(args.cache_dir, f"cache_{sha256_id}")
            os.makedirs(cache_subdir, exist_ok=True)
            
            latent = generate_latent_for_file(
                blend_path, device, xm, cache_subdir, args.images_dir, args.verbose
            )
            
            if latent is not None:
                # Save latent code
                output_path = os.path.join(args.output_dir, f"{sha256_id}.pt")
                torch.save(latent, output_path)
                successful += 1
                
                if args.verbose:
                    print(f"Saved latent: {output_path}")
            else:
                failed += 1
        
        print(f"Category {category} complete: {successful} successful, {failed} failed")
        total_successful += successful
        total_failed += failed
    
    print(f"\n{'='*60}")
    print(f"ALL CATEGORIES PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_failed}")
    print(f"Latent codes saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
