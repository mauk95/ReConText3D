#!/usr/bin/env python3
"""
Create metadata.csv for generated GLB assets to enable rendering with the evaluation pipeline.
"""

import os
import argparse
import pandas as pd


def create_generated_metadata(generated_assets_dir, eval_split_file, output_dir, ext='.glb'):
    """
    Create metadata.csv for generated GLB assets.

    Args:
        generated_assets_dir: Path to directory containing generated asset files.
        eval_split_file: CSV file containing evaluation split with a sha256 column.
        output_dir: Output directory to save metadata.csv.
        ext: File extension for generated assets.
    """

    eval_df = pd.read_csv(eval_split_file)

    if 'sha256' not in eval_df.columns:
        raise ValueError(
            f"CSV must contain a 'sha256' column. Found columns: {list(eval_df.columns)}"
        )

    eval_sha256s = eval_df['sha256'].astype(str).tolist()

    glb_files = []
    asset_dir_name = os.path.basename(generated_assets_dir)

    for filename in os.listdir(generated_assets_dir):
        if filename.endswith(ext):
            sha256 = filename.replace(ext, '')
            glb_files.append({
                'sha256': sha256,
                'local_path': f'{asset_dir_name}/{filename}',
                'rendered': False
            })

    metadata_df = pd.DataFrame(glb_files)

    metadata_df = metadata_df[metadata_df['sha256'].isin(eval_sha256s)]

    print(f"Found {len(metadata_df)} generated assets matching evaluation split")
    print(f"Total evaluation split size: {len(eval_sha256s)}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'metadata.csv')
    metadata_df.to_csv(output_path, index=False)

    print(f"Metadata saved to: {output_path}")
    return metadata_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create metadata for generated GLB assets')

    parser.add_argument(
        '--generated_assets_dir',
        type=str,
        default='Toys4k_eval_generated_assets',
        help='Directory containing generated asset files'
    )
    parser.add_argument(
        '--eval_split_file',
        type=str,
        required=True,
        help='CSV file containing evaluation split SHA256s in a sha256 column'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='datasets/Toys4k_eval_appearance_generated',
        help='Output directory for metadata'
    )
    parser.add_argument(
        '--ext',
        type=str,
        default='.glb',
        help='File extension for generated assets'
    )

    args = parser.parse_args()

    create_generated_metadata(
        args.generated_assets_dir,
        args.eval_split_file,
        args.output_dir,
        args.ext
    )