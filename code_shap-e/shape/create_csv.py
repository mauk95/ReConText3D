#!/usr/bin/env python3
"""
Create CSV file with UIDs and captions for Toys4k dataset

"""

import os
import sys
import argparse
import pandas as pd
import pickle
import random
from pathlib import Path
import csv

def load_metadata(metadata_path):
    """Load the metadata CSV file"""
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} total entries")
    return df


def filter_category(df, category):
    """Filter dataframe for specific category"""
    # Filter for the specific category
    category_df = df[df['file_identifier'].str.startswith(f"{category}/")]
    # print(f"Found {len(category_df)} entries for category '{category}'")
    return category_df


def extract_captions(captions_str):
    """Extract the first caption from the captions string"""
    try:
        import ast
           
        # Parse captions using ast.literal_eval
        captions = ast.literal_eval(captions_str) if isinstance(captions_str, str) else captions_str
        
        if isinstance(captions, list) and len(captions) > 0:
            return captions[0]
        else:
            return "A 3D model"
            
    except Exception as e:
        print(f"Error parsing captions: {e}")
        return "A 3D model"


def create_train_val_split(df, train_ratio=1, val_ratio=0, random_seed=42):
    """Create train/validation split"""
    random.seed(random_seed)
    
    total_files = len(df)
    train_size = int(total_files * train_ratio)
    val_size = total_files - train_size
    
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split
    train_df = df_shuffled[:train_size]
    val_df = df_shuffled[train_size:]
    
    # print(f"Train set: {len(train_df)} files")
    # print(f"Validation set: {len(val_df)} files")
    
    return train_df, val_df


def get_all_categories(df):
    """Get all available categories from the dataset"""
    categories = df['file_identifier'].str.split('/').str[0].unique()
    return sorted(categories)


def main():
    parser = argparse.ArgumentParser(description='Create CSV file with UIDs and captions for Toys4k dataset')
    parser.add_argument('--category', type=str, default='all', 
                       help='Category to process (default: all). Use "all" for all categories or specify a single category like "airplane"')
    parser.add_argument('--metadata_path', type=str, 
                       default='',
                       help='Path to metadata CSV file')
    parser.add_argument('--output_file', type=str, default='',
                       help='Output CSV file path')
    parser.add_argument('--train_ratio', type=float, default=1,
                       help='Ratio of data to use for training (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0,
                       help='Ratio of data to use for validation (default: 0.2)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for train/val split (default: 42)')
    parser.add_argument('--create_splits', action='store_true',
                       help='Create separate train and validation pickle files')
    
    args = parser.parse_args()
    
    # Load metadata
    df = load_metadata(args.metadata_path)
    
    # Determine categories to process
    if args.category == 'all':
        categories = get_all_categories(df)
        print(f"Processing all {len(categories)} categories: {', '.join(categories)}")
    else:
        categories = [args.category]
        print(f"Processing single category: {args.category}")
    
    # Process each category
    all_csv_data = []
    all_train_uids = []
    all_val_uids = []
    
    for category in categories:
        # print(f"\n{'='*60}")
        # print(f"Processing category: {category}")
        # print(f"{'='*60}")
        
        # Filter for this category
        category_df = filter_category(df, category)
        
        if len(category_df) == 0:
            print(f"No entries found for category '{category}', skipping...")
            continue
        
        # Create train/val split for this category
        train_df, val_df = create_train_val_split(
            category_df, 
            args.train_ratio, 
            args.val_ratio, 
            args.random_seed
        )
        
        # Process captions for this category
        # print("Processing captions...")
        
        for idx, row in category_df.iterrows():
            sha256_id = row['sha256']
            captions_str = row['captions']
            
            # Extract first caption
            caption = extract_captions(captions_str)
            
            all_csv_data.append({
                'uid': sha256_id,
                'caption': caption
            })
        
        # Add to train/val splits
        all_train_uids.extend(train_df['sha256'].tolist())
        all_val_uids.extend(val_df['sha256'].tolist())
        
        # print(f"Category {category} processed: {len(category_df)} entries")
    
    # Create the main CSV file
    print(f"\n{'='*60}")
    print("Creating final CSV file...")
    print(f"{'='*60}")
    
    csv_df = pd.DataFrame(all_csv_data)
    # Write CSV using csv module with minimal quoting
    with open(args.output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for _, row in csv_df.iterrows():
            writer.writerow([row['uid'], row['caption']])
    print(f"Saved CSV file: {args.output_file}")
    print(f"Total entries: {len(csv_df)}")
    
    # Create train/val splits if requested
    if args.create_splits:
        # Create train set
        train_file = f"./training_set_{args.category}.pkl"
        with open(train_file, 'wb') as f:
            pickle.dump(all_train_uids, f)
        print(f"Saved training set: {train_file} ({len(all_train_uids)} files)")
        
        # Create validation set
        val_file = f"./validation_set_{args.category}.pkl"
        with open(val_file, 'wb') as f:
            pickle.dump(all_val_uids, f)
        print(f"Saved validation set: {val_file} ({len(all_val_uids)} files)")
    
    
    
    print(f"\nProcessing complete! Total entries: {len(csv_df)}")


if __name__ == "__main__":
    main()
