# Shap-E Fine-tuning 

This repository provides tools for fine-tuning Shap-E models on custom 3D shape datasets, also implementing the L2-SP (L2-Sparse Prior) regularization. The workflow consists of two main steps: (1) generating latent codes from 3D models, and (2) fine-tuning the Shap-E diffusion model using these latents with text captions.

## Installation
First, set up the environment following the official [Cap3D installation instructions](https://github.com/crockwell/Cap3D/). 

Then install the additional dependencies required for this project:

```bash
pip install wandb
```


```bash
pip install --upgrade 'typing_extensions>=4.10.0'
```


## Workflow

### Step 1: Generate Latent Codes

Before fine-tuning, you need to generate Shap-E latent codes from your 3D models. The `generate_latents.py` script processes `.blend` files (Blender format) and encodes them into latent representations.

#### Basic Usage

```bash
python shape/generate_latents.py \
    --category all \
    --output_dir ./latents_output \
    --metadata_path /path/to/metadata.csv \
    --dataset_root /path/to/blend_files
```

#### Output Format

The script generates one `.pt` file per 3D model, named using the SHA256 hash from the metadata:
```
output_dir/
  ├── <sha256_1>.pt
  ├── <sha256_2>.pt
  └── ...
```

### Step 2: Fine-tune Shap-E Model

Once you have generated the latent codes, you can fine-tune the Shap-E model using `finetune_shapE.py`.

Note: For running each approach, please remember to use the appropriate .csv

#### Basic Usages

#### 1. Training 

```bash
python finetune_shapE.py \
    --latent_code_path ./latents_output \
    --captions_csv_path ./captions.csv \
    --valid_uid_pkl_path ./valid_uids.pkl \
    --ckpt_dir ./checkpoints \
    --save_name my_finetune \
    --epoch 1000 \
    --lr 1e-5 \
    --batch_size 16
```

## Experiment Configurations

For each experiment use the follwoing metadata csv to create the captions csv.
| Experiment | Metadata CSV |
|---|---|
| Base Training | `base_train_metadata.csv` |
| Fine-tuning | `novel_train_metadata.csv` |
| L2-SP | `novel_train_metadata.csv` with `--use_l2sp` |
| Ours | `novel_recontext3d_train_metadata.csv` |
| Ours + L2-SP | `novel_recontext3d_train_metadata.csv` with `--use_l2sp` |
| Joint Training | `joint_train_metadata.csv` |

#### 2. Resuming Training

```bash
python finetune_shapE.py \
    --resume_path ./checkpoints/toys4k_finetune_latest.pth \
    --latent_code_path ./toys4k_latents \
    --captions_csv_path ./captions.csv \
    --valid_uid_pkl_path ./train_uids.pkl \
    --ckpt_dir ./checkpoints \
    --save_name toys4k_finetune \
    --wandb_run_id <previous_run_id>
```
#### Data Format Requirements

1. **Captions CSV**: A CSV file with two columns (no header):
   ```csv
   <sha256_uid>,<caption_text>
   abc123...,"A red airplane with blue wings"
   def456...,"A small car with four wheels"
   ```

2. **Valid UIDs Pickle**: A Python pickle file containing a list of UIDs (SHA256 hashes) to use for training:
   ```python
   import pickle
   valid_uids = ['abc123...', 'def456...', ...]
   with open('valid_uids.pkl', 'wb') as f:
       pickle.dump(valid_uids, f)
   ```

3. **Latent Codes**: Directory containing `.pt` files named `<sha256_uid>.pt`, each containing a PyTorch tensor with the latent code.

#### Creating the Captions CSV and UID Pickle

Use `shape/create_csv.py` to create the captions CSV and the train/validation UID pickle files from a metadata CSV.

Example:

```bash
python shape/create_csv.py \
    --metadata_path /path/to/metadata.csv \
    --category all \
    --output_file ./captions.csv \
    --train_ratio 1.0 \
    --val_ratio 0.0 \
    --create_splits
```


#### Checkpoint Files

The training script saves several types of checkpoints:

- **Best Model**: `{save_name}_best_epoch{epoch}_loss{loss}.pth` - Saved whenever a new best loss is achieved
- **Epoch Checkpoints**: `{save_name}_epoch{epoch}.pth` - Saved every 50 epochs
- **Latest Checkpoint**: `{save_name}_latest.pth` - Saved at the end of each epoch


### Step 3: Generate 3D Meshes from Text

After fine-tuning, use `text2ply_shapE.py` to generate 3D meshes (`.ply` files) from text captions using your fine-tuned model.

#### Basic Usage

```bash
python shape/text2ply_shapE.py \
    --ckpt ./checkpoints/my_finetune_best_epoch100_loss0.02.pth \
    --metadata_csv ./test_metadata.csv \
    --output_dir ./generated_meshes
```


#### Metadata CSV Format

The metadata CSV should have at least two columns:
- `sha256`: Unique identifier for each sample (used as output filename)
- `captions`: A stringified Python list of captions, e.g., `"['A red airplane', 'An aircraft']"`. The script will use the first caption.

Example CSV:
```csv
sha256,captions
abc123...,"['A red airplane with blue wings']"
def456...,"['A small car with four wheels', 'An automobile']"
```

#### Output Format

The script generates one `.ply` mesh file per row in the metadata CSV:
```
output_dir/
  ├── <sha256_1>.ply
  ├── <sha256_2>.ply
  └── ...
```

