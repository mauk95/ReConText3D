<!-- Installation -->
## 📦 Installation

### Prerequisites
- **System**: The code is currently tested only on **Linux**.  For windows setup, you may refer to [#3](https://github.com/microsoft/TRELLIS/issues/3) (not fully tested).
- **Hardware**: An NVIDIA GPU with at least 16GB of memory is necessary. The code has been verified on NVIDIA A100 and A6000 GPUs.  
- **Software**:   
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain submodules. The code has been tested with CUDA versions 11.8 and 12.2.  
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is recommended for managing dependencies.  
  - Python version 3.8 or higher is required. 

### Installation Steps
1. Change the current working directory to code_trellis:
    ```sh
    cd code_trellis
    ```

2. Install the dependencies:
    
    **Before running the following command there are somethings to note:**
    - By adding `--new-env`, a new conda environment named `trellis` will be created. If you want to use an existing conda environment, please remove this flag.
    - By default the `trellis` environment will use pytorch 2.4.0 with CUDA 11.8. If you want to use a different version of CUDA (e.g., if you have CUDA Toolkit 12.2 installed and do not want to install another 11.8 version for submodule compilation), you can remove the `--new-env` flag and manually install the required dependencies. Refer to [PyTorch](https://pytorch.org/get-started/previous-versions/) for the installation command.
    - If you have multiple CUDA Toolkit versions installed, `PATH` should be set to the correct version before running the command. For example, if you have CUDA Toolkit 11.8 and 12.2 installed, you should run `export PATH=/usr/local/cuda-11.8/bin:$PATH` before running the command.
    - By default, the code uses the `flash-attn` backend for attention. For GPUs do not support `flash-attn` (e.g., NVIDIA V100), you can remove the `--flash-attn` flag to install `xformers` only and set the `ATTN_BACKEND` environment variable to `xformers` before running the code.
    - The installation may take a while due to the large number of dependencies. Please be patient. If you encounter any issues, you can try to install the dependencies one by one, specifying one flag at a time.
    - If you encounter any issues during the installation, feel free to open an issue.
    
    Create a new conda environment named `trellis` and install the dependencies:
    ```sh
    . ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
    ```
    The detailed usage of `setup.sh` can be found by running `. ./setup.sh --help`.
    ```sh
    Usage: setup.sh [OPTIONS]
    Options:
        -h, --help              Display this help message
        --new-env               Create a new conda environment
        --basic                 Install basic dependencies
        --train                 Install training dependencies
        --xformers              Install xformers
        --flash-attn            Install flash-attn
        --diffoctreerast        Install diffoctreerast
        --vox2seq               Install vox2seq
        --spconv                Install spconv
        --mipgaussian           Install mip-splatting
        --kaolin                Install kaolin
        --nvdiffrast            Install nvdiffrast
        --demo                  Install all dependencies for demo
    ```

## Pretrained Models

Our pretrained SS and SLat models are hosted on [Hugging Face](https://huggingface.co/mauk95/ReConText3D). You can directly load the models with their repository names in the code. For e.g, you can load our ReConText3D model, finetuned on Novel Set:
```python
TrellisTextTo3DPipeline.from_pretrained("mauk95/ReConText3D/checkpoints/TRELLIS-text-xlarge/novel_recontext3d")
```

If you prefer loading the model from local, you can download the model files from the links above and load the model with the folder path (folder structure should be maintained):
```python
TrellisTextTo3DPipeline.from_pretrained("/path/to/TRELLIS-text-xlarge/novel_recontext3d")
```

## Training

### Training Setup

1. **Prepare the Environment:**
   - Ensure all training dependencies are installed.
   - Use a Linux system with an NVIDIA GPU (The models are trained on NVIDIA A100 GPUs).

2. **Dataset Preparation:**
   - Organize your dataset similar to [TRELLIS-500K](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md). Specify your dataset path using the `--data_dir` argument when launching training.
   - Make sure that you already have ran Step 4 - Step 8 and have the encoded Sparse Structures and SLATs inside the `data_dir`.
   - Place the metadata csv file for the relevant experiment inside the `--data_dir`.

3. **Configuration Files:**
   - Training hyperparameters and model architectures are defined in configuration files under the `configs/` directory.
   - Example configuration files include:

| Config | Pretained Model | Description |
| --- | --- | --- |
| [`vae/ss_vae_conv3d_16l8_fp16.json`](configs/vae/ss_vae_conv3d_16l8_fp16.json) | [Encoder](https://huggingface.co/microsoft/TRELLIS-image-large/blob/main/ckpts/ss_enc_conv3d_16l8_fp16.safetensors) [Decoder](https://huggingface.co/microsoft/TRELLIS-image-large/blob/main/ckpts/ss_dec_conv3d_16l8_fp16.safetensors) | Sparse structure VAE |
| [`vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json`](configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json) | [Encoder](https://huggingface.co/microsoft/TRELLIS-image-large/blob/main/ckpts/slat_enc_swin8_B_64l8_fp16.safetensors) [Decoder](https://huggingface.co/microsoft/TRELLIS-image-large/blob/main/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors) | SLat VAE with Gaussian Decoder |
| [`vae/slat_vae_dec_rf_swin8_B_64l8_fp16.json`](configs/vae/slat_vae_dec_rf_swin8_B_64l8_fp16.json) | [Decoder](https://huggingface.co/microsoft/TRELLIS-image-large/blob/main/ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.safetensors) | SLat Radiance Field Decoder |
| [`vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json`](configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json) | [Decoder](https://huggingface.co/microsoft/TRELLIS-image-large/blob/main/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors) | SLat Mesh Decoder |
| [`generation/base_ss_flow_txt_dit_XL_16l8_fp16.json`](configs/generation/base_ss_flow_txt_dit_XL_16l8_fp16.json) | [Denoiser](https://huggingface.co/mauk95/ReConText3D/blob/main/checkpoints/TRELLIS-text-xlarge/base_training/ckpts/ss_flow_txt_dit_XL_16l8_fp16_step0360000.safetensors) | Base Training text-conditioned sparse structure Flow Model |
| [`generation/base_slat_flow_txt_dit_XL_64l8p2_fp16.json`](configs/generation/base_slat_flow_txt_dit_XL_64l8p2_fp16.json) | [Denoiser](https://huggingface.co/mauk95/ReConText3D/blob/main/checkpoints/TRELLIS-text-xlarge/base_training/ckpts/slat_flow_txt_dit_XL_64l8p2_fp16_step0150000.safetensors) | Base Training text-conditioned SLat Flow Model |
| [`generation/novel_ss_flow_txt_dit_XL_16l8_fp16.json`](configs/generation/novel_ss_flow_txt_dit_XL_16l8_fp16.json) | [Denoiser](https://huggingface.co/mauk95/ReConText3D/blob/main/checkpoints/TRELLIS-text-xlarge/novel_recontext3d/ckpts/ss_flow_txt_dit_XL_16l8_fp16_step0560000.safetensors) | ReConText3D text-conditioned sparse structure Flow Model |
| [`generation/novel_slat_flow_txt_dit_XL_64l8p2_fp16.json`](configs/generation/novel_slat_flow_txt_dit_XL_64l8p2_fp16.json) | [Denoiser](https://huggingface.co/mauk95/ReConText3D/blob/main/checkpoints/TRELLIS-text-xlarge/novel_recontext3d/ckpts/slat_flow_txt_dit_XL_64l8p2_fp16_step0270000.safetensors) | ReConText3D text-conditioned SLat Flow Model |
| [`generation/novel_l2sp_ss_flow_txt_dit_XL_16l8_fp16.json`](configs/generation/novel_l2sp_ss_flow_txt_dit_XL_16l8_fp16.json) | [Denoiser](https://huggingface.co/mauk95/ReConText3D/blob/main/checkpoints/TRELLIS-text-xlarge/novel_recontext3d_with_l2sp/ckpts/ss_flow_txt_dit_XL_16l8_fp16_step0560000.safetensors) | ReConText3D+L2-SP text-conditioned sparse structure Flow Model |
| [`generation/novel_l2sp_slat_flow_txt_dit_XL_64l8p2_fp16.json`](configs/generation/novel_l2sp_slat_flow_txt_dit_XL_64l8p2_fp16.json) | [Denoiser](https://huggingface.co/mauk95/ReConText3D/blob/main/checkpoints/TRELLIS-text-xlarge/novel_recontext3d_with_l2sp/ckpts/slat_flow_txt_dit_XL_64l8p2_fp16_step0270000.safetensors) | ReConText3D+L2-SP text-conditioned SLat Flow Model |

### Training Commands

#### Base Training

To train a text-to-3D SS Flow model on **Base class set** with a single machine.
```sh
python train.py \
  --config configs/generation/base_ss_flow_txt_dit_XL_16l8_fp16.json \
  --output_dir outputs/base_ss_flow_txt_dit_XL_16l8_fp16_1node \
  --data_dir /path/to/Toys4K-CL/dataset \
  --metadata_filename base_train_metadata.csv \
  --num_gpus $NUM_GPUS \
  --num_workers $NUM_CPUS
```

To train a text-to-3D SLat Flow model on **Base class set** with a single machine.
```sh
python train.py \
  --config configs/generation/base_slat_flow_txt_dit_XL_64l8p2_fp16.json \
  --output_dir outputs/base_slat_flow_txt_dit_XL_64l8p2_fp16_1node \
  --data_dir /path/to/Toys4K-CL/dataset \
  --metadata_filename base_train_metadata.csv \
  --num_gpus $NUM_GPUS \
  --num_workers $NUM_CPUS
```

#### Finetuning

To finetune a text-to-3D Base SS Flow model on **Novel class set** with a single machine.
```sh
python train.py \
  --config configs/generation/novel_ft_ss_flow_txt_dit_XL_16l8_fp16.json \
  --output_dir outputs/novel_ft_ss_flow_txt_dit_XL_16l8_fp16_1node \
  --data_dir /path/to/Toys4K-CL/dataset \
  --metadata_filename novel_train_metadata.csv \
  --load_dir path/to/your/pretrained/base_ss_flow/checkpoint \
  --ckpt 360000 \
  --num_gpus $NUM_GPUS \
  --num_workers $NUM_CPUS
```

To finetune a text-to-3D Base SLat Flow model on **Novel class set** with a single machine.
```sh
python train.py \
  --config configs/generation/novel_ft_slat_flow_txt_dit_XL_64l8p2_fp16.json \
  --output_dir outputs/novel_ft_slat_flow_txt_dit_XL_64l8p2_fp16_1node \
  --data_dir /path/to/Toys4K-CL/dataset \
  --metadata_filename novel_train_metadata.csv \
  --load_dir path/to/your/pretrained/base_slat_flow/checkpoint \
  --ckpt 150000 \
  --num_gpus $NUM_GPUS \
  --num_workers $NUM_CPUS
```

#### ReConText3D (Ours)

To finetune a text-to-3D Base SS Flow model on **Novel class set** with a single machine.
```sh
python train.py \
  --config configs/generation/novel_ft_ss_flow_txt_dit_XL_16l8_fp16.json \
  --output_dir outputs/novel_ft_ss_flow_txt_dit_XL_16l8_fp16_1node \
  --data_dir /path/to/Toys4K-CL/dataset \
  --metadata_filename novel_recontext3d_train_metadata.csv \
  --load_dir path/to/your/pretrained/base_ss_flow/checkpoint \
  --ckpt 360000 \
  --num_gpus $NUM_GPUS \
  --num_workers $NUM_CPUS
```

To finetune a text-to-3D Base SLat Flow model on **Novel class set** with a single machine.
```sh
python train.py \
  --config configs/generation/novel_ft_slat_flow_txt_dit_XL_64l8p2_fp16.json \
  --output_dir outputs/novel_ft_slat_flow_txt_dit_XL_64l8p2_fp16_1node \
  --data_dir /path/to/Toys4K-CL/dataset \
  --metadata_filename novel_recontext3d_train_metadata.csv \
  --load_dir path/to/your/pretrained/base_slat_flow/checkpoint \
  --ckpt 150000 \
  --num_gpus $NUM_GPUS \
  --num_workers $NUM_CPUS
```

#### Resuming Training

By default, training will resume from the latest saved checkpoint in the same output directory. To specify a specific checkpoint to resume from, use the `--load_dir` and `--ckpt` flags:
```sh
python train.py \
  --config configs/generation/base_ss_flow_txt_dit_XL_16l8_fp16.json \
  --output_dir outputs/base_ss_flow_txt_dit_XL_16l8_fp16_1node_resume \
  --data_dir /path/to/Toys4K-CL/dataset \
  --metadata_filename base_train_metadata.csv \
  --load_dir /path/to/your/checkpoint \
  --ckpt [step]
```

## 3D Mesh Generation

To generate the 3D meshes using the trained SS and SLat Flow models, run:

```sh
python generate_text_2_mesh.py \
  --model /path/to/pretrained/ckpts \
  --output-dir $GENERATED_ASSETS_DIR \
  --metadata $METADATA_CSV \
  --rank $RANK --world-size $WORLD_SIZE --max-workers $MAX_WORKERS
```

Description of arguments:

- `--model` : Path to directory containing /ckpts and pipeline.json.
- `--output-dir`: The path of directory to save generated .glb 3D mesh files
- `--metadata`: Path to metadata CSV file.
- `--rank`: Rank of current process for multi-job processing.
- `--world-size`: Total number of processes for multi-job processing.
- `--max-workers`: Maximum number of worker threads per process.