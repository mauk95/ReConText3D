# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Based on https://github.com/openai/shap-e
#
# This code is licensed under the MIT License.
# ==============================================================================

import os

import torch
import torch.optim as optim

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.models.configs import model_from_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from IPython import embed
from torch.utils.data.distributed import DistributedSampler
# Import L2SP loss functions from local l2sp.py file
from l2sp import make_l2sp_anchor, l2sp_loss_for_model

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import argparse

import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import pandas as pd
import csv
import time
import wandb
import random
import numpy as np

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)
from datetime import datetime


def setup_ddp(gpu, args):
    dist.init_process_group(                                   
        backend='nccl',      # backend='gloo',#                                    
        init_method='env://',     
        world_size=args.world_size,                              
        rank=gpu)

    torch.cuda.set_device(gpu)

class shapE_train_dataset(Dataset):
    def __init__(self, latent_code_path, captions_csv_path, valid_uid_pkl_path):
        self.captions = pd.read_csv(captions_csv_path, header=None)
        self.valid_uid = list(pickle.load(open(valid_uid_pkl_path,'rb')))
        self.final_uid = self.valid_uid
        self.n2idx = {}
        for i in range(len(self.captions)):
            self.n2idx[self.captions[0][i]] = i
        self.latent_code_path = latent_code_path        

    def __len__(self):
        return len(self.final_uid)

    def __getitem__(self, i):
        idx = self.n2idx[self.final_uid[i]]
        assert self.final_uid[i] == self.captions[0][idx]
        latent = torch.load(os.path.join(self.latent_code_path,self.captions[0][idx]+'.pt')).squeeze()

        return {'caption': self.captions[1][idx], 'latent': latent}


def train(rank, args):
    if args.gpus > 1:
        setup_ddp(rank, args)

    niter = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    save_name = args.save_name
    ckpt_dir = args.ckpt_dir
    
    # Create logs directory if it doesn't exist
    os.makedirs('./logs', exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    f = open('./logs/%s.csv'%save_name, 'a')
    writer = csv.writer(f)
    
    # Initialize wandb (only on rank 0 for multi-GPU training)
    if args.gpus == 1 or (args.gpus > 1 and rank == 0):
        wandb_init_kwargs = {
            "project": getattr(args, 'wandb_project', 'shape'),
            "name": getattr(args, 'wandb_name', 'testing'),
            "config": {
                "epochs": niter,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "save_name": save_name,
                "gpus": args.gpus,
                "use_l2sp": args.use_l2sp,
                "lambda_main": args.lambda_main,
                "lambda_cond": args.lambda_cond,
                "l2sp_warmup_steps": args.l2sp_warmup_steps,
            }
        }
        
        # Add entity if provided
        if hasattr(args, 'wandb_entity') and args.wandb_entity != '':
            wandb_init_kwargs["entity"] = args.wandb_entity
        
        # Resume existing run if run_id is provided
        if hasattr(args, 'wandb_run_id') and args.wandb_run_id != '':
            wandb_init_kwargs["id"] = args.wandb_run_id
            wandb_init_kwargs["resume"] = "allow"
            if args.gpus == 1 or (args.gpus > 1 and rank == 0):
                print(f'Resuming wandb run: {args.wandb_run_id}')
        
        wandb.init(**wandb_init_kwargs)

    torch.manual_seed(rank+int(learning_rate*1e6)+int(datetime.now().timestamp()))

    # Check if resuming from a direct path or using resume_name
    resume_flag = False
    checkpoint_path = None
    
    if hasattr(args, 'resume_path') and args.resume_path != '' and os.path.exists(args.resume_path):
        # Use direct checkpoint path
        resume_flag = True
        checkpoint_path = args.resume_path
        print(f'Resuming from checkpoint path: {checkpoint_path}')
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if resume_flag:
        print(f'Loading checkpoint from: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract epoch and iteration from checkpoint dict if available, otherwise from filename
        if 'epoch' in checkpoint and 'iteration' in checkpoint:
            start_epoch = checkpoint['epoch']
            start_iter = checkpoint['iteration']
            print(f'Resuming from epoch {start_epoch}, iteration {start_iter} (from checkpoint dict)')
        
    else:
        start_epoch = 0
        start_iter = 0

    if not resume_flag:
        if getattr(args, 'from_scratch', False):
            print('Initializing model from config (no pretrained weights).')
            model = model_from_config(load_config('text300M'), device=device)
        else:
            model = load_model('text300M', device=device) # Original loading with pretrained weights
        # Optionally initialize weights from a provided checkpoint (fresh finetune, not resume)
        if hasattr(args, 'init_ckpt_path') and args.init_ckpt_path != '':
            print(f'initialize weights from {args.init_ckpt_path}')
            init_ckpt = torch.load(args.init_ckpt_path, map_location=device)
            model.load_state_dict(init_ckpt['model_state_dict'])
    else:
        model = model_from_config(load_config('text300M'), device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    if args.gpus > 1:
        model = DistributedDataParallel(
                model, device_ids=[rank], find_unused_parameters=False
        )
    
    # If training from scratch, L2SP is not meaningful; disable it proactively
    if getattr(args, 'from_scratch', False) and args.use_l2sp:
        if args.gpus == 1 or (args.gpus > 1 and rank == 0):
            print('from_scratch set: Disabling L2SP since no source weights are provided.')
        args.use_l2sp = False

    # Create L2SP anchor if L2SP loss is enabled
    l2sp_anchor = None
    if args.use_l2sp:
        if args.gpus == 1 or (args.gpus > 1 and rank == 0):
            print("Creating L2SP anchor for regularization...")
        # Get the actual model (unwrap DDP if needed)
        actual_model = model.module if args.gpus > 1 else model
        l2sp_anchor = make_l2sp_anchor(actual_model)
        if args.gpus == 1 or (args.gpus > 1 and rank == 0):
            print(f"L2SP anchor created with {len(l2sp_anchor)} parameters")
            
       
    
    diffusion = diffusion_from_config(load_config('diffusion'))
    my_dataset_train = shapE_train_dataset(args.latent_code_path, args.captions_csv_path, args.valid_uid_pkl_path)
    if args.gpus > 1:
        sampler = DistributedSampler(my_dataset_train)
    else:
        sampler = None
    data_loader = DataLoader(my_dataset_train, batch_size=batch_size, num_workers=8, prefetch_factor=4, shuffle=False, sampler=sampler, drop_last=True)


    optimizer= optim.AdamW(model.parameters(), lr=learning_rate)
    total_iter_per_epoch = int(len(my_dataset_train)/batch_size)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, niter*total_iter_per_epoch)
    if resume_flag:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Initialize global step counter and best loss tracking
    if resume_flag and 'global_step' in checkpoint:
        global_step = checkpoint['global_step']
        print(f'Resuming from global_step: {global_step}')
    else:
        global_step = start_epoch * total_iter_per_epoch + start_iter
    
    if resume_flag and 'best_loss' in checkpoint:
        best_loss = checkpoint['best_loss']
        print(f'Resuming with best_loss: {best_loss:.6f}')
    else:
        best_loss = float('inf')
    
    for epoch in range(start_epoch, niter):
        s = time.time()
        last_iteration = 0
        for i, data in enumerate(data_loader):
            last_iteration = i
            if i + start_iter == total_iter_per_epoch:
                start_iter = 0
                break
            s2 = time.time()
            prompt = data['caption']
            model_kwargs=dict(texts=prompt)
            t = torch.randint(0, load_config('diffusion')['timesteps'], size=(batch_size,), device=device) 
            x_start = data['latent'].cuda()

            optimizer.zero_grad()
            loss = diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs)
            diffusion_loss = torch.mean(loss['loss'])
            
            # Calculate L2SP loss if enabled using global step
            l2sp_loss_value = torch.tensor(0.0, device=device)
            if args.use_l2sp and l2sp_anchor is not None:
                # Get the actual model (unwrap DDP if needed)
                actual_model = model.module if args.gpus > 1 else model
                l2sp_config = {
                    'lambda_main': args.lambda_main,
                    'lambda_cond': args.lambda_cond,
                    'warmup_steps': args.l2sp_warmup_steps
                }
                l2sp_loss_value = l2sp_loss_for_model(actual_model, l2sp_anchor, global_step, l2sp_config)
            
            # Combine losses
            final_loss = diffusion_loss + l2sp_loss_value

            skip_step = torch.isnan(final_loss.detach()) or not torch.isfinite(final_loss.detach())
            skip_step_tensor = torch.tensor(skip_step, dtype=torch.int).to(device)
            if args.gpus > 1:
                dist.all_reduce(skip_step_tensor, op=dist.ReduceOp.SUM)
            skip_step = skip_step_tensor.item() > 0
            if skip_step:
                del final_loss
                torch.cuda.empty_cache()
            else:
                final_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                # Update global step counter
                global_step += 1
                
                if args.gpus == 1 or (args.gpus >1 and dist.get_rank() == 0):
                    print('rank: ',rank,time.time()-s2,' epoch: ', epoch, i, 'global_step:', global_step, final_loss.item())
                    # Log to wandb
                    log_dict = {
                        "train_loss": final_loss.item(),
                        "diffusion_loss": diffusion_loss.item(),
                        "l2sp_loss": l2sp_loss_value.item(),
                        "epoch": epoch,
                        "iteration": i,
                        "global_step": global_step,
                        "learning_rate": lr_scheduler.get_last_lr()[0]
                    }
                    wandb.log(log_dict)
                    
                    # Check if this is the best model so far
                    if final_loss.item() < best_loss:
                        best_loss = final_loss.item()
                        if args.gpus > 1:
                            torch.save({'model_state_dict': model.module.state_dict(), 
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'scheduler_state_dict': lr_scheduler.state_dict(),
                                        'best_loss': best_loss,
                                        'epoch': epoch,
                                        'iteration': i+start_iter,
                                        'global_step': global_step
                                        }, os.path.join(ckpt_dir, '%s_best_epoch%d_loss%.6f.pth'%(save_name, epoch, best_loss)))
                        else:
                            torch.save({'model_state_dict': model.state_dict(), 
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'scheduler_state_dict': lr_scheduler.state_dict(),
                                        'best_loss': best_loss,
                                        'epoch': epoch,
                                        'iteration': i+start_iter,
                                        'global_step': global_step
                                        }, os.path.join(ckpt_dir, '%s_best_epoch%d_loss%.6f.pth'%(save_name, epoch, best_loss)))
                        print(f'New best model saved with loss: {best_loss:.6f}')
                
                # Save checkpoint every 50 epochs
                if epoch % 50 == 0 and epoch > 0:
                    if args.gpus == 1 or (args.gpus >1 and dist.get_rank() == 0):
                        if args.gpus > 1:
                            torch.save({'model_state_dict': model.module.state_dict(), 
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'scheduler_state_dict': lr_scheduler.state_dict(),
                                        'epoch': epoch,
                                        'iteration': i+start_iter,
                                        'global_step': global_step
                                        }, os.path.join(ckpt_dir, '%s_epoch%d.pth'%(save_name, epoch)))
                        else:
                            torch.save({'model_state_dict': model.state_dict(), 
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'scheduler_state_dict': lr_scheduler.state_dict(),
                                        'epoch': epoch,
                                        'iteration': i+start_iter,
                                        'global_step': global_step
                                        }, os.path.join(ckpt_dir, '%s_epoch%d.pth'%(save_name, epoch)))
                        print(f'Checkpoint saved at epoch {epoch}')
        
        # Save latest checkpoint at the end of each epoch
        if args.gpus == 1 or (args.gpus > 1 and dist.get_rank() == 0):
            if args.gpus > 1:
                torch.save({'model_state_dict': model.module.state_dict(), 
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': lr_scheduler.state_dict(),
                            'best_loss': best_loss,
                            'epoch': epoch,
                            'iteration': last_iteration,
                            'global_step': global_step
                            }, os.path.join(ckpt_dir, '%s_latest.pth' % save_name))
            else:
                torch.save({'model_state_dict': model.state_dict(), 
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': lr_scheduler.state_dict(),
                            'best_loss': best_loss,
                            'epoch': epoch,
                            'iteration': last_iteration,
                            'global_step': global_step
                            }, os.path.join(ckpt_dir, '%s_latest.pth' % save_name))
            print(f'Latest checkpoint saved at end of epoch {epoch} (global_step: {global_step})')


    # Finish wandb run
    if args.gpus == 1 or (args.gpus > 1 and rank == 0):
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_group = parser.add_argument_group('Model settings')
    model_group.add_argument('--port', type = str, default = '12356', help = 'port for parallel')
    model_group.add_argument('--gpus', type = int, default = 1, help = 'how many gpu use')
    model_group.add_argument('--resume_name', type = str, default = 'none', help = 'any name different from "none" will resume the training')
    model_group.add_argument('--resume_path', type = str, default = '', help = 'full path to checkpoint file to resume training from (overrides resume_name)')
    model_group.add_argument('--save_name', type = str, default = 'none', help = 'name for the save file')
    model_group.add_argument('--lr', type = float, default = 1e-5, help = 'learning rate')
    model_group.add_argument('--batch_size', type = int, default = 16, help = 'batch size')
    model_group.add_argument('--epoch', type = int, default = 1000, help = 'total epoch')
    model_group.add_argument('--latent_code_path', type = str, default = '', help = 'the directory to the .pt file which store Shap-E latent codes')
    model_group.add_argument('--captions_csv_path', type = str, default = '', help = 'path to the captions CSV file')
    model_group.add_argument('--valid_uid_pkl_path', type = str, default = '', help = 'path to the valid UID pickle file')
    model_group.add_argument('--ckpt_dir', type = str, default = '', help = 'directory to save model checkpoints')
    model_group.add_argument('--init_ckpt_path', type = str, default = '', help = 'path to a checkpoint to initialize weights from (fresh finetune, not resume)')
    model_group.add_argument('--from_scratch', action='store_true', help = 'initialize randomly without pretrained weights')
    
    # L2SP loss arguments
    l2sp_group = parser.add_argument_group('L2SP loss settings')
    l2sp_group.add_argument('--use_l2sp', action='store_true', help = 'enable L2SP loss regularization')
    l2sp_group.add_argument('--lambda_main', type = float, default = 5e-5, help = 'L2SP regularization strength for main backbone parameters')
    l2sp_group.add_argument('--lambda_cond', type = float, default = 1e-5, help = 'L2SP regularization strength for conditioning parameters')
    l2sp_group.add_argument('--l2sp_warmup_steps', type = int, default = 3000, help = 'number of steps to warm up L2SP loss from 0 to full strength')
    
    # Wandb arguments
    wandb_group = parser.add_argument_group('Wandb settings')
    wandb_group.add_argument('--wandb_entity', type = str, default = 'harisamir4-dfki', help = 'wandb entity (username or team name)')
    wandb_group.add_argument('--wandb_project', type = str, default = 'shape', help = 'wandb project name')
    wandb_group.add_argument('--wandb_name', type = str, default = '', help = 'wandb run name')
    wandb_group.add_argument('--wandb_run_id', type = str, default = '', help = 'wandb run ID to resume existing run')


    args = parser.parse_args()

    if args.gpus == 1:
        train(args.gpus, args)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        args.world_size = args.gpus
        mp.spawn(train, nprocs=args.gpus, args=(args,))


