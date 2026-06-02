from abc import abstractmethod
import os
import time
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np

from torchvision import utils
from torch.utils.tensorboard import SummaryWriter

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler
from .l2sp import make_l2sp_anchor, ddp_broadcast_anchor
from .lora import wrap_with_lora

class Trainer:
    """
    Base class for training.
    """
    def __init__(self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode='inflat_all',
        fp16_scale_growth=1e-3,
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        num_workers=0,
        regularizer=None,
        adapter=None,
        **kwargs
    ):
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'

        self.models = models
        self.dataset = dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        self.fp16_mode = fp16_mode
        self.fp16_scale_growth = fp16_scale_growth
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck
        self.regularizer_config = regularizer
        self.adapter_config = adapter
        
        # L2SP anchor for continual learning
        self.l2sp_anchor = None        

        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        # print(f'Using {self.world_size} World Size GPUs, batch size {self.batch_size}, {self.batch_size_per_gpu} per GPU.')

        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        assert self.batch_size % self.world_size == 0, 'Batch size must be divisible by the number of GPUs.'
        assert self.batch_size_per_gpu % self.batch_split == 0, 'Batch size per GPU must be divisible by batch split.'

        # self.num_workers = num_workers if num_workers > 0 else int(np.ceil(os.cpu_count() / torch.cuda.device_count()))
        # self.num_workers = int(np.ceil(num_workers / torch.cuda.device_count())) if num_workers > 0 else num_workers
        self.num_workers = num_workers

        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)
        

        # #### Print optimizer parameter groups ########################
        # print ('\n\nOptimizer parameter groups:')
        
        # # First, let's debug what we have
        # print(f"Total master_params: {len(self.master_params)}")
        # print(f"Total optimizer params: {sum(len(g['params']) for g in self.optimizer.param_groups)}")
        
        # # Debug model_params construction (this is the source of the problem!)
        # print(f"Total model_params: {len(self.model_params)}")
        
        # # Count trainable parameters in models
        # total_trainable = 0
        # for model_name, model in self.models.items():
        #     trainable = sum(1 for p in model.parameters() if p.requires_grad)
        #     print(f"Model '{model_name}': {trainable} trainable parameters")
        #     total_trainable += trainable
        # print(f"Total trainable parameters in models: {total_trainable}")
        
        # # Debug: Show how model_params was built
        # print("\nDebugging model_params construction:")
        # reconstructed_model_params = []
        # for model_name, model in self.models.items():
        #     model_trainable = [p for p in model.parameters() if p.requires_grad]
        #     print(f"Model '{model_name}': {len(model_trainable)} trainable params")
        #     reconstructed_model_params.extend(model_trainable)
        # print(f"Reconstructed model_params would have: {len(reconstructed_model_params)} parameters")
        
        # # Check if self.model_params matches
        # if len(self.model_params) != len(reconstructed_model_params):
        #     print(f"🚨 PROBLEM: self.model_params has {len(self.model_params)} but should have {len(reconstructed_model_params)}")
        # else:
        #     print("✅ self.model_params length looks correct")
        
        # # Create mapping from master params to original parameter names
        # param_name_mapping = {}
        # param_idx = 0
        # print("\nMapping master params to model params:")
        # for model_name, model in self.models.items():
        #     for param_name, param in model.named_parameters():
        #         if param.requires_grad:
        #             if param_idx < len(self.master_params):
        #                 param_name_mapping[self.master_params[param_idx]] = f"{model_name}.{param_name}"
        #                 if param_idx < 5:  # Show first few for debugging
        #                     print(f"  {param_idx}: {model_name}.{param_name} (shape: {param.shape})")
        #                 param_idx += 1
        #             else:
        #                 print(f"WARNING: More trainable params than master params! Missing: {model_name}.{param_name}")
        
        # print(f"\nSuccessfully mapped {len(param_name_mapping)} parameters")
        
        # for i, g in enumerate(self.optimizer.param_groups):
        #     print(f'\nGroup {i}: {len(g["params"])} params, lr: {g.get("lr", None)}, weight_decay: {g.get("weight_decay", None)}')
        #     # Map master params back to original parameter names
        #     names = [param_name_mapping.get(p, f"unknown_param_{j}") for j, p in enumerate(g['params'])]
        #     print(f'  Names: {names[:10]}{"..." if len(names) > 10 else ""}')  # Show first 10 to avoid clutter
            
        # ##############################

        # Load checkpoint
        self.start_step = step if step is not None else 0
        self.step = 0
        if load_dir is not None and step is not None:
            self.load(load_dir, step)
        elif finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)
        
        # #### FOR LORA FINETUNING - Apply AFTER checkpoint loading
        if self.adapter_config is not None and self.adapter_config["name"] == 'LORA':
            if self.is_master:
                print(f"\n🔧 Applying LoRA adaptation...")
                print(f"LoRA config: {self.adapter_config.args}")
                print(f"📋 Model state before LoRA: {type(self.models['denoiser'])}")
            
            # Put model in eval mode first (PEFT recommendation)
            self.models["denoiser"].eval()

            # 2) Inject LoRA
            denoiser_with_lora, target_names = wrap_with_lora(
                self.models["denoiser"],
                **self.adapter_config.args
            )
            if self.is_master:
                print(f'✅ LoRA target modules: {target_names}')
                print(f'📊 LoRA model type: {type(denoiser_with_lora)}')

            self.models["denoiser"] = denoiser_with_lora
            
            # Validate LoRA setup
            if self.is_master:
                trainable_params = sum(p.numel() for p in denoiser_with_lora.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in denoiser_with_lora.parameters())
                print(f"📈 LoRA Parameters: {trainable_params:,} trainable / {total_params:,} total ({trainable_params/total_params*100:.2f}% trainable)")
                
                # Write complete model summary to txt file
                model_summary_path = os.path.join(self.output_dir, 'lora_model_summary.txt')
                with open(model_summary_path, 'w') as f:
                    f.write("="*80 + "\n")
                    f.write("TRELLIS MODEL SUMMARY WITH LORA ADAPTATION\n")
                    f.write("="*80 + "\n\n")
                    
                    # Overall statistics
                    f.write(f"Total Parameters: {total_params:,}\n")
                    f.write(f"Trainable Parameters: {trainable_params:,}\n")
                    f.write(f"Trainable Percentage: {trainable_params/total_params*100:.4f}%\n\n")
                    
                    # LoRA configuration
                    f.write("LoRA Configuration:\n")
                    for key, value in self.adapter_config.args.items():
                        f.write(f"  {key}: {value}\n")
                    f.write(f"\nLoRA Target Modules ({len(target_names)}):\n")
                    for i, target in enumerate(target_names, 1):
                        f.write(f"  {i:2d}. {target}\n")
                    f.write("\n")
                    
                    # Detailed parameter breakdown
                    f.write("DETAILED PARAMETER BREAKDOWN:\n")
                    f.write("=" * 150 + "\n")
                    f.write(f"{'Name':<90}{'Shape':<25}{'Type':<20}{'Trainable'}\n")
                    f.write("=" * 150 + "\n")
                    
                    total_trainable_detailed = 0
                    total_params_detailed = 0
                    
                    for name, param in denoiser_with_lora.named_parameters():
                        is_trainable = param.requires_grad
                        param_count = param.numel()
                        shape_str = str(tuple(param.shape))
                        dtype_str = str(param.dtype)
                        
                        total_params_detailed += param_count
                        if is_trainable:
                            total_trainable_detailed += param_count
                        
                        # Truncate very long names to fit the column
                        display_name = name if len(name) <= 89 else name[:86] + "..."
                        trainable_str = "✓ Yes" if is_trainable else "✗ No"
                        f.write(f"{display_name:<90}{shape_str:<25}{dtype_str:<20}{trainable_str}\n")
                    
                    f.write("=" * 150 + "\n")
                    f.write(f"{'TOTAL PARAMETERS':<90}{'':<25}{'':<20}{total_params_detailed:,}\n")
                    f.write(f"{'TRAINABLE PARAMETERS':<90}{'':<25}{'':<20}{total_trainable_detailed:,}\n")
                    f.write(f"{'TRAINABLE PERCENTAGE':<90}{'':<25}{'':<20}{total_trainable_detailed/total_params_detailed*100:.4f}%\n")
                    
                    # Model architecture summary
                    f.write(f"\nMODEL ARCHITECTURE:\n")
                    f.write("=" * 150 + "\n")
                    f.write(str(denoiser_with_lora))
                    f.write("\n" + "=" * 150 + "\n")
                    
                print(f"📝 Model summary written to: {model_summary_path}")
            
            # Rebuild optimizer and related components for LoRA parameters
            self._rebuild_optimizer_for_lora()
        
        # LoRA is already applied above after checkpoint loading
        # Switch to train mode after LoRA setup
        if self.adapter_config is not None and self.adapter_config["name"] == 'LORA':
            self.models["denoiser"].train()
            if self.is_master:
                print("🏋️ LoRA model switched to train mode")


        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, 'tb_logs'))

        if self.world_size > 1:
            self.check_ddp()
            
        if self.is_master:
            print('\n\nTrainer initialized.')
            print(self)

        # # Confirm requires_grad:
        # bad = [n for n,p in self.models["denoiser"].named_parameters() if ('blocks' in n and p.requires_grad is False)]
        # print('\nfrozen_in_blocks:', bad[:10])
        # ###########

        # # Anchor sanity: check one tensor:
        # w = dict(self.models["denoiser"].named_parameters())['blocks.0.self_attn.to_qkv.weight'].detach().cpu()
        # a = self.l2sp_anchor["denoiser"]['blocks.0.self_attn.to_qkv.weight'].cpu()
        # print('anchor_match:', torch.allclose(w, a))  # should be True right after creating the anchor
        # input()

        # ################

            
    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, 'device'):
                return model.device
        return next(list(self.models.values())[0].parameters()).device
            
    @abstractmethod
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        pass
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = ResumableSampler(
            self.dataset,
            shuffle=True,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            # num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    @abstractmethod
    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        pass
    
    @abstractmethod
    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        pass
    
    def create_l2sp_anchor(self):
        """
        Create L2SP anchor from current model weights for continual learning.
        Should be called after loading base checkpoint and before Stage-2 training.
        Should be called by all processes.
        """
        if self.regularizer_config is not None and 'lambda_main' in self.regularizer_config["args"]:
            if self.is_master:
                print("Creating L2SP anchor for continual learning...")
            
            # Create anchor for each model that requires regularization
            self.l2sp_anchor = {}
            for name, model in self.models.items():
                if hasattr(model, 'named_parameters'):  # Skip non-model objects
                    anchor = make_l2sp_anchor(model)
                    if anchor:  # Only store if there are eligible parameters
                        self.l2sp_anchor[name] = anchor
            
            if self.world_size > 1:
                # # Broadcast each model's anchor dict from rank-0
                # for name in list(self.l2sp_anchor.keys()):
                #     self.l2sp_anchor[name] = ddp_broadcast_anchor(self.l2sp_anchor[name], src=0)
                # Ensure all processes have the same anchor
                dist.barrier()
            
            if self.is_master:
                total_params = sum(len(anchor) for anchor in self.l2sp_anchor.values())
                print(f"L2SP anchor created with {total_params} parameters across {len(self.l2sp_anchor)} models")
                print (f'self.l2sp_anchor keys: {list(self.l2sp_anchor["denoiser"].keys())}')
        else:
            if self.is_master:
                print("No regularizer config found, skipping L2SP anchor creation")
    
    def _rebuild_optimizer_for_lora(self):
        """
        Rebuild optimizer and related components after LoRA adaptation.
        This is needed because LoRA changes which parameters are trainable.
        """
        if self.is_master:
            print("🔄 Rebuilding optimizer for LoRA parameters...")
        
        # This method will be implemented in BasicTrainer
        # Base class just provides the interface
        pass
    
    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        """
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image.
        """
        if hasattr(self.dataset, 'visualize_sample'):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100):
        """
        Sample images from the dataset.
        """
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=num_samples,
            num_workers=0,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        data = next(iter(dataloader))
        data = recursive_to_device(data, self.device)
        vis = self.visualize_sample(data)
        if isinstance(vis, dict):
            save_cfg = [(f'dataset_{k}', v) for k, v in vis.items()]
        else:
            save_cfg = [('dataset', vis)]
        for name, image in save_cfg:
            utils.save_image(
                image,
                os.path.join(self.output_dir, 'samples', f'{name}.jpg'),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=4, verbose=False):
        """
        Sample images from the model.
        NOTE: This function should be called by all processes.
        """
        if self.is_master:
            print(f'\nSampling {num_samples} images...', end='')

        if suffix is None:
            suffix = f'step{self.step:07d}'

        # Assign tasks
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        # samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose)
        samples = self.run_snapshot(num_samples_per_process, batch_size=self.batch_size_per_gpu, verbose=verbose)

        # Preprocess images
        for key in list(samples.keys()):
            if samples[key]['type'] == 'sample':
                vis = self.visualize_sample(samples[key]['value'])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                    del samples[key]
                else:
                    samples[key] = {'value': vis, 'type': 'image'}

        # Gather results
        if self.world_size > 1:
            for key in samples.keys():
                samples[key]['value'] = samples[key]['value'].contiguous()
                if self.is_master:
                    all_images = [torch.empty_like(samples[key]['value']) for _ in range(self.world_size)]
                else:
                    all_images = []
                dist.gather(samples[key]['value'], all_images, dst=0)
                if self.is_master:
                    samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples]

        # Save images
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'samples', suffix), exist_ok=True)
            for key in samples.keys():
                if samples[key]['type'] == 'image':
                    utils.save_image(
                        samples[key]['value'],
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        nrow=int(np.sqrt(num_samples)),
                        normalize=True,
                        value_range=self.dataset.value_range,
                    )
                elif samples[key]['type'] == 'number':
                    min = samples[key]['value'].min()
                    max = samples[key]['value'].max()
                    images = (samples[key]['value'] - min) / (max - min)
                    images = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(num_samples)),
                        normalize=False,
                    )
                    save_image_with_notes(
                        images,
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        notes=f'{key} min: {min}, max: {max}',
                    )

        if self.is_master:
            print(' Done.')

    @abstractmethod
    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        pass

    @abstractmethod
    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        pass

    @abstractmethod
    def training_losses(**mb_data):
        """
        Compute training losses.
        """
        pass
    
    def load_data(self):
        """
        Load data.
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        
        # if the data is a dict, we need to split it into multiple dicts with batch_size_per_gpu
        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                batch_size = list(data.values())[0].shape[0]
                data_list = [
                    {k: v[i * batch_size // self.batch_split:(i + 1) * batch_size // self.batch_split] for k, v in data.items()}
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError('Data must be a dict or a list of dicts.')
        
        return data_list

    @abstractmethod
    def run_step(self, data_list):
        """
        Run a training step.
        """
        pass

    def run(self):
        """
        Run training.
        """
        if self.is_master:
            print('\nStarting training...')
            self.snapshot_dataset()
        if self.step == 0:
            self.snapshot(suffix='init')
        else: # resume
            self.snapshot(suffix=f'resume_step{self.step:07d}')

        log = []
        time_last_print = 0.0
        time_elapsed = 0.0
        print (f'\n==== Starting from step {self.step} / {self.max_steps} ====\n')
        
        while self.step < self.max_steps:
            time_start = time.time()

            data_list = self.load_data()
            step_log = self.run_step(data_list)

            time_end = time.time()
            time_elapsed += time_end - time_start

            self.step += 1

            # Print progress
            if self.is_master and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                columns = [
                    f'Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)',
                    f'Elapsed: {time_elapsed / 3600:.2f} h',
                    f'Speed: {speed:.2f} steps/h',
                    f'ETA: {(self.max_steps - self.step) / speed:.2f} h',
                ]
                print(' | '.join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed

            # Check ddp
            if self.world_size > 1 and self.i_ddpcheck is not None and self.step % self.i_ddpcheck == 0:
                self.check_ddp()

            # Sample images
            if self.step % self.i_sample == 0:
                self.snapshot()

            if self.is_master:
                log.append((self.step, {}))

                # Log time
                log[-1][1]['time'] = {
                    'step': time_end - time_start,
                    'elapsed': time_elapsed,
                }

                # Log losses
                if step_log is not None:
                    log[-1][1].update(step_log)

                # Log scale
                if self.fp16_mode == 'amp':
                    log[-1][1]['scale'] = self.scaler.get_scale()
                elif self.fp16_mode == 'inflat_all':
                    log[-1][1]['log_scale'] = self.log_scale

                # Save log
                if self.step % self.i_log == 0:
                    ## save to log file
                    log_str = '\n'.join([
                        f'{step}: {json.dumps(log)}' for step, log in log
                    ])
                    with open(os.path.join(self.output_dir, 'log.txt'), 'a') as log_file:
                        log_file.write(log_str + '\n')

                    # show with mlflow
                    log_show = [l for _, l in log if not dict_any(l, lambda x: np.isnan(x))]
                    log_show = dict_reduce(log_show, lambda x: np.mean(x))
                    log_show = dict_flatten(log_show, sep='/')
                    for key, value in log_show.items():
                        self.writer.add_scalar(key, value, self.step)
                    log = []

                # Save checkpoint
                if self.step % self.i_save == 0:
                    self.save()

        if self.is_master:
            self.snapshot(suffix='final')
            self.writer.close()
            print('Training finished.')
            
    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop.
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, 'profile')),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                self.run_step()
                prof.step()
            