# lora_utils.py
import re
from typing import List, Iterable
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from typing import Union
from ..modules import sparse as sp

PRIMARY_TARGET_SUBSTR = [
    "self_attn.to_qkv",
    "self_attn.to_out",
    "cross_attn.to_q",
    "cross_attn.to_kv",
    "cross_attn.to_out",
]

MLP_TARGET_SUBSTR = [
    "mlp.mlp.0",   # first FC in MLP
    "mlp.mlp.2",   # last FC in MLP
]

def _collect_linear_module_names(model: nn.Module) -> List[str]:
    """Return full qualified names of all nn.Linear modules."""
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.append(name)
    return names

def _filter_targets(all_linear_names: Iterable[str], substr_list: List[str]) -> List[str]:
    """Pick module names that contain any of the substrings."""
    picked = []
    for n in all_linear_names:
        if any(s in n for s in substr_list):
            picked.append(n)
    # Deduplicate and keep stable order
    seen = set()
    uniq = []
    for n in picked:
        if n not in seen:
            uniq.append(n); seen.add(n)
    return uniq

class PEFTCompatibleWrapper(nn.Module):
    """
    A wrapper that makes TRELLIS models compatible with PEFT by 
    providing a standard forward interface while preserving the original signature.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._original_forward = model.forward
        
        # Store the original forward method signature info
        import inspect
        sig = inspect.signature(model.forward)
        self.forward_params = list(sig.parameters.keys())
        
    def forward(self, input_ids=None, **kwargs):
        """
        PEFT-compatible forward method that translates standard transformer
        inputs back to TRELLIS model signatures.
        """
        # Filter out PEFT-specific arguments that TRELLIS models don't expect
        peft_specific_args = {'attention_mask', 'token_type_ids', 'position_ids', 'labels', 'inputs_embeds', 'output_attentions', 'output_hidden_states', 'return_dict'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in peft_specific_args}
        
        # Extract the actual arguments based on the original model's signature
        if len(self.forward_params) >= 3:
            # For models like: forward(x, t, cond)
            x = filtered_kwargs.get('x', input_ids)  # Use input_ids as x if x not provided
            t = filtered_kwargs.get('t')
            cond = filtered_kwargs.get('cond')
            
            if x is not None and t is not None and cond is not None:
                return self._original_forward(x, t, cond)
        
        # Fallback: call with filtered kwargs only
        return self._original_forward(**filtered_kwargs)

def choose_lora_targets(model, include_mlp: bool = False):
    """
    Choose the target modules for LoRA adaptation using precise substring matching.
    This function identifies specific linear layers within attention modules.
    """
    # Get all linear module names
    all_linear_names = _collect_linear_module_names(model)
    
    # Use precise targeting for attention layers
    targets = _filter_targets(all_linear_names, PRIMARY_TARGET_SUBSTR)
    
    # Optionally include MLP layers
    if include_mlp:
        mlp_targets = _filter_targets(all_linear_names, MLP_TARGET_SUBSTR)
        targets.extend(mlp_targets)
    
    # Fallback to input/output layers if no attention targets found
    if not targets:
        fallback_targets = ["input_layer", "out_layer"]
        targets = _filter_targets(all_linear_names, fallback_targets)
        if not targets:
            # Final fallback: just use first few linear layers
            targets = all_linear_names[:2] if len(all_linear_names) >= 2 else all_linear_names
    
    return targets

def wrap_with_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    bias: str = "none",
    include_mlp: bool = False,
    modules_to_save: List[str] = None,
):
    """
    Wrap a model with LoRA adapters, using a compatibility wrapper for TRELLIS models.
    
    Args:
        model: The model to wrap with LoRA
        r: LoRA rank
        alpha: LoRA alpha parameter
        dropout: LoRA dropout rate
    
    Returns:
        PEFT model with LoRA adapters
    """
    target_modules = choose_lora_targets(model, include_mlp=include_mlp)
    print(f"Applying LoRA to modules: {target_modules}")
    
    # Wrap the model for PEFT compatibility
    wrapped_model = PEFTCompatibleWrapper(model)
    
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias
    )
    
    peft_model = get_peft_model(wrapped_model, lora_config)
    
    # Create a final wrapper that restores the original interface
    class TRELLISLoRAWrapper(nn.Module):
        def __init__(self, peft_model, original_model):
            super().__init__()
            self.peft_model = peft_model
            self.original_model = original_model
            
        def forward(self, *args, **kwargs):
            """
            Restore the original TRELLIS forward signature while using PEFT internally.
            """
            if len(args) == 3:
                # Standard TRELLIS signature: forward(x, t, cond)
                x, t, cond = args
                return self.peft_model(x=x, t=t, cond=cond)
            else:
                # Pass through kwargs
                return self.peft_model(**kwargs)
                
        def __getattr__(self, name):
            # Delegate other attributes to the PEFT model
            if name in ['peft_model', 'original_model']:
                return super().__getattr__(name)
            return getattr(self.peft_model, name)
    
    final_wrapper = TRELLISLoRAWrapper(peft_model, model)
    return final_wrapper, target_modules
