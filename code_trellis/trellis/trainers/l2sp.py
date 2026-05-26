# l2sp.py
import torch
import torch.distributed as dist
import re

def _is_norm_or_bias(name: str) -> bool:
    n = name.lower()
    return (".bias" in n) or ("norm" in n) or ("rms_norm" in n)

def _is_embedish(name: str) -> bool:
    n = name.lower()
    # covers time emb, pos emb, token/text emb, adaLN modulation MLPs
    return any(k in n for k in ["embed", "pos_embed", "t_embed", "text_encoder", "adaln_modulation"])

def make_l2sp_anchor(model: torch.nn.Module, strict: bool = False):
    """
    Snapshot current model weights to use as the L2-SP anchor (theta*).
    Returns a dict: {param_name: tensor(anchor, same dtype/device as param)} for eligible params.
    """
    anchor = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if _is_norm_or_bias(n):
                continue
            # store anchor in the *same dtype* as the live param to avoid casts each step
            anchor[n] = p.detach().clone()
    if strict and not anchor:
        raise RuntimeError("No eligible parameters found for L2-SP.")
    return anchor

# def l2sp_loss(model: torch.nn.Module, anchor: dict, lambda_l2sp: float = 1e-4):
#     """
#     Compute sum_i ||theta_i - theta*_i||^2 over eligible params present in anchor.
#     Works with fp16/fp32 mixed models (assumes anchor dtype==param dtype).
#     """
#     if lambda_l2sp <= 0.0:
#         return torch.tensor(0.0, device=next(model.parameters()).device)
#     loss = torch.zeros((), device=next(model.parameters()).device)
#     for n, p in model.named_parameters():
#         if n in anchor:
#             # param and anchor should share dtype; if not, do a safe cast
#             a = anchor[n]
#             if a.dtype != p.dtype:
#                 a = a.to(dtype=p.dtype)
#             loss = loss + torch.sum((p - a) ** 2)
#     return lambda_l2sp * loss

# def l2sp_loss_new(model: torch.nn.Module,
#               anchor: dict,
#               lambda_main: float = 1e-4,
#               lambda_embed: float = 5e-5):
#     """
#     L2-SP with separate λ for 'embed-ish' parameters.
#     """
#     if (lambda_main <= 0.0) and (lambda_embed <= 0.0):
#         return torch.zeros((), device=next(model.parameters()).device)

#     loss = torch.zeros((), device=next(model.parameters()).device)
#     for n, p in model.named_parameters():
#         a = anchor.get(n, None)
#         if a is None:
#             continue
#         a = a.to(dtype=p.dtype, device=p.device, non_blocking=True)
#         lam = (lambda_embed if _is_embedish(n) else lambda_main)
#         if lam > 0:
#             loss = loss + lam * torch.sum((p - a) ** 2)
#     return loss

def param_groups_with_weight_decay(model: torch.nn.Module, base_lr: float, wd: float):
    """
    Optional: build optimizer groups that already exclude norm/bias from WD.
    You can still use L2-SP on non-norm weights; WD and L2-SP are orthogonal.
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if _is_norm_or_bias(n) else decay).append(p)
    return [
        {"params": decay, "lr": base_lr, "weight_decay": wd},
        {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
    ]


# --- set these once (config) ---
LAMBDA_MAIN = 5e-5     # backbone self-attn & MLP
LAMBDA_COND = 1e-5     # cross-attn, adaLN, input/out, t_embedder
L2SP_WARMUP_STEPS = 3000  # steps to ramp from 0->1

# patterns for routing (compiled once)
PAT_MAIN = re.compile(r"(blocks\.\d+\.self_attn\.(to_qkv|to_out)\.weight|blocks\.\d+\.mlp\.mlp\.(0|2)\.weight)")
PAT_COND = re.compile(r"(blocks\.\d+\.cross_attn\.(to_q|to_kv|to_out)\.weight|blocks\.\d+\.adaLN_modulation\.\d+\.weight|t_embedder\.mlp\.\d+\.weight|^(input_layer|out_layer)\.weight$)")

def l2sp_loss_for_model(model, anchor_state, global_step, config=None):
    if anchor_state is None:
        return model.weight.new_tensor(0.0)

    # ramp
    alpha = min(1.0, float(global_step) / float(max(1, config.get("warmup_steps", L2SP_WARMUP_STEPS))))

    l2 = 0.0
    count = 0
    for name, p in model.named_parameters():
        if (p is None) or (not p.requires_grad):
            continue
        if name not in anchor_state:
            continue

        # skip norms/bias (your anchor builder already does this, but keep guard)
        if name.endswith(".bias") or "norm" in name or "ln" in name:
            continue

        with torch.no_grad():
            target = anchor_state[name].to(p.device)

        # route strength
        if PAT_MAIN.search(name):
            lam = config.get("lambda_main", LAMBDA_MAIN)
        elif PAT_COND.search(name):
            lam = config.get("lambda_cond", LAMBDA_COND)
        else:
            # everything else (usually safe to keep light)
            lam = config.get("lambda_cond", LAMBDA_COND)

        # mean over elements keeps scale comparable across layers
        l2 += lam * torch.mean((p - target) ** 2)
        count += 1

    if count == 0:
        return model.weight.new_tensor(0.0)

    return alpha * l2

# def ddp_broadcast_anchor(anchor: dict, src: int = 0):
#     """
#     Broadcast a nested {param_name: tensor} dict from src to all ranks.
#     Call under torch.distributed initialized process group.
#     """
#     if not dist.is_available() or not dist.is_initialized():
#         return anchor

#     world_size = dist.get_world_size()
#     rank = dist.get_rank()

#     # 1) broadcast list of keys
#     if rank == src:
#         keys = list(anchor.keys())
#         num = torch.tensor([len(keys)], dtype=torch.int64, device='cuda' if torch.cuda.is_available() else 'cpu')
#     else:
#         keys = None
#         num = torch.zeros(1, dtype=torch.int64, device='cuda' if torch.cuda.is_available() else 'cpu')

#     dist.broadcast(num, src)
#     num = int(num.item())

#     if rank != src:
#         keys = [None] * num

#     # broadcast each key length + bytes
#     for i in range(num):
#         if rank == src:
#             k_bytes = keys[i].encode('utf-8')
#             k_len = torch.tensor([len(k_bytes)], dtype=torch.int64, device=num.device)
#         else:
#             k_len = torch.zeros(1, dtype=torch.int64, device=num.device)
#         dist.broadcast(k_len, src)

#         if rank != src:
#             k_bytes = bytearray(k_len.item())
#         buf = torch.frombuffer(memoryview(k_bytes), dtype=torch.uint8)
#         buf = buf.to(device=num.device)
#         dist.broadcast(buf, src)
#         if rank != src:
#             keys[i] = bytes(buf.cpu().numpy().tobytes()).decode('utf-8')

#     # 2) broadcast tensors by key
#     out = {}
#     for k in keys:
#         if rank == src:
#             t = anchor[k]
#         else:
#             # create a placeholder tensor of correct shape/dtype later; we need meta first
#             t = anchor.get(k, None)

#         # broadcast meta: dtype code + numel + shape length + shape
#         if rank == src:
#             dtype_code = torch.tensor([t.dtype == torch.float16], dtype=torch.int64, device=num.device)
#             shape = torch.tensor(t.shape, dtype=torch.int64, device=num.device)
#         else:
#             dtype_code = torch.zeros(1, dtype=torch.int64, device=num.device)
#             shape = torch.zeros(0, dtype=torch.int64, device=num.device)

#         dist.broadcast(dtype_code, src)
#         dist.broadcast(shape, src)
#         shape = tuple(int(x) for x in shape.tolist())
#         dtype = torch.float16 if int(dtype_code.item()) == 1 else torch.float32

#         # allocate recv tensor on device
#         if rank != src:
#             t = torch.empty(shape, dtype=dtype, device=num.device)
#         dist.broadcast(t, src)
#         out[k] = t
#     return out