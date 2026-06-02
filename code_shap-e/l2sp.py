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
    # print(f"global_step: {global_step}")
    # print(f"L2SP_WARMUP_STEPS: {L2SP_WARMUP_STEPS}")
    # print(f"alpha: {alpha}")
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

