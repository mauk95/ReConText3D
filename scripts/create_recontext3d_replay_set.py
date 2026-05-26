#!/usr/bin/env python3
"""
Build a replay memory from Base train metadata using text-embedding k-Center selection.

Outputs:
  - replay_<type>_<pct>[_percent_cap]_auto_caps.txt
  - replay_<type>_<pct>[_percent_cap]_auto_caps_metadata.csv
"""

from typing import *
import os
import json
import argparse

import clip
import torch
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Path to base_train_metadata.csv")
    ap.add_argument("--outdir", default="replay_out", help="Output dir for replay files")

    ap.add_argument("--replay_percentage", type=int, default=20,
                    help="Percentage of Novel train set to use for replay (20, 40, 60, 80)")
    ap.add_argument("--num_samples_novel", type=int, default=1242,
                    help="Number of samples in Novel train set")

    # if you set any of these >=0 we use them, else auto from pct
    ap.add_argument("--min_per_class", type=int, default=-1)
    ap.add_argument("--max_per_class", type=int, default=-1)
    ap.add_argument("--max_percentage_per_class", type=float, default=-1.0)

    ap.add_argument("--use_percentage_cap", action="store_true",
                    help="Enforce per-class percentage cap (recommended, same as before)")

    ap.add_argument("--clip_model", default="ViT-L/14")
    ap.add_argument("--selection_type", default="clip", choices=["clip", "random"])

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_captions_per_asset", type=int, default=11)

    return ap.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["sha256", "class", "captions"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {path}")
    df = df[df["captions"].notna() & (df["captions"] != "")]
    df = df[df["class"].notna() & (df["class"] != "")]
    return df


def build_class2assets(df: pd.DataFrame) -> Dict[str, List[Tuple[str, List[str]]]]:
    class2assets: Dict[str, List[Tuple[str, List[str]]]] = {}
    for _, row in df.iterrows():
        sha = str(row["sha256"])
        cls = str(row["class"])
        caps_raw = row["captions"]

        if isinstance(caps_raw, str):
            try:
                maybe = json.loads(caps_raw)
                if isinstance(maybe, list):
                    caps = [str(x).strip() for x in maybe if isinstance(x, str) and x.strip()]
                else:
                    caps = [caps_raw.strip()]
            except Exception:
                caps = [caps_raw.strip()]
        else:
            caps = [str(caps_raw).strip()]

        caps = [c for c in caps if c]
        if not caps:
            continue

        class2assets.setdefault(cls, []).append((sha, caps))
    return class2assets


def auto_caps_from_pct(pct: int):
    """
    Heuristic scaling so that:
      20% → (3, 20, 0.30)
      40% → (3, 28, 0.45)
      60% → (3, 36, 0.55)
      80% → (3, 44, 0.65)
    For >80% we just extrapolate a bit.
    """
    if pct <= 20:
        return 3, 20, 0.30
    elif pct <= 40:
        return 3, 28, 0.45
    elif pct <= 60:
        return 3, 36, 0.55
    elif pct <= 80:
        return 3, 44, 0.65
    else:
        mmin = 3
        mmax = 20 + int(0.2 * pct)
        mpct = min(0.3 + 0.004 * pct, 0.75)
        return mmin, mmax, mpct


def allocate_counts(class2num: dict, B: int, mmin: int, mmax: int) -> dict:
    """sqrt allocation without pct cap"""
    def total(alpha):
        s = 0
        for c, n in class2num.items():
            want = int(np.clip(round(alpha * np.sqrt(n)), mmin, min(mmax, n)))
            s += want
        return s

    if B <= 0 or not class2num:
        return {c: 0 for c in class2num}

    alpha = 1.0
    prev = -1
    # grow
    for _ in range(100):
        t = total(alpha)
        if t >= B or t == prev:
            break
        prev = t
        alpha *= 1.05

    # shrink
    prev = -1
    for _ in range(100):
        t = total(alpha)
        if t <= B or t == prev:
            break
        prev = t
        alpha *= 0.98

    m_c = {}
    running = 0
    for c, n in class2num.items():
        k = int(np.clip(round(alpha * np.sqrt(n)), mmin, min(mmax, n)))
        m_c[c] = k
        running += k

    # minor trim/top-up
    diff = running - B
    if diff != 0:
        if diff > 0:
            for c in sorted(m_c, key=lambda x: m_c[x], reverse=True):
                while diff > 0 and m_c[c] > mmin:
                    m_c[c] -= 1
                    diff -= 1
                    if diff == 0:
                        break
        else:
            diff = -diff
            for c in sorted(m_c, key=lambda x: m_c[x]):
                while diff > 0 and m_c[c] < min(mmax, class2num[c]):
                    m_c[c] += 1
                    diff -= 1
                    if diff == 0:
                        break

    return m_c


def allocate_counts_with_percentage_cap(
    class2num: dict,
    B: int,
    mmin: int,
    mmax: int,
    max_percentage: float
) -> dict:
    """sqrt allocation + per-class percentage cap"""
    def total(alpha):
        s = 0
        for c, n in class2num.items():
            pct_cap = int(n * max_percentage)
            eff_max = min(mmax, n, pct_cap if pct_cap > 0 else n)
            want = int(np.clip(round(alpha * np.sqrt(n)), mmin, eff_max))
            s += want
        return s

    if B <= 0 or not class2num:
        return {c: 0 for c in class2num}

    alpha = 1.0
    prev = -1
    # grow
    for _ in range(100):
        t = total(alpha)
        if t >= B or t == prev:
            break
        prev = t
        alpha *= 1.05
    # shrink
    prev = -1
    for _ in range(100):
        t = total(alpha)
        if t <= B or t == prev:
            break
        prev = t
        alpha *= 0.98

    m_c = {}
    running = 0
    for c, n in class2num.items():
        pct_cap = int(n * max_percentage)
        eff_max = min(mmax, n, pct_cap if pct_cap > 0 else n)
        k = int(np.clip(round(alpha * np.sqrt(n)), mmin, eff_max))
        m_c[c] = k
        running += k

    # minor trim/top-up
    diff = running - B
    if diff != 0:
        if diff > 0:
            for c in sorted(m_c, key=lambda x: m_c[x], reverse=True):
                while diff > 0 and m_c[c] > mmin:
                    m_c[c] -= 1
                    diff -= 1
                    if diff == 0:
                        break
        else:
            diff = -diff
            for c in sorted(m_c, key=lambda x: m_c[x]):
                pct_cap = int(class2num[c] * max_percentage)
                eff_max = min(mmax, class2num[c], pct_cap if pct_cap > 0 else class2num[c])
                while diff > 0 and m_c[c] < eff_max:
                    m_c[c] += 1
                    diff -= 1
                    if diff == 0:
                        break

    return m_c


def fill_to_budget(
    alloc: dict,
    class2num: dict,
    B: int,
    mmin: int,
    mmax: int,
    max_percentage: float,
) -> dict:
    """
    alloc: first-pass allocation (may be below B)
    Goal: push total to B by relaxing constraints.
    Steps:
      1) try to grow up to min(mmax, n_c) ignoring pct cap
      2) if still not enough, grow up to n_c
    """
    total_now = sum(alloc.values())
    if total_now >= B:
        return alloc

    deficit = B - total_now

    # 1) grow up to min(mmax, n_c) ignoring pct cap
    for c in sorted(alloc, key=lambda x: class2num[x], reverse=True):
        if deficit <= 0:
            break
        cur = alloc[c]
        n_c = class2num[c]
        cap1 = min(mmax, n_c)
        while deficit > 0 and cur < cap1:
            cur += 1
            deficit -= 1
        alloc[c] = cur

    if deficit <= 0:
        return alloc

    # 2) final grow up to n_c
    for c in sorted(alloc, key=lambda x: class2num[x], reverse=True):
        if deficit <= 0:
            break
        cur = alloc[c]
        n_c = class2num[c]
        while deficit > 0 and cur < n_c:
            cur += 1
            deficit -= 1
        alloc[c] = cur

    return alloc


def build_text_encoder(model_name: str, device: str):
    model, _ = clip.load(model_name, device=device)
    model.eval()
    return model


@torch.no_grad()
def encode_captions(model, captions, device="cuda", batch_size=64):
    all_feats = []
    for i in range(0, len(captions), batch_size):
        batch_caps = captions[i:i + batch_size]
        tokens = clip.tokenize(batch_caps).to(device)
        feats = model.encode_text(tokens).float()
        feats = torch.nn.functional.normalize(feats, dim=-1)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0).numpy()


def per_asset_embedding(captions_list, model, device, max_caps=4, batch_size=64):
    caps = captions_list[:max_caps]
    embs = encode_captions(model, caps, device=device, batch_size=batch_size)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    avg = embs.mean(axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-8)
    return avg


def kcenter_indices(embs: np.ndarray, k: int, seed: int = 42):
    N, _ = embs.shape
    if k >= N:
        return list(range(N))

    mean = embs.mean(axis=0, keepdims=True)
    mean = mean / (np.linalg.norm(mean) + 1e-8)
    sims = (embs @ mean.T).squeeze(1)
    seed_idx = int(np.argmax(sims))

    selected = [seed_idx]
    dmin = 1.0 - (embs @ embs[seed_idx].reshape(-1, 1)).squeeze(1)
    for _ in range(k - 1):
        dmin[selected] = -1.0
        nxt = int(np.argmax(dmin))
        selected.append(nxt)
        sims_new = (embs @ embs[nxt].reshape(-1, 1)).squeeze(1)
        dnew = 1.0 - sims_new
        dmin = np.minimum(dmin, dnew)
    return selected


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # budget
    budget = int(args.num_samples_novel * args.replay_percentage / 100)
    print(f"[INFO] Replay pct {args.replay_percentage}% of {args.num_samples_novel} → budget = {budget}")

    # data
    df = load_df(args.metadata)
    class2assets = build_class2assets(df)
    class2num = {c: len(v) for c, v in class2assets.items()}
    total_avail = sum(class2num.values())
    print(f"[INFO] Classes: {len(class2num)} | Captioned assets: {total_avail}")

    # caps
    if args.min_per_class < 0 or args.max_per_class < 0 or args.max_percentage_per_class < 0:
        mmin, mmax, mpct = auto_caps_from_pct(args.replay_percentage)
        print(f"[AUTO] Caps for {args.replay_percentage}% → min={mmin}, max={mmax}, max_pct={mpct:.2f}")
    else:
        mmin, mmax, mpct = args.min_per_class, args.max_per_class, args.max_percentage_per_class
        print(f"[MANUAL] Caps → min={mmin}, max={mmax}, max_pct={mpct:.2f}")

    # clamp budget
    B = min(budget, total_avail)

    # first allocation
    if args.use_percentage_cap:
        print(f"[INFO] Allocator: sqrt + pct-cap ({mpct:.2f})")
        alloc = allocate_counts_with_percentage_cap(class2num, B, mmin, mmax, mpct)
    else:
        print("[INFO] Allocator: sqrt (no pct-cap)")
        alloc = allocate_counts(class2num, B, mmin, mmax)

    sum_before = sum(alloc.values())
    print(f"[INFO] First-pass allocated: {sum_before} / {B}")

    # top-up to hit B
    if sum_before < B:
        alloc = fill_to_budget(alloc, class2num, B, mmin, mmax, mpct)
        print(f"[INFO] After top-up: {sum(alloc.values())} / {B}")

    # build encoder
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if args.selection_type == "clip":
        print(f"[INFO] Loading CLIP {args.clip_model} on {device}")
        txt_model = build_text_encoder(args.clip_model, device)
    else:
        txt_model = None

    replay_ids = []
    picked_per_class = {}

    for cls, items in sorted(class2assets.items(), key=lambda x: x[0]):
        k = alloc.get(cls, 0)
        if k <= 0:
            continue

        if args.selection_type == "random":
            idxs = np.random.choice(len(items), size=k, replace=False)
            for i in idxs:
                sha, _ = items[i]
                replay_ids.append(sha)
            picked_per_class[cls] = k
            continue

        # CLIP selection
        embs = []
        ids = []
        for sha, caps in items:
            try:
                e = per_asset_embedding(
                    caps,
                    txt_model,
                    device,
                    max_caps=args.max_captions_per_asset,
                    batch_size=args.batch_size
                )
                embs.append(e)
                ids.append(sha)
            except Exception as e:
                print(f"[WARN] Failed to encode {sha} in {cls}: {e}")

        if not embs:
            continue
        embs = np.stack(embs, axis=0)
        k = min(k, embs.shape[0])
        idxs = kcenter_indices(embs, k, seed=args.seed)
        chosen_ids = [ids[i] for i in idxs]
        replay_ids.extend(chosen_ids)
        picked_per_class[cls] = len(chosen_ids)

    # dedupe
    seen = set()
    uniq = []
    for sid in replay_ids:
        if sid not in seen:
            uniq.append(sid)
            seen.add(sid)

    print(f"[INFO] Selected replay assets: {len(uniq)} (target {B})")

    # write replay ids to txt
    if args.use_percentage_cap:
        prefix = f"replay_{args.selection_type}_{args.replay_percentage}_percent_cap_auto_caps"
    else:
        prefix = f"replay_{args.selection_type}_{args.replay_percentage}_auto_caps"

    out_txt = os.path.join(args.outdir, f"{prefix}.txt")
    with open(out_txt, "w") as f:
        for s in uniq:
            f.write(f"{s}\n")
    print(f"[OK] wrote {out_txt}")

    # write metadata,csv for replay samples only
    cols = list(df.columns)
    replay_meta = df[df["sha256"].isin(uniq)].copy()
    out_csv = os.path.join(args.outdir, f"{prefix}_metadata.csv")
    replay_meta.to_csv(out_csv, index=False, columns=cols)
    print(f"[OK] wrote {out_csv}")

    print("\n[SUMMARY] Per-class picks (nonzero):")
    lines = []
    for cls in sorted(picked_per_class):
        line = f"  {cls:20s} -> {picked_per_class[cls]} (avail={class2num[cls]})"
        print(line)
        lines.append(line)

    out_summary = os.path.join(args.outdir, f"{prefix}_class_summary.txt")
    with open(out_summary, "w") as f:
        f.write(f"Replay Selection Summary ({args.selection_type})\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total budget: {B}\n")
        f.write(f"Total selected: {len(uniq)}\n")
        f.write(f"Total classes: {len(class2num)}\n")
        f.write(f"Classes with selections: {len(picked_per_class)}\n")
        f.write(f"Auto caps: min={mmin}, max={mmax}, max_pct={mpct:.2f}\n\n")
        f.write("Per-class breakdown:\n")
        for line in lines:
            f.write(line + "\n")
    print(f"[OK] wrote {out_summary}")


if __name__ == "__main__":
    main()
