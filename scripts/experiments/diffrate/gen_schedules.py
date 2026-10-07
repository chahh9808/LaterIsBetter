"""Flat and late-concentrated (gamma=2) per-block kept-token schedules for the DiffRate model,
calibrated to the FLOPs of DiffRate's searched schedule with DiffRate's own analytic FLOP count, so
no model or GPU is needed. The prune/merge split of the searched schedule is kept. The kept-token
lists are passed to the model through set_kept_num at evaluation.

    python scripts/experiments/diffrate/gen_schedules.py
"""
import json, math

# ---- DeiT-S (vit_deit_small_patch16_224) constants ----
C = 384            # embed_dim
NUM_PATCHES = 196
N0 = NUM_PATCHES + 1   # 197 (incl. cls)
NUM_CLASSES = 1000
PATCH = 16
L = 12             # blocks
NONCOMP = 0        # block index kept uncompressed (apply_patch: non_compressed_block_index=[0])
MODEL_KEY = "ViT-S-DeiT"
TARGET = "3.1"

def flops_inference(prune_kept, merge_kept):
    """Exact replica of DiffRateVisionTransformer.calculate_flop_inference (raw FLOPs)."""
    N = float(N0)
    total = 0.0
    patch_embedding_flops = N * C * (PATCH * PATCH * 3)
    classifier_flops = C * NUM_CLASSES
    for p, m in zip(prune_kept, merge_kept):
        mhsa = 4 * N * C * C + 2 * N * N * C        # attention on tokens ENTERING the block
        total += mhsa
        N = min(N, float(p), float(m))               # tokens after prune+merge (= m, since m<=p<=N)
        ffn = 8 * N * C * C                           # FFN on reduced tokens
        total += ffn
    total += patch_embedding_flops + classifier_flops
    return total

def gflops(prune_kept, merge_kept):
    return flops_inference(prune_kept, merge_kept) / 1e9

def load_clean():
    with open("compression_rate.json") as f:
        cr = json.load(f)
    p = eval(cr[MODEL_KEY][TARGET]["prune_kept_num"])
    m = eval(cr[MODEL_KEY][TARGET]["merge_kept_num"])
    return p, m

def clean_split_fractions(clean_p, clean_m):
    """Per-block fraction of that block's total removal done by PRUNING (vs merging).
    removal_j = N_in_j - merge_j ;  prune_removed_j = N_in_j - prune_j ."""
    fracs = []
    N_in = float(N0)
    for j in range(L):
        removal = N_in - clean_m[j]
        prune_removed = N_in - clean_p[j]
        fracs.append((prune_removed / removal) if removal > 0 else 0.0)
        N_in = clean_m[j]
    return fracs

def build_schedule(total_removed, gamma, split_fracs):
    """Distribute `total_removed` tokens over compressible blocks (j=1..L-1) with
    per-block removal r_j proportional to (j/(L-1))**gamma  (gamma=0 -> flat).
    Returns integer prune_kept, merge_kept lists (block 0 uncompressed)."""
    comp = [j for j in range(L) if j != NONCOMP]            # blocks 1..11
    K = len(comp)
    # weights over compressible blocks by normalized depth
    w = [((i + 1) / K) ** gamma for i in range(K)]          # i=0..K-1 -> depth (i+1)/K
    sw = sum(w)
    # real-valued per-block removals, then largest-remainder rounding to ints summing to total_removed
    real = [total_removed * wi / sw for wi in w]
    floor = [int(math.floor(x)) for x in real]
    rem = total_removed - sum(floor)
    order = sorted(range(K), key=lambda i: real[i] - floor[i], reverse=True)
    r = floor[:]
    for i in range(rem):
        r[order[i]] += 1
    # cumulative -> merge_kept; prune_kept keeps clean split
    prune_kept = [0] * L
    merge_kept = [0] * L
    prune_kept[NONCOMP] = N0
    merge_kept[NONCOMP] = N0
    N_in = float(N0)
    ri = 0
    for j in comp:
        removal = r[ri]; ri += 1
        m_j = int(round(N_in - removal))
        m_j = max(1, min(m_j, int(N_in)))
        prune_removed = int(round(split_fracs[j] * (N_in - m_j)))
        p_j = int(N_in - prune_removed)
        p_j = max(m_j, min(p_j, int(N_in)))               # m_j <= p_j <= N_in
        prune_kept[j] = p_j
        merge_kept[j] = m_j
        N_in = m_j
    return prune_kept, merge_kept

def calibrate(gamma, split_fracs, clean_gf):
    """Smallest total_removed whose schedule FLOPs <= clean FLOPs. gf DECREASES as total
    grows, so the first hit is the largest gf <= clean (closest iso-FLOPs from below;
    conservative: never exceeds clean)."""
    for total in range(0, N0 - 1):                          # 0 .. 195
        p, m = build_schedule(total, gamma, split_fracs)
        gf = gflops(p, m)
        if gf <= clean_gf + 1e-9:
            return (total, p, m, gf)
    return None

def fmt(lst):
    return "[" + ",".join(str(x) for x in lst) + "]"

if __name__ == "__main__":
    clean_p, clean_m = load_clean()
    clean_gf = gflops(clean_p, clean_m)
    fracs = clean_split_fractions(clean_p, clean_m)
    clean_total = N0 - clean_m[-1] if False else (sum((([N0] + clean_m)[j] - clean_m[j]) for j in range(L)))

    print(f"=== DiffRate-clean (ViT-S-DeiT @{TARGET}) ===")
    print(f"  prune_kept = {fmt(clean_p)}")
    print(f"  merge_kept = {fmt(clean_m)}")
    print(f"  GFLOPs     = {clean_gf:.4f}   (should be ~3.1)")
    print(f"  per-block removal = {[ (([N0]+clean_m)[j]-clean_m[j]) for j in range(L)]}")
    print(f"  prune-split fracs = {[round(x,2) for x in fracs]}")

    for name, gamma in [("flat", 0.0), ("late(gamma=2)", 2.0)]:
        total, p, m, gf = calibrate(gamma, fracs, clean_gf)
        print(f"\n=== {name} @ iso-GFLOPs ===")
        print(f"  total_removed = {total}   GFLOPs = {gf:.4f}  (<= clean {clean_gf:.4f})")
        print(f"  prune_kept = {fmt(p)}")
        print(f"  merge_kept = {fmt(m)}")
        print(f"  per-block removal = {[ (([N0]+m)[j]-m[j]) for j in range(L)]}")
