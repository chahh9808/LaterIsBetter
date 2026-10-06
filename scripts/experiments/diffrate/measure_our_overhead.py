"""Time to produce a deployable schedule with the closed-form power law and the analytic iso-GFLOPs
calibration (DiffRate's FLOP formula): no model, GPU, data, labels or gradients are involved.

    python scripts/experiments/diffrate/measure_our_overhead.py
"""
import time, math

C = 384; PATCH = 16; NUMC = 1000; N0 = 197; L = 12

def analytic_gflops(merge_kept):
    """DiffRate's flops_inference, merge-only (prune=enter). No model, pure arithmetic."""
    N = float(N0); total = 0.0
    for m in merge_kept:
        total += 4*N*C*C + 2*N*N*C
        N = min(N, float(m))
        total += 8*N*C*C
    return (total + N0*C*PATCH*PATCH*3 + C*NUMC) / 1e9

def schedule(total_removed, gamma):
    w = [((i+1)/L)**gamma for i in range(L)]; sw = sum(w)
    real = [total_removed*wi/sw for wi in w]
    fl = [int(x) for x in real]; rem = total_removed - sum(fl)
    order = sorted(range(L), key=lambda i: real[i]-fl[i], reverse=True)
    for i in range(rem): fl[order[i % L]] += 1
    mk = []; N = float(N0)
    for j in range(L):
        r = min(fl[j], int(N)-1); m = int(N-r); mk.append(m); N = m
    return mk

def calibrate(gamma, target_gf):
    lo, hi, best = 0, N0-1, None
    while lo <= hi:
        mid = (lo+hi)//2; gf = analytic_gflops(schedule(mid, gamma))
        if gf <= target_gf: best = (mid, gf); hi = mid-1
        else: lo = mid+1
    return best

# produce schedules for the paper's full operating grid (r_base in {8,10,12} x gamma in {1,2})
# (target budgets here = arbitrary iso points; cost is dominated by the binary search, not the target)
targets = [3.15, 2.90, 2.70]   # ~ our r8/r10/r12 THOP points
N_REPEAT = 100                 # average over repeats for a stable timing
t0 = time.perf_counter()
for _ in range(N_REPEAT):
    for tg in targets:
        for g in [1.0, 2.0]:
            calibrate(g, tg)
elapsed = time.perf_counter() - t0
per_grid = elapsed / N_REPEAT
per_sched = per_grid / (len(targets)*2)
print(f"produce full (r_base x gamma) grid of {len(targets)*2} schedules:")
print(f"  total {N_REPEAT} repeats = {elapsed*1000:.1f} ms")
print(f"  per grid  = {per_grid*1000:.3f} ms")
print(f"  per schedule = {per_sched*1e6:.1f} us")
print(f"  resources: CPU only, no model, no GPU, no data, no labels, no gradients")
