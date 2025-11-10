#!/usr/bin/env python3
# synthetic_memetic.py  — gpickle-only memetic CARP warm start
import os, sys, math, random, pickle as pkl
from typing import List, Tuple, Dict
from collections import defaultdict

import networkx as nx
import numpy as np

from multiprocessing import Pool

# ----------------------- knobs -----------------------
random.seed(0)
MAX_ROUTE_MIN = 120.0         # per-route cap in minutes
ROUTE_PENALTY_MIN = 20.0     # per-route penalty inside DP split

# ----------------- worker fitness context -----------------
_HEAD = None
_TAIL = None
_SERVICE = None
_CAP = None
_RPEN = None
_H = None           # DiGraph with attr 't' = minutes
_SP = {}            # per-process SSSP cache: src -> {v: dist}

def _worker_init(head, tail, service, cap, rpen, H):
    """Per-process immutable context."""
    global _HEAD, _TAIL, _SERVICE, _CAP, _RPEN, _H, _SP
    _HEAD, _TAIL, _SERVICE = head, tail, service
    _CAP, _RPEN, _H = float(cap), float(rpen), H
    _SP = {}

def _deadhead(u, v):
    """Shortest-path minutes on H using cached single-source Dijkstra."""
    if u not in _SP:
        _SP[u] = nx.single_source_dijkstra_path_length(_H, u, weight="t")
    return float(_SP[u].get(v, float('inf')))

def _precompute_seg_cost(order: List[int]):
    """Cost of serving order[i..j] as one open route; INF if infeasible."""
    n = len(order); INF = 1e18
    seg = [[INF]*n for _ in range(n)]
    for i in range(n):
        t = 0.0; cur = None
        for j in range(i, n):
            a = order[j]
            dh = 0.0 if cur is None else _deadhead(cur, _HEAD[a])
            if not math.isfinite(dh) or t + dh + _SERVICE[a] > _CAP:
                break
            t += dh + _SERVICE[a]
            seg[i][j] = t
            cur = _TAIL[a]
    return seg

def _split_dp_min_cost(order: List[int]):
    """DP over precomputed segment costs. Returns (cost, routes)."""
    seg = _precompute_seg_cost(order)
    n = len(order); INF = 1e18
    dp = [INF]*(n+1); prv = [-1]*(n+1)
    dp[0] = 0.0
    for j in range(1, n+1):
        best = INF; bi = -1
        for i in range(j):
            c = seg[i][j-1]
            if c < INF:
                val = dp[i] + c + _RPEN
                if val < best:
                    best, bi = val, i
        dp[j], prv[j] = best, bi
    if dp[n] >= INF:
        return float('inf'), []
    routes, j = [], n
    while j > 0:
        i = prv[j]
        routes.append(order[i:j])
        j = i
    routes.reverse()
    return dp[n], routes

def fit_perm(perm: List[int]) -> float:
    c, _ = _split_dp_min_cost(perm)
    return c

def eval_population(perms: List[List[int]], workers: int,
                    head, tail, service, cap_min, route_penalty, H) -> List[float]:
    """Parallel fitness; safe for multiprocessing."""
    if workers is None or workers <= 1:
        _worker_init(head, tail, service, cap_min, route_penalty, H)
        return [fit_perm(p) for p in perms]
    chunks = max(1, len(perms) // (workers*4) or 1)
    with Pool(processes=workers,
              initializer=_worker_init,
              initargs=(head, tail, service, cap_min, route_penalty, H)) as pool:
        return list(pool.map(fit_perm, perms, chunksize=chunks))

# -------------------- GA utilities --------------------
def nn_seed(A, head, tail):
    """Nearest-neighbor heuristic with deadhead on _H in worker; used only after init."""
    # uses local deadhead _deadhead; but seeds are built in main process too.
    # For reproducibility, we implement a version that does *not* rely on worker cache:
    def dd(u, v, H):
        # one-off single-source; cheap on small degree graphs
        try:
            sp = nx.single_source_dijkstra_path_length(H, u, weight="t")
            return float(sp.get(v, float('inf')))
        except Exception:
            return float('inf')

    # We pass H explicitly from caller for seeding
    raise RuntimeError("nn_seed without H is disabled. Use knn_seed_simple or random seeds.")

def knn_seed_simple(A: List[int], head, tail, H: nx.DiGraph, k=5) -> List[int]:
    """k-NN seeding using on-the-fly SSSP on H; robust and simple."""
    unused = set(A)
    start = random.choice(tuple(unused))
    unused.remove(start)
    perm = [start]
    cur_tail = tail[start]
    while unused:
        sp = nx.single_source_dijkstra_path_length(H, cur_tail, weight="t")
        cand = []
        for a in unused:
            d = float(sp.get(head[a], float('inf')))
            if math.isfinite(d):
                cand.append((a, d))
        cand.sort(key=lambda x: x[1])
        pool = [a for a, _ in cand[:max(1, min(k, len(cand)))]]
        nxt = random.choice(pool) if pool else random.choice(tuple(unused))
        perm.append(nxt)
        unused.remove(nxt)
        cur_tail = tail[nxt]
    return perm

def relocate(g):
    if len(g) < 3: return g
    i = random.randrange(len(g)); j = random.randrange(len(g))
    if i == j: return g
    x = g[i]; h = g[:i] + g[i+1:]
    return h[:j] + [x] + h[j:]

def swap(g):
    if len(g) < 2: return g
    i, j = random.sample(range(len(g)), 2)
    g2 = g[:]; g2[i], g2[j] = g2[j], g2[i]
    return g2

def two_opt_segment(g):
    n = len(g)
    if n < 4: return g
    i = random.randrange(n-3)
    j = random.randrange(i+2, n-1)
    return g[:i+1] + list(reversed(g[i+1:j+1])) + g[j+1:]

def local_improve(perm, eval_fn, iters=30):
    best_p = perm[:]; best = eval_fn(best_p)
    for _ in range(iters):
        for op in (relocate, swap, two_opt_segment):
            cand = op(best_p[:])
            val = eval_fn(cand)
            if val < best:
                best_p, best = cand, val
    return best_p, best

def ox_crossover(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None]*n
    child[a:b+1] = p1[a:b+1]
    fill = [x for x in p2 if x not in child]
    j = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[j]; j += 1
    return child

def mutate(p, pm=0.2):
    g = p[:]
    if random.random() < pm:
        i, j = sorted(random.sample(range(len(g)), 2))
        g[i:j] = reversed(g[i:j])
    if random.random() < pm:
        i, j = random.sample(range(len(g)), 2)
        g[i], g[j] = g[j], g[i]
    return g

# -------------------- memetic core --------------------
def memetic(A_req: List[int], head, tail, service_time,
            H_for_dd: nx.DiGraph,
            pop=30, gens=50, ls_iters=20, elite=2, workers=1,
            route_penalty_min=ROUTE_PENALTY_MIN, cap_min=MAX_ROUTE_MIN,
            seed_ratio=0.6, knn_k=5, progress=True):
    """Parallel fitness; GA operators single-threaded."""

    # ---- build seeds ----
    seeds = []
    # k-NN seeds using H_for_dd
    for _ in range(int(pop * seed_ratio)):
        perm = knn_seed_simple(A_req, head, tail, H_for_dd, k=knn_k)
        # cheap local refine with *serial* fitness using temp context
        _worker_init(head, tail, service_time, cap_min, route_penalty_min, H_for_dd)
        perm, _ = local_improve(perm, fit_perm, iters=ls_iters)
        seeds.append(perm)
    # random seeds
    for _ in range(pop - len(seeds)):
        perm = A_req[:]
        random.shuffle(perm)
        _worker_init(head, tail, service_time, cap_min, route_penalty_min, H_for_dd)
        perm, _ = local_improve(perm, fit_perm, iters=max(1, ls_iters//2))
        seeds.append(perm)

    # initial population fitness (parallel)
    fitnesses = eval_population(seeds, workers, head, tail, service_time,
                                cap_min, route_penalty_min, H_for_dd)
    P = sorted(zip(fitnesses, seeds), key=lambda x: x[0])

    best_hist = [P[0][0]]
    if progress:
        print(f"[gen 0] best={P[0][0]:.3f}")

    # ---- generations ----
    for g in range(1, gens+1):
        # produce children
        children = []
        while len(children) < pop - elite:
            a, b = random.sample(P, 2)
            c, d = random.sample(P, 2)
            p1 = a[1] if a[0] < b[0] else b[1]
            p2 = c[1] if c[0] < d[0] else d[1]
            child = ox_crossover(p1, p2)
            child = mutate(child, pm=0.25)
            children.append(child)

        # optional quick serial polish before parallel eval
        _worker_init(head, tail, service_time, cap_min, route_penalty_min, H_for_dd)
        children = [local_improve(ch, fit_perm, iters=5)[0] for ch in children]

        # evaluate children in parallel
        child_f = eval_population(children, workers, head, tail, service_time,
                                  cap_min, route_penalty_min, H_for_dd)

        # next population = elite + best children
        newP = P[:elite] + sorted(zip(child_f, children), key=lambda x: x[0])[:pop-elite]
        P = sorted(newP, key=lambda x: x[0])

        best_hist.append(P[0][0])
        if progress and (g % 1 == 0):
            print(f"[gen {g}] best={P[0][0]:.3f}")

    # finalize: reconstruct routes for best perm using worker context
    best_cost, best_perm = P[0]
    _worker_init(head, tail, service_time, cap_min, route_penalty_min, H_for_dd)
    best_cost, best_routes = _split_dp_min_cost(best_perm)
    return best_cost, best_routes, best_perm

# -------------------- graph loading (gpickle only) --------------------
def read_gpickle_any(path):
    try:
        return nx.read_gpickle(path)        # available on recent NX
    except Exception:
        with open(path, "rb") as f:
            obj = pkl.load(f)
        if not hasattr(obj, "nodes") or not hasattr(obj, "edges"):
            raise ValueError("Unpickled object is not a NetworkX graph.")
        return obj

def load_multidigraph(path, time_attr="travel_time_sec", cycle_attr="cycle_time_hr"):
    G = read_gpickle_any(path)
    if not isinstance(G, nx.MultiDiGraph):
        MG = nx.MultiDiGraph()
        MG.add_nodes_from(G.nodes(data=True))
        for u, v, d in G.edges(data=True):
            MG.add_edge(u, v, **(d or {}))
        G = MG
    # Collapse to DiGraph H with minutes
    H = nx.DiGraph()
    for u, v, k, d in G.edges(keys=True, data=True):
        tt_sec = d.get(time_attr, d.get("weight", 60.0))
        tt_min = max(1e-3, float(tt_sec)/60.0)
        if H.has_edge(u, v):
            if tt_min < H[u][v]["t"]: H[u][v]["t"] = tt_min
        else:
            H.add_edge(u, v, t=tt_min)
    return G, H

# -------------------- main --------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="NetworkX gpickle MultiDiGraph")
    ap.add_argument("--time-attr", default="travel_time_sec")
    ap.add_argument("--cycle-attr", default="cycle_time_hr")
    ap.add_argument("--min-cycle-min", type=float, default=0.0)
    ap.add_argument("--max-cycle-min", type=float, default=180.0)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--gens", type=int, default=50)
    ap.add_argument("--ls-iters", type=int, default=20)
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-pkl", default=None)
    args = ap.parse_args()

    # Load graphs
    Gm, H = load_multidigraph(args.graph, args.time-attr if hasattr(args, "time-attr") else args.time_attr,
                              args.cycle-attr if hasattr(args, "cycle-attr") else args.cycle_attr)

    # Enumerate eids and filter REQUIRED by cycle window
    head: Dict[int,int] = {}
    tail: Dict[int,int] = {}
    service_time: Dict[int,float] = {}
    E_req: List[int] = []

    eid = 0
    for u, v, k, d in Gm.edges(keys=True, data=True):
        head[eid] = int(u)
        tail[eid] = int(v)
        service_time[eid] = float(H[u][v]["t"])  # collapsed min minutes

        cyc_hr = d.get(args.cycle_attr, None)
        cyc_min = float(cyc_hr)*60.0 if cyc_hr is not None else None
        if cyc_min is not None and args.min_cycle_min <= cyc_min <= args.max_cycle_min:
            E_req.append(eid)
        eid += 1

    if not E_req:
        print("No required arcs in given cycle-time window. Adjust --min-cycle-min/--max-cycle-min.")
        sys.exit(1)

    # Depots: dummy, not used by DP splitter; still compute for completeness
    nodes = list({*head.values(), *tail.values()})
    depots = [nodes[0]]

    # Run memetic
    best_cost, routes, perm = memetic(
        E_req, head, tail, service_time,
        H_for_dd=H,
        pop=args.pop, gens=args.gens, ls_iters=args.ls_iters, elite=args.elite,
        workers=args.workers,
        route_penalty_min=ROUTE_PENALTY_MIN, cap_min=MAX_ROUTE_MIN,
        seed_ratio=0.6, knn_k=5, progress=True
    )

    # Report
    print("Number of required arcs:", len(E_req))
    print("Best cost:", best_cost)
    print("Routes:", len(routes), "Total serviced arcs:", sum(len(r) for r in routes))
    for k, r in enumerate(routes[:5]):
        print(f"Route {k}: len={len(r)} first5={r[:5]}")

    # Write pickle
    routes_eid = {k: list(map(int, routes[k])) for k in range(len(routes))}
    routes_uv  = {k: [(int(head[e]), int(tail[e])) for e in routes[k]] for k in range(len(routes))}
    out_pkl = args.out_pkl or f"memetic_results_{os.path.splitext(os.path.basename(args.graph))[0]}_m{int(MAX_ROUTE_MIN)}.pkl"
    out_obj = {"best_cost": float(best_cost), "routes_eid": routes_eid, "routes_uv": routes_uv}
    with open(out_pkl, "wb") as f:
        pkl.dump(out_obj, f)
    print(f"Wrote {out_pkl}")

if __name__ == "__main__":
    main()
