#!/usr/bin/env python3
# schedule_routes_periodic_from_gpickle.py
import argparse, csv, math, os, pickle
from collections import deque
from typing import Dict, List, Tuple

import networkx as nx
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ---------- Graph IO and processing ----------
def _edge_minutes_from_data(d: dict) -> float:
    # priority: already-minutes 't' or 'travel_time_min'; else sec/60; fallback weight
    if "t" in d:
        return float(d["t"])
    if "travel_time_min" in d:
        return float(d["travel_time_min"])
    if "travel_time_sec" in d:
        return float(d["travel_time_sec"]) / 60.0
    if "deadhead_time_sec" in d:  # rare, but prefer service time if available
        return float(d["deadhead_time_sec"]) / 60.0
    if "weight" in d:
        w = float(d["weight"])
        return w / 60.0 if w > 10 else w
    return 1.0

def load_multidigraph(path: str) -> Tuple[nx.DiGraph, Dict[int, Tuple[int, int]]]:
    """
    Load a MultiDiGraph from gpickle and return:
      H: DiGraph with minutes 't' per (u,v) = min over parallel edges
      eid_uv: eid -> (u,v) using the same enumeration scheme as the MIP
    """
    # robust gpickle load for NX 2/3
    try:
        from networkx.readwrite.gpickle import read_gpickle
        G = read_gpickle(path)
    except Exception:
        with open(path, "rb") as fh:
            G = pickle.load(fh)
    if not isinstance(G, nx.MultiDiGraph):
        MG = nx.MultiDiGraph()
        MG.add_nodes_from(G.nodes(data=True))
        for u, v, d in G.edges(data=True):
            MG.add_edge(u, v, **(d or {}))
        G = MG

    # collapse to DiGraph with min time per (u,v)
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    eid_uv: Dict[int, Tuple[int, int]] = {}
    eid = 0
    for u, v, k, d in G.edges(keys=True, data=True):
        t = max(1e-3, _edge_minutes_from_data(d))
        if H.has_edge(u, v):
            if t < H[u][v]["t"]:
                H[u][v]["t"] = t
        else:
            H.add_edge(u, v, t=t)
        eid_uv[eid] = (int(u), int(v))
        eid += 1
    return H, eid_uv

# ---------- Routes from pickles ----------
def load_routes_from_pickles(pkl_paths: List[str]) -> Tuple[Dict[int, List[Tuple[int,int]]], Dict[int, float]]:
    """
    Merge multiple pickles into contiguous route ids 0..K-1.
    Accepts dict pickles like: {"routes_uv": {k:[(u,v),...]}, "f_by_route": {k:cycle_min}}
    Also tolerates (best_cost, routes, perm) with routes as list of (u,v).
    """
    merged_uv: Dict[int, List[Tuple[int, int]]] = {}
    merged_f: Dict[int, float] = {}
    rid = 0
    for pth in pkl_paths:
        with open(pth, "rb") as f:
            obj = pickle.load(f)

        routes_uv, f_by_route = {}, {}
        if isinstance(obj, dict):
            if "routes_uv" in obj and isinstance(obj["routes_uv"], dict):
                for k, uv in obj["routes_uv"].items():
                    routes_uv[int(k)] = [(int(u), int(v)) for (u, v) in uv]
            if "f_by_route" in obj and isinstance(obj["f_by_route"], dict):
                for k, v in obj["f_by_route"].items():
                    f_by_route[int(k)] = float(v)
        elif isinstance(obj, tuple) and len(obj) >= 2:
            routes = obj[1]
            if routes and isinstance(routes, list) and routes and all(
                isinstance(x, tuple) and len(x) == 2 for x in routes[0]
            ):
                for k, uv in enumerate(routes):
                    routes_uv[k] = [(int(u), int(v)) for (u, v) in uv]

        for ok in sorted(routes_uv.keys()):
            merged_uv[rid] = routes_uv[ok]
            merged_f[rid] = float(f_by_route.get(ok, float("nan")))
            rid += 1

    return merged_uv, merged_f

# ---------- Route utilities ----------
def ordered_walk_from_arcs(arcs: List[Tuple[int,int]]):
    """Eulerian-like stitch that respects multiplicity."""
    if not arcs:
        return [], None, None
    G = nx.DiGraph()
    for u, v in arcs:
        if G.has_edge(u, v):
            G[u][v]["m"] += 1
        else:
            G.add_edge(u, v, m=1)
    outd, ind = G.out_degree(), G.in_degree()
    start = next((n for n in G if outd[n] - ind[n] == 1), None)
    if start is None:
        start = next((n for n in G if outd[n] > 0), None)
    succ = {u: deque([v for v in G.successors(u) for _ in range(G[u][v]["m"])]) for u in G.nodes()}
    stack, trail = [start], []
    while stack:
        u = stack[-1]
        if succ.get(u) and succ[u]:
            stack.append(succ[u].popleft())
        else:
            trail.append(stack.pop())
    trail.reverse()
    ordered = [(trail[i], trail[i+1]) for i in range(len(trail)-1)]
    s = ordered[0][0] if ordered else start
    e = ordered[-1][1] if ordered else start
    return ordered, s, e

# ---------- Distances ----------
def all_pairs_shortest_time(H: nx.DiGraph):
    return {u: nx.single_source_dijkstra_path_length(H, u, weight="t") for u in H.nodes()}

# ---------- Job expansion ----------
def build_jobs(H: nx.DiGraph,
               routes_uv: Dict[int, List[Tuple[int,int]]],
               f_by_route: Dict[int, float],
               default_cycle: float,
               horizon_M: float):
    route_meta = {}
    for k, uv in routes_uv.items():
        ordered, s_node, e_node = ordered_walk_from_arcs(uv)
        if s_node is None or e_node is None:
            raise ValueError(f"Route {k}: empty or ill-formed.")
        # service time = sum of per-arc mins across lanes (already in H)
        try:
            svc = sum(H[u][v]["t"] for (u, v) in uv)
        except KeyError as e:
            raise KeyError(f"Route {k} uses arc {e} not present in collapsed graph.") from None
        fk = float(f_by_route.get(k, default_cycle))
        if not math.isfinite(fk):
            fk = float(default_cycle)
        if svc > fk + 1e-9:
            raise ValueError(f"Route {k}: service {svc:.3f} > cycle {fk:.3f}.")
        route_meta[k] = dict(start=s_node, end=e_node, svc=svc, cycle=fk)

    jobs = []
    for k, meta in route_meta.items():
        fk, svc = meta["cycle"], meta["svc"]
        copies = max(0, math.ceil(horizon_M / fk))
        for p in range(copies):
            rel = p * fk            # window start of the period
            due = min((p + 1) * fk, horizon_M)  # end of the period or horizon
            # end-of-service window: must FINISH between rel+svc and due
            rel_end = rel + svc
            if rel_end <= due + 1e-9:
                jobs.append(dict(id=len(jobs), k=k, p=p, proc=svc, rel=rel_end, due=due))
    return route_meta, jobs

# ---------- OR-Tools scheduling ----------
def solve_schedule(H: nx.DiGraph,
                   route_meta: Dict[int, dict],
                   jobs: List[dict],
                   horizon_M: float,
                   fleet_max: int = None,
                   vehicle_fixed_cost: int = 10000,
                   allow_wait: bool = True):
    if fleet_max is None:
        fleet_max = max(1, len(jobs))
    N = len(jobs)
    depot = N

    dist = all_pairs_shortest_time(H)
    start_node = [route_meta[j["k"]]["start"] for j in jobs]
    end_node   = [route_meta[j["k"]]["end"]   for j in jobs]
    svc_time   = [j["proc"] for j in jobs]
    rel        = [int(math.floor(j["rel"])) for j in jobs]
    due        = [int(math.ceil(j["due"]))  for j in jobs]

    def travel(i, j):
        if i == depot or j == depot:
            return 0
        ui, vj = end_node[i], start_node[j]
        return int(round(dist.get(ui, {}).get(vj, 10**6)))

    manager = pywrapcp.RoutingIndexManager(N + 1, fleet_max, depot)
    routing = pywrapcp.RoutingModel(manager)

    def time_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        t = travel(i, j)
        if j != depot:
            t += int(round(svc_time[j]))
        return t

    transit = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    slack = int(horizon_M if allow_wait else 0)
    routing.AddDimension(
        transit,
        slack_max=slack,
        capacity=int(math.ceil(horizon_M)),
        fix_start_cumul_to_zero=True,
        name="Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # end-of-service windows
    for j in range(N):
        idx = manager.NodeToIndex(j)
        lo, hi = rel[j], due[j]
        if lo > hi:
            lo = hi
        time_dim.CumulVar(idx).SetRange(lo, hi)

    for v in range(fleet_max):
        time_dim.CumulVar(routing.Start(v)).SetRange(0, int(math.ceil(horizon_M)))
        time_dim.CumulVar(routing.End(v)).SetRange(0, int(math.ceil(horizon_M)))
        routing.SetFixedCostOfVehicle(int(vehicle_fixed_cost), v)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(60)
    params.log_search = True

    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("No schedule found. Increase horizon, allow waiting, or raise fleet_max.")

    schedule, used = [], 0
    for v in range(fleet_max):
        idx = routing.Start(v)
        if routing.IsEnd(sol.Value(routing.NextVar(idx))):
            continue
        used += 1
        timeline = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            tstart = sol.Value(time_dim.CumulVar(idx))
            nxt = sol.Value(routing.NextVar(idx))
            if node != depot:
                j = node
                dur = int(round(svc_time[j]))
                timeline.append(dict(job=j, k=jobs[j]["k"], p=jobs[j]["p"],
                                     start=tstart, dur=dur, end=tstart + dur))
            idx = nxt
        schedule.append(dict(vehicle=v, tasks=timeline))
    return schedule, used

# ---------- Export ----------
def write_csv(schedule, jobs, out_csv):
    with open(out_csv, "w", newline="") as f:
        W = csv.writer(f)
        W.writerow(["vehicle","job_id","route_k","copy_p","start","dur","end"])
        for s in schedule:
            v = s["vehicle"]
            for t in s["tasks"]:
                j = t["job"]
                W.writerow([v, j, jobs[j]["k"], jobs[j]["p"], t["start"], t["dur"], t["end"]])

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="NetworkX gpickle MultiDiGraph (H.gpickle)")
    ap.add_argument("--pkls", nargs="+", required=True, help="One or more MIP result pickles to merge")
    ap.add_argument("--default_cycle", type=float, default=180.0)
    ap.add_argument("--horizon", type=float, default=360.0)
    ap.add_argument("--fleet_max", type=int, default=None)
    ap.add_argument("--veh_fixed_cost", type=float, default=10000.0)
    ap.add_argument("--out_csv", default="truck_schedule.csv")
    ap.add_argument("--allow_wait", action="store_true")
    args = ap.parse_args()

    H, _ = load_multidigraph(args.graph)  # unpack; we need the collapsed DiGraph H
    routes_uv, f_by_route = load_routes_from_pickles(args.pkls)
    if not routes_uv:
        raise SystemExit("No routes found in provided pickles.")

    route_meta, jobs = build_jobs(H, routes_uv, f_by_route, args.default_cycle, args.horizon)
    schedule, used = solve_schedule(
        H, route_meta, jobs, args.horizon,
        fleet_max=args.fleet_max,
        vehicle_fixed_cost=int(round(args.veh_fixed_cost)),
        allow_wait=args.allow_wait,
    )

    print(f"Used trucks: {used} / max {args.fleet_max or len(jobs)}  | jobs: {len(jobs)}")
    write_csv(schedule, jobs, args.out_csv)
    print(f"Wrote {args.out_csv}")

if __name__ == "__main__":
    main()
