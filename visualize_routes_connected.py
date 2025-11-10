#!/usr/bin/env python3
# visualize_routes_connected.py
import argparse, pickle, math, csv
from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt

# ---------- IO helpers ----------
def load_edges(edges_csv):
    """Return GN (DiGraph), eid->(u,v), (u,v)->eid, t dict."""
    G = nx.DiGraph()
    eid_to_uv, uv_to_eid, t = {}, {}, {}
    with open(edges_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # tolerant to different headers
    has_eid = "eid" in rows[0]
    for idx, r in enumerate(rows):
        u = int(r.get("u") or r.get("src") or r.get("i"))
        v = int(r.get("v") or r.get("dst") or r.get("j"))
        eid = int(r["eid"]) if has_eid else idx
        tt = float(r.get("travel_time_min") or r.get("t") or r.get("time") or 1.0)
        G.add_edge(u, v, t=tt, eid=eid)
        eid_to_uv[eid] = (u, v)
        uv_to_eid[(u, v)] = eid
        t[(u, v)] = tt
    return G, eid_to_uv, uv_to_eid, t

def load_solution(pkl_path, eid_to_uv):
    """
    Accepts either:
      - dict-like with 'routes_uv' or 'routes_eid'
      - tuple (best_cost, routes, perm) where routes is list of lists of eids
    Returns: routes_uv: dict k -> list[(u,v)], deadheads_uv: dict k -> list[(u,v)]
    """
    with open(pkl_path, "rb") as f:
        sol = pickle.load(f)

    routes_uv = defaultdict(list)
    dead_uv   = defaultdict(list)

    # case 1: dict with explicit keys
    if isinstance(sol, dict):
        if "routes_uv" in sol:
            for k, uvlist in sol["routes_uv"].items():
                routes_uv[int(k)] = [(int(u), int(v)) for (u, v) in uvlist]
        elif "routes_eid" in sol:
            for k, eids in sol["routes_eid"].items():
                routes_uv[int(k)] = [eid_to_uv[int(e)] for e in eids]
        if "deadheads_uv" in sol:
            for k, uvlist in sol["deadheads_uv"].items():
                dead_uv[int(k)] = [(int(u), int(v)) for (u, v) in uvlist]
        return routes_uv, dead_uv

    # case 2: tuple from memetic-like dump
    if isinstance(sol, tuple) and len(sol) >= 2:
        _, routes = sol[0], sol[1]
        # routes may be list of lists of eids
        if routes and isinstance(routes[0], (list, tuple)) and routes and all(isinstance(x, (list, tuple)) for x in routes):
            for k, eids in enumerate(routes):
                routes_uv[k] = [eid_to_uv[int(e)] for e in eids]
        # or already uv arcs
        elif routes and isinstance(routes[0], (list, tuple)) and all(len(x)==2 for x in routes):
            # single route list → put into k=0
            routes_uv[0] = [(int(u), int(v)) for (u, v) in routes]
        return routes_uv, dead_uv

    # fallback empty
    return routes_uv, dead_uv

# ---------- graph utilities ----------
def ordered_walk_from_arcs(arcs):
    """
    Recover an ordered trail that uses each arc once.
    Works for one weakly connected component with either 0/0 or 1/1 degree imbalance.
    Returns (ordered_edges, start_node, end_node)
    """
    if not arcs:
        return [], None, None
    G = nx.DiGraph()
    for u, v in arcs:
        G.add_edge(u, v)
    outd = G.out_degree()
    ind  = G.in_degree()
    start = None
    # pick +1 imbalance if exists, else any node with out>0
    for v in G.nodes():
        if outd[v] - ind[v] == 1:
            start = v; break
    if start is None:
        for v in G.nodes():
            if outd[v] > 0:
                start = v; break
    # Hierholzer-style trail
    succ = {u: deque(G.successors(u)) for u in G.nodes()}
    stack = [start]; path_nodes = []
    while stack:
        u = stack[-1]
        if succ[u]:
            v = succ[u].popleft()
            stack.append(v)
        else:
            path_nodes.append(stack.pop())
    path_nodes.reverse()
    ordered = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
    return ordered, (ordered[0][0] if ordered else start), (ordered[-1][1] if ordered else start)

def build_pos(G, seed=42, layout="spring"):
    if layout == "kamada":
        return nx.kamada_kawai_layout(G)
    return nx.spring_layout(G, seed=seed, k=0.7 / math.sqrt(max(1, G.number_of_nodes())))

def edge_curvature(u, v, twin_exists, base=0.18):
    """Return connectionstyle string for curved edges.
       Curve forward/backward opposite if antiparallel exists."""
    if not twin_exists:
        return "arc3,rad=0.0"
    # deterministic opposite curve using node id ordering
    return f"arc3,rad={base if u < v else -base}"

# ---------- drawing ----------
def draw_routes(G, routes_uv, dead_uv, pos, outfile="routes.png", max_routes=None, title=None):
    plt.figure(figsize=(9, 7), dpi=160)
    ax = plt.gca(); ax.axis("off")

    # base graph light gray
    base_edges = list(G.edges())
    # precompute twin map
    has_twin = {(u, v): G.has_edge(v, u) for (u, v) in base_edges}

    # draw base edges thin
    for (u, v) in base_edges:
        cs = edge_curvature(u, v, has_twin[(u, v)], base=0.10)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               edge_color="#D0D0D0", width=0.6,
                               arrows=True, arrowsize=6,
                               connectionstyle=cs, ax=ax)

    # node markers faint
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color="#666666", alpha=0.6)

    # color palette
    palette = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
    ]

    # draw each route
    items = sorted(routes_uv.items(), key=lambda kv: kv[0])
    if max_routes is not None:
        items = items[:max_routes]

    for ridx, (k, arcs) in enumerate(items):
        color = palette[ridx % len(palette)]

        # ordered path
        ordered, s, t = ordered_walk_from_arcs(arcs)

        # draw service arcs (thick)
        for (u, v) in arcs:
            cs = edge_curvature(u, v, has_twin.get((u, v), False), base=0.22)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                                   edge_color=color, width=2.0,
                                   arrows=True, arrowsize=10,
                                   connectionstyle=cs, ax=ax)

        # overlay order numbers lightly along the ordered trail
        for i, (u, v) in enumerate(ordered):
            xm = 0.5 * (pos[u][0] + pos[v][0])
            ym = 0.5 * (pos[u][1] + pos[v][1])
            ax.text(xm, ym, str(i), fontsize=6, color=color, alpha=0.8)

        # start/end markers
        if s is not None:
            nx.draw_networkx_nodes(G, pos, nodelist=[s], node_size=80,
                                   node_color=color, edgecolors="k", linewidths=0.6)
        if t is not None:
            nx.draw_networkx_nodes(G, pos, nodelist=[t], node_size=80,
                                   node_color="#000000", edgecolors="k", linewidths=0.6)

        # draw deadheads for this route, if present (dashed)
        dd = dead_uv.get(k, [])
        for (u, v) in dd:
            cs = edge_curvature(u, v, has_twin.get((u, v), False), base=0.26)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                                   style="dashed", edge_color=color, width=1.2,
                                   arrows=True, arrowsize=8,
                                   connectionstyle=cs, alpha=0.8, ax=ax)

    if title:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    print(f"Wrote {outfile}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True, help="edges.csv with u,v,(optional eid),travel_time_min")
    ap.add_argument("--pkl", required=True, help="memetic_results_from_mip.pkl (or similar)")
    ap.add_argument("--layout", choices=["spring","kamada"], default="spring")
    ap.add_argument("--max_routes", type=int, default=None)
    ap.add_argument("--outfile", default="routes.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    G, eid_to_uv, uv_to_eid, t = load_edges(args.edges)
    routes_uv, dead_uv = load_solution(args.pkl, eid_to_uv)
    if not routes_uv:
        raise RuntimeError("No routes parsed from pickle. Check file format.")

    pos = build_pos(G, seed=args.seed, layout=args.layout)
    draw_routes(G, routes_uv, dead_uv, pos, outfile=args.outfile,
                max_routes=args.max_routes,
                title=f"Routes: {len(routes_uv)} (deadheads dashed)")

if __name__ == "__main__":
    main()
