# visualize_memetic_routes.py
# Usage:
#   python visualize_memetic_routes.py --pkl memetic_results_e40.pkl --edges edges.csv \
#       --max_routes 15 --outfile route_map

import argparse
import pickle
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def load_memetic(pkl_path):
    with open(pkl_path, "rb") as f:
        best_cost, routes, perm = pickle.load(f)
    return best_cost, routes, perm

def build_graph(edges_csv):
    # expects: edge_id,u,v,travel_time_min,cycle_time_h,priority
    df = pd.read_csv(edges_csv)
    G = nx.DiGraph()
    for _, r in df.iterrows():
        G.add_edge(int(r.edge_id), (int(r.u), int(r.v)))  # key by edge_id -> (u,v) mapping
    # Also a node graph for layout:
    GN = nx.DiGraph()
    for _, r in df.iterrows():
        u, v = int(r.u), int(r.v)
        GN.add_edge(u, v, edge_id=int(r.edge_id))
    return df, G, GN  # G: id->(u,v) map; GN: actual road digraph

def compute_layout(GN, seed=1):
    # If you have real coordinates, replace with pos = {node: (lon,lat), ...}
    pos = nx.spring_layout(GN.to_undirected(), seed=seed, k=1.5/(len(GN.nodes())**0.5))
    return pos

def routes_to_uv(routes, edge_map_df):
    # Build dict: route_k -> list of (u,v) tuples (service arcs only)
    id2uv = {int(r.edge_id):(int(r.u), int(r.v)) for _, r in edge_map_df.iterrows()}
    route_uv = []
    for r in routes:
        uv_list = []
        for eid in r:
            uv_list.append(id2uv[int(eid)])
        route_uv.append(uv_list)
    return route_uv

def draw_bidirected(GN, pos, edgelist=None, width=1.8, color="#1f77b4",
                    rad=0.15, arrowsize=10, zorder=2, ax=None):
    """
    Draw directed edges. Opposite directions are curved away from each other.
    If edgelist is None, draw all edges in GN.
    """
    if ax is None:
        ax = plt.gca()
    if edgelist is None:
        edgelist = list(GN.edges())

    E = set(edgelist)
    forward_curved, backward_curved, singles = [], [], []
    for (u, v) in edgelist:
        if (v, u) in E:
            # put each direction in opposite curvature bucket once
            if u < v:    # arbitrary tie-break to avoid duplication
                forward_curved.append((u, v))
                backward_curved.append((v, u))
        else:
            singles.append((u, v))

    # straight singles
    if singles:
        nx.draw_networkx_edges(
            GN, pos, edgelist=singles, width=width, edge_color=color,
            arrows=True, arrowsize=arrowsize, arrowstyle='-|>', ax=ax,
            connectionstyle="arc3,rad=0.0", min_source_margin=1, min_target_margin=1
        )

    # curved pairs
    if forward_curved:
        nx.draw_networkx_edges(
            GN, pos, edgelist=forward_curved, width=width, edge_color=color,
            arrows=True, arrowsize=arrowsize, arrowstyle='-|>', ax=ax,
            connectionstyle=f"arc3,rad={rad}", min_source_margin=1, min_target_margin=1
        )
    if backward_curved:
        nx.draw_networkx_edges(
            GN, pos, edgelist=backward_curved, width=width, edge_color=color,
            arrows=True, arrowsize=arrowsize, arrowstyle='-|>', ax=ax,
            connectionstyle=f"arc3,rad={-rad}", min_source_margin=1, min_target_margin=1
        )

def draw_edges_curved(GN, pos, edgelist, color, width=2.0, rad=0.20,
                      arrowsize=10, ax=None):
    """
    Curve u->v iff the reverse edge exists in GN. Opposite directions always get
    opposite curvature by comparing node indices, so drawing in separate layers still works.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    if ax is None:
        ax = plt.gca()

    # Stable integer index for nodes (works for ints/strings alike)
    nodes = list(GN.nodes())
    node_idx = {n:i for i, n in enumerate(nodes)}

    straight, poscurve, negcurve = [], [], []
    for (u, v) in edgelist:
        if GN.has_edge(v, u):
            # Deterministic opposite curvature:
            # if idx(u) < idx(v) -> curve positive, else negative
            if node_idx[u] < node_idx[v]:
                poscurve.append((u, v))
            else:
                negcurve.append((u, v))
        else:
            straight.append((u, v))

    if straight:
        nx.draw_networkx_edges(
            GN, pos, edgelist=straight, edge_color=color, width=width,
            arrows=True, arrowsize=arrowsize, arrowstyle='-|>',
            connectionstyle="arc3,rad=0.0", ax=ax
        )
    if poscurve:
        nx.draw_networkx_edges(
            GN, pos, edgelist=poscurve, edge_color=color, width=width,
            arrows=True, arrowsize=arrowsize, arrowstyle='-|>',
            connectionstyle=f"arc3,rad={rad}", ax=ax
        )
    if negcurve:
        nx.draw_networkx_edges(
            GN, pos, edgelist=negcurve, edge_color=color, width=width,
            arrows=True, arrowsize=arrowsize, arrowstyle='-|>',
            connectionstyle=f"arc3,rad={rad}", ax=ax
        )




def draw_routes(GN, pos, routes_uv, max_routes=15, outfile="route_map"):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_aspect("equal")

    # Base graph (all nodes and edges) in light gray
    #nx.draw_networkx_nodes(GN, pos, node_size=8, node_color="#bbbbbb", ax=ax, linewidths=0.0)
    #nx.draw_networkx_edges(GN, pos, width=0.5, edge_color="#dddddd", arrows=False, ax=ax)

    # Base graph
    # base graph
    



    # Color palette for routes
    # Repeatable but distinct enough for ~15 routes
    random.seed(0)
    palette = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#4c72b0","#dd8452","#55a868","#c44e52","#8172b3",
        "#937860","#da8bc3","#8c8c8c","#b9a778","#4c97a8"
    ]
    # Base graph first
    draw_edges_curved(GN, pos, list(GN.edges()),
        color="#cccccc", width=0.6, rad=0.12, arrowsize=6, ax=ax)

# Per-route overlays; opposite directions will bend apart even across colors
    for k, uv_list in enumerate(routes_uv[:max_routes]):
        draw_edges_curved(GN, pos, uv_list,
                      color=palette[k % len(palette)], width=2.0, rad=0.22, arrowsize=10, ax=ax)


    # Draw up to max_routes; aggregate the rest as one color
    nR = len(routes_uv)
    showK = min(max_routes, nR)
    # Build per-route edge multi-sets (service arcs only)
    """for k in range(showK):
        uv_list = routes_uv[k]
        # Draw as directed segments with arrows. Use a slightly thicker line.
        color = palette[k % len(palette)]
        # Draw edges one by one to preserve direction arrows
        for (u, v) in uv_list:
            nx.draw_networkx_edges(
                GN, pos,
                edgelist=[(u, v)],
                width=2.0,
                edge_color=color,
                arrows=True,
                arrowsize=10,
                arrowstyle='-|>',
                ax=ax
            )

    if nR > showK:
        # Remaining routes in a muted blue
        rest_color = "#7fa2ff"
        for k in range(showK, nR):
            for (u, v) in routes_uv[k]:
                nx.draw_networkx_edges(
                    GN, pos,
                    edgelist=[(u, v)],
                    width=1.5,
                    edge_color=rest_color,
                    arrows=True,
                    arrowsize=8,
                    arrowstyle='-|>',
                    ax=ax
                )"""
    

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{outfile}.png", dpi=300)
    plt.savefig(f"{outfile}.pdf")
    print(f"Saved {outfile}.png and {outfile}.pdf")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="pickle file with (best_cost, routes, perm)")
    ap.add_argument("--edges", required=True, help="edges.csv from your synthetic generator")
    ap.add_argument("--max_routes", type=int, default=15, help="max distinct route colors to draw")
    ap.add_argument("--outfile", default="route_map", help="output filename prefix")
    args = ap.parse_args()

    best_cost, routes, perm = load_memetic(args.pkl)
    df_edges, G_id2uv, GN = build_graph(args.edges)
    pos = compute_layout(GN, seed=1)
    routes_uv = routes_to_uv(routes, df_edges)

    # Basic stats
    total_serviced = sum(len(r) for r in routes)
    print(f"Best cost: {best_cost:.2f}")
    print(f"Routes: {len(routes)}  |  Total serviced arcs: {total_serviced}")
    lens = [len(r) for r in routes]
    print(f"Route length (arcs) min/avg/max: {min(lens)}/{sum(lens)/len(lens):.1f}/{max(lens)}")

    draw_routes(GN, pos, routes_uv, max_routes=args.max_routes, outfile=args.outfile)

if __name__ == "__main__":
    main()
