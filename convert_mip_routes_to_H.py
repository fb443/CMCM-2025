#!/usr/bin/env python3
# convert_mip_routes_to_H.py
import argparse, pickle, math
import networkx as nx

def load_gpickle(path):
    # NetworkX 2.x/3.x compatible loader
    try:
        from networkx.readwrite.gpickle import read_gpickle
        return read_gpickle(path)
    except Exception:
        with open(path, "rb") as fh:
            G = pickle.load(fh)
        if not hasattr(G, "nodes") or not hasattr(G, "edges"):
            raise ValueError(f"{path} did not unpickle to a NetworkX graph")
        return G

def rebuild_eid_map(H, time_attr_s="travel_time_sec", dead_attr_s="deadhead_time_sec"):
    """
    Recreate the same EID ordering used by your MIP code:
      for u,v,k,data in H.edges(keys=True,data=True): assign eid++ in iteration order.
    Returns:
      eid2uvk: {eid: (u,v,k)}
      svc_min: {eid: travel_time_min}   # seconds->minutes with >=1e-3 floor
      dead_min:{eid: deadhead_time_min}
    """
    eid2uvk, svc_min, dead_min = {}, {}, {}
    eid = 0
    for u, v, k, d in H.edges(keys=True, data=True):
        tt_sec = d.get(time_attr_s, d.get("weight", 60.0))
        dd_sec = d.get(dead_attr_s, tt_sec)
        tt_min = max(1e-3, float(tt_sec) / 60.0)
        dd_min = max(1e-3, float(dd_sec) / 60.0)
        eid2uvk[eid] = (u, v, k)
        svc_min[eid] = tt_min
        dead_min[eid] = dd_min
        eid += 1
    return eid2uvk, svc_min, dead_min

def ordered_walk_from_arcs(uv_list):
    """Greedy Eulerian stitch; returns ordered (u,v) list, start, end."""
    if not uv_list:
        return [], None, None
    G = nx.DiGraph()
    for u, v in uv_list:
        G.add_edge(u, v, used=(G[u][v]["used"] + 1 if G.has_edge(u, v) else 1))
    outd, ind = G.out_degree(), G.in_degree()
    start = next((n for n in G if outd[n] - ind[n] == 1), None)
    if start is None:
        start = next(iter(G.nodes()))
    succ = {u: list(G.successors(u)) for u in G.nodes()}
    stack, trail = [start], []
    while stack:
        u = stack[-1]
        if succ[u]:
            v = succ[u].pop()
            stack.append(v)
        else:
            trail.append(stack.pop())
    trail.reverse()
    ordered = [(trail[i], trail[i+1]) for i in range(len(trail)-1)]
    s = ordered[0][0] if ordered else start
    e = ordered[-1][1] if ordered else start
    return ordered, s, e

def convert(pkl_path, H_path, out_path=None):
    # 1) load artifacts
    H = load_gpickle(H_path)
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    # 2) extract routes from pickle
    routes_eid = obj.get("routes_eid", {})
    routes_uv  = obj.get("routes_uv", {})
    f_by_route = obj.get("f_by_route", {})

    # 3) rebuild eid mapping on H to recover (u,v,key)
    eid2uvk, svc_min, _ = rebuild_eid_map(H)

    # 4) produce “H-native” outputs
    out = {
        "best_cost": float(obj.get("best_cost", float("nan"))),
        "f_by_route": {int(k): float(v) for k, v in f_by_route.items()},
        "routes_eid": {int(k): [int(e) for e in es] for k, es in routes_eid.items()},
        "routes_uv_from_eid": {},
        "routes_uv_as_given": {int(k): [(int(u), int(v)) for (u, v) in uv]
                               for k, uv in routes_uv.items()},
        "routes_uvk_on_H": {},        # uses eid->(u,v,key) when routes_eid present
        "routes_ordered_uv": {},      # stitched sequence for visualization
        "route_service_minutes": {},  # sum of per-arc service mins from H
    }

    # 4a) map eid -> (u,v,key) and accumulate per-route service times
    for k, es in routes_eid.items():
        uvk_list = []
        uv_list  = []
        svc_sum  = 0.0
        for e in es:
            u, v, key = eid2uvk[e]
            uvk_list.append((u, v, key))
            uv_list.append((u, v))
            svc_sum += svc_min[e]
        out["routes_uvk_on_H"][int(k)] = uvk_list
        out["routes_uv_from_eid"][int(k)] = uv_list
        out["route_service_minutes"][int(k)] = svc_sum

    # 4b) create ordered trail per route for plotting or GIS export
    base = out["routes_uv_from_eid"] if out["routes_uv_from_eid"] else out["routes_uv_as_given"]
    for k, uv in base.items():
        ordered, s, e = ordered_walk_from_arcs(uv)
        out["routes_ordered_uv"][int(k)] = {
            "ordered_uv": ordered,
            "start": s,
            "end": e
        }

    # 5) optionally annotate a copy of H with route ids for quick inspection
    H_annot = H.copy()
    for k, uvk in out["routes_uvk_on_H"].items():
        for (u, v, key) in uvk:
            # push a list of routes serving this multiedge
            rlist = H_annot[u][v][key].get("routes", [])
            if k not in rlist:
                rlist = list(rlist) + [k]
            H_annot[u][v][key]["routes"] = rlist

    # 6) write outputs
    if out_path:
        with open(out_path, "wb") as f:
            pickle.dump(out, f)
        nx.write_gpickle(H_annot, out_path.replace(".pkl", "_annotated_H.gpickle"))

    return out, H_annot

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="memetic_results_from_mip_ithaca_c150m.pkl")
    ap.add_argument("--H", required=True, help="H.gpickle (original MultiDiGraph)")
    ap.add_argument("--out", default="routes_on_H.pkl", help="output pickle with uv/uvk mappings")
    args = ap.parse_args()
    convert(args.pkl, args.H, args.out)
    print(f"Wrote {args.out} and {args.out.replace('.pkl','_annotated_H.gpickle')}")
