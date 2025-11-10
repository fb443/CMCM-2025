# gurobi_mip_carp.py
import argparse, csv, pickle
from collections import defaultdict
import networkx as nx
import pandas as pd
from gurobipy import Model, GRB, quicksum
import os


# objective: punish deadheads too
ALPHA_DEAD = 20.0   # try 1.0–2.0
ROUTE_PENALTY = 20.0  # minutes per used route; tune

# ---------- I/O helpers ----------
"""def load_graph(path, fmt, directed=True):
    if fmt == "gpickle":
        G = nx.read_gpickle(path)
    elif fmt == "graphml":
        G = nx.read_graphml(path)
    elif fmt == "edgelist":
        G = nx.read_edgelist(path, create_using=nx.DiGraph() if directed else nx.Graph(), data=(("weight", float),))
    else:
        raise ValueError("Unsupported --graph-format")
    if directed and not G.is_directed():
        G = G.to_directed()
    return G"""

def export_edges_csv(G: nx.DiGraph, csv_path: str, time_attr="travel_time_min", fallback_weight="weight", default_time=1.0):
    best = {}
    for u, v, data in G.edges(data=True):
        t = data.get(time_attr, data.get(fallback_weight, default_time))
        t = float(t)
        key = (int(u), int(v))
        if key not in best or t < best[key]:
            best[key] = t
    rows = sorted(best.items())  # [((u,v), t)]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "u", "v", "travel_time_min"])
        for eid, ((u, v), t) in enumerate(rows):
            w.writerow([eid, u, v, t])
    eid_of = {uv: eid for eid, (uv, _) in enumerate(rows)}
    time_of = {uv: best[uv] for uv in best}
    return eid_of, time_of

def load_edges_csv(path):
    E = pd.read_csv(path)
    E["u"] = E["u"].astype(int)
    E["v"] = E["v"].astype(int)
    E["edge_id"] = E["edge_id"].astype(int)
    E["travel_time_min"] = E["travel_time_min"].astype(float)
    eid_of = {(int(r.u), int(r.v)): int(r.edge_id) for _, r in E.iterrows()}
    time_of = {(int(r.u), int(r.v)): float(r.travel_time_min) for _, r in E.iterrows()}
    V = set(E["u"]).union(set(E["v"]))
    A = list(eid_of.keys())
    return E, V, A, eid_of, time_of


def _iter_sol(trivar):
    # yields (i,j,k) with value > 0.5 from a Gurobi 3-index Var dict
    for i,j,k in trivar.keys():
        if trivar[i,j,k].X > 0.5:
            yield (i,j,k)

def summarize_solution(V, A_req, R, x, z, u, r=None, label=""):
    """
    V: iterable of nodes
    A_req: list/set of required arcs (u,v)
    R: list/set of deadhead-allowed arcs (u,v)
    x,z,u: Gurobi variables as built in your model
    r: optional root selector r[k,v] (for depot-free connectivity models)
    """
    V = list(V)
    K = len(u)
    svc_by_k = defaultdict(list)
    dd_by_k  = defaultdict(list)

    # collect chosen arcs
    for i,j,k in _iter_sol(x):
        svc_by_k[k].append((i,j))
    for i,j,k in _iter_sol(z):
        dd_by_k[k].append((i,j))

    print(f"=== Connectivity report {label} ===")
    for k in range(K):
        if u[k].X < 0.5:
            continue

        svc = svc_by_k[k]; dd = dd_by_k[k]
        Hs  = nx.DiGraph(); Hs.add_nodes_from(V); Hs.add_edges_from(svc)     # service-only
        Ha  = nx.DiGraph(); Ha.add_nodes_from(V); Ha.add_edges_from(svc+dd)  # service + deadhead

        # components
        comps_s = list(nx.weakly_connected_components(Hs.subgraph(set(sum(([u,v] for u,v in svc), [])))))
        comps_a = list(nx.weakly_connected_components(Ha.subgraph(set(sum(([u,v] for u,v in svc+dd), [])))))

        # degree imbalances on used edges
        def deg_imb(G):
            outd = G.out_degree(); ind = G.in_degree()
            imbal = {v: outd[v] - ind[v] for v in G.nodes() if G.degree(v) > 0}
            s_plus  = [v for v,d in imbal.items() if d== 1]
            s_minus = [v for v,d in imbal.items() if d==-1]
            bad     = {v:d for v,d in imbal.items() if d not in (-1,0,1)}
            return s_plus, s_minus, bad

        splus_s, sminus_s, bad_s = deg_imb(Hs)
        splus_a, sminus_a, bad_a = deg_imb(Ha)

        print(f"\nRoute k={k}: |svc|={len(svc)} |dd|={len(dd)}")
        print(f"  svc components: {len(comps_s)} ; used nodes: {sum(len(c) for c in comps_s) or 0}")
        print(f"  all components: {len(comps_a)} ; used nodes: {sum(len(c) for c in comps_a) or 0}")
        print(f"  svc start/end candidates (+1/-1): {len(splus_s)}/{len(sminus_s)} ; bad deg nodes: {len(bad_s)}")
        print(f"  all start/end candidates (+1/-1): {len(splus_a)}/{len(sminus_a)} ; bad deg nodes: {len(bad_a)}")

        # depot-free: root reachability on selected edges
        if r is not None:
            root = next((v for v in V if r[k,v].X > 0.5), None)
            if root is not None and Ha.number_of_edges() > 0:
                reach = {root} | nx.descendants(Ha, root)
                touched_nodes = set(sum(([u,v] for u,v in svc), []))
                unreachable = touched_nodes - reach
                print(f"  root={root} ; touched={len(touched_nodes)} ; unreachable_from_root={len(unreachable)}")
                if unreachable:
                    print(f"    sample unreachable: {list(unreachable)[:10]}")

        # longest deadheads
        if dd:
            print("  sample dd arcs (up to 10):", dd[:10])

def check_connectivity_hard(V, x, z, u, require_connected=True):
    """
    Returns list of offending routes that are not connected in Ha (service+deadhead).
    """
    V = list(V); K = len(u)
    offenders = []
    for k in range(K):
        if u[k].X < 0.5: continue
        svc = [(i,j) for (i,j,kk) in x.keys() if kk==k and x[i,j,k].X>0.5]
        dd  = [(i,j) for (i,j,kk) in z.keys() if kk==k and z[i,j,k].X>0.5]
        Ha  = nx.DiGraph(); Ha.add_nodes_from(V); Ha.add_edges_from(svc+dd)
        used_nodes = set(sum(([u,v] for u,v in svc+dd), []))
        comps = list(nx.weakly_connected_components(Ha.subgraph(used_nodes)))
        if require_connected and len(comps) > 1:
            offenders.append((k, len(comps)))
    return offenders

def deadhead_counts(z, u, t_dead=None):
    """
    z: Gurobi var dict z[i,j,k] (binary)
    u: Gurobi var dict u[k]     (binary)
    t_dead: optional dict {(i,j): minutes} to also sum time
    """
    K = len(u)
    per_route = []
    total_edges = 0
    total_time  = 0.0
    for k in range(K):
        if u[k].X < 0.5:
            per_route.append((k, 0, 0.0)); continue
        cnt = 0; tm = 0.0
        for (i,j,kk) in z.keys():
            if kk==k and z[i,j,k].X > 0.5:
                cnt += 1
                if t_dead is not None:
                    tm += float(t_dead[i,j])
        per_route.append((k, cnt, tm))
        total_edges += cnt
        total_time  += tm
    print("Deadheads per route (k, count, time_min):", per_route)
    print(f"Total deadhead edges: {total_edges}  total deadhead minutes: {total_time:.1f}")
    return per_route, total_edges, total_time

"""def solve_carp_connected_nondepot_eid(
    V, E_req, E_all, head, tail, t_svc, t_dead,
    K_routes, F_vals,
    alpha_dead=1.0, route_penalty=20.0,
    cover_equals_one=False,
    time_limit=None, mipfocus=1, threads=0
):
    # adjacency over eids
    out_req, in_req = defaultdict(list), defaultdict(list)
    for e in E_req:
        u, v = head[e], tail[e]
        out_req[u].append(e); in_req[v].append(e)

    out_all, in_all = defaultdict(list), defaultdict(list)
    for e in E_all:
        u, v = head[e], tail[e]
        out_all[u].append(e); in_all[v].append(e)

    # small constants
    deg_req = {v: len(out_req[v]) + len(in_req[v]) for v in V}
    M_deg = max(1, max(deg_req.values(), default=1))
    M_cap = max(1, len(V))

    m = Model("CARP_connected_linear_eid")

    # variables (eid-indexed)
    x = m.addVars(((e,k) for e in E_req for k in range(K_routes)), vtype=GRB.BINARY, name="x")  # service
    z = m.addVars(((e,k) for e in E_all for k in range(K_routes)), vtype=GRB.BINARY, name="z")  # deadhead
    u = m.addVars(range(K_routes), vtype=GRB.BINARY, name="u")                                   # route used

    y = m.addVars(((k,F) for k in range(K_routes) for F in F_vals), vtype=GRB.BINARY, name="y") # cap choice
    f = m.addVars(range(K_routes), vtype=GRB.CONTINUOUS, lb=0.0, name="f")                      # cap value

    s = m.addVars(((k,v) for k in range(K_routes) for v in V), vtype=GRB.BINARY, name="start")  # ≤1
    evar = m.addVars(((k,v) for k in range(K_routes) for v in V), vtype=GRB.BINARY, name="end") # ≤1

    a = m.addVars(((k,v) for k in range(K_routes) for v in V), vtype=GRB.BINARY, name="a_touch")
    r = m.addVars(((k,v) for k in range(K_routes) for v in V), vtype=GRB.BINARY, name="root")
    rho = m.addVars(((k,v) for k in range(K_routes) for v in V), vtype=GRB.CONTINUOUS, lb=0.0, name="rho")
    g = m.addVars(((e,k) for e in E_all for k in range(K_routes)), vtype=GRB.CONTINUOUS, lb=0.0, name="g")
    b = m.addVars(range(K_routes), vtype=GRB.CONTINUOUS, lb=0.0, name="b_touched")

    # activation
    m.addConstrs((x[e,k] <= u[k] for e in E_req for k in range(K_routes)), name="x_le_u")
    m.addConstrs((z[e,k] <= u[k] for e in E_all for k in range(K_routes)), name="z_le_u")

    # start/end limits
    m.addConstrs((quicksum(s[k,v] for v in V) <= u[k] for k in range(K_routes)), name="start_le_u")
    m.addConstrs((quicksum(evar[k,v] for v in V) <= u[k] for k in range(K_routes)), name="end_le_u")

    # cap choice
    m.addConstrs((quicksum(y[k,F] for F in F_vals) == u[k] for k in range(K_routes)), name="choose_cap_if_used")
    m.addConstrs((f[k] == quicksum(F * y[k,F] for F in F_vals) for k in range(K_routes)), name="f_equals_choice")
    Fmax = max(F_vals)
    m.addConstrs((f[k] <= Fmax * u[k] for k in range(K_routes)), name="f_zero_if_unused")

    # coverage
    if cover_equals_one:
        m.addConstrs((quicksum(x[e,k] for k in range(K_routes)) == 1 for e in E_req), name="coverage_eq1")
    else:
        m.addConstrs((quicksum(x[e,k] for k in range(K_routes)) >= 1 for e in E_req), name="coverage_ge1")

    # open-route balance (over eids)
    for k in range(K_routes):
        for v in V:
            m.addConstr(
                quicksum(x[e,k] for e in out_req[v]) + quicksum(z[e,k] for e in out_all[v]) -
                quicksum(x[e,k] for e in  in_req[v]) - quicksum(z[e,k] for e in  in_all[v])
                == s[k,v] - evar[k,v],
                name=f"flow_open[{k},{v}]"
            )

    # touch from serviced eids
    for k in range(K_routes):
        for v in V:
            m.addConstr(
                quicksum(x[e,k] for e in out_req[v]) + quicksum(x[e,k] for e in in_req[v])
                <= M_deg * a[k,v],
                name=f"touch_imp[{k},{v}]"
            )
            m.addConstr(a[k,v] <= u[k], name=f"a_le_u[{k},{v}]")
        m.addConstr(b[k] == quicksum(a[k,v] for v in V), name=f"b_def[{k}]")
        m.addConstr(quicksum(a[k,v] for v in V) >= u[k], name=f"a_ge_u[{k}]")

    # choose one root among touched nodes
    for k in range(K_routes):
        m.addConstr(quicksum(r[k,v] for v in V) == u[k], name=f"one_root[{k}]")
        m.addConstrs((r[k,v] <= a[k,v] for v in V), name=f"root_le_a[{k}]")

    # connectivity capacity: g only where selected
    for k in range(K_routes):
        for e in E_all:
            if e in set(E_req):
                m.addConstr(g[e,k] <= M_cap * ( (x[e,k] if e in E_req else 0) + z[e,k] ), name=f"g_cap_req[{e},{k}]")
            else:
                m.addConstr(g[e,k] <= M_cap * z[e,k], name=f"g_cap_dead[{e},{k}]")

    # allocate total b[k] to the unique root (linear)
    for k in range(K_routes):
        m.addConstr(quicksum(rho[k,v] for v in V) == b[k], name=f"rho_total[{k}]")
        m.addConstrs((rho[k,v] <= M_cap * r[k,v] for v in V), name=f"rho_at_root[{k}]")

    # connectivity balances (OUT - IN = rho - a)
    for k in range(K_routes):
        for v in V:
            m.addConstr(
                quicksum(g[e,k] for e in out_all[v]) -
                quicksum(g[e,k] for e in  in_all[v])
                == rho[k,v] - a[k,v],
                name=f"g_bal[{k},{v}]"
            )

    # time cap per route
    for k in range(K_routes):
        m.addConstr(
            quicksum(t_svc[e]  * x[e,k] for e in E_req) +
            quicksum(t_dead[e] * z[e,k] for e in E_all)
            <= f[k],
            name=f"time_cap[{k}]"
        )

    # objective
    m.setObjective(
        quicksum(t_svc[e]  * x[e,k] for e in E_req for k in range(K_routes)) +
        alpha_dead * quicksum(t_dead[e] * z[e,k] for e in E_all for k in range(K_routes)) +
        route_penalty * quicksum(u[k] for k in range(K_routes)),
        GRB.MINIMIZE
    )

    if time_limit is not None: m.Params.TimeLimit = float(time_limit)
    if threads: m.Params.Threads = int(threads)
    m.Params.MIPFocus = int(mipfocus); m.Params.OutputFlag = 1

    m.optimize()
    return m, x, z, f, u, a, r, g"""

from gurobipy import Model, GRB, quicksum
from collections import defaultdict

def solve_carp_connected_nondepot_eid(
    V, E_req, E_all, head, tail, t_svc, t_dead,
    K_routes, F_vals,
    alpha_dead=1.0, route_penalty=20.0,
    cover_equals_one=False,
    time_limit=None, mipfocus=1, threads=0,
    # optional solver boosts
    heuristics=0.5, rins=20, pump_passes=20, cuts=2, presolve=-1, nodefile_start=0.0,
):
    """
    Improvements vs. your version:
      - Feasibility-first parameters (Heuristics/RINS/FPump/Cuts/Presolve).
      - Symmetry breaking: u[k] >= u[k+1], f[k] >= f[k+1]-eps.
      - Tighter connectivity capacities (g[e,k] <= b[k] and g[e,k] <= M*(x+z)).
      - Linear arc-count bound: sum_e x[e,k] <= f[k]/t_min.
    """
    # adjacency over eids
    out_req, in_req = defaultdict(list), defaultdict(list)
    for e in E_req:
        u_, v_ = head[e], tail[e]
        out_req[u_].append(e); in_req[v_].append(e)

    out_all, in_all = defaultdict(list), defaultdict(list)
    for e in E_all:
        u_, v_ = head[e], tail[e]
        out_all[u_].append(e); in_all[v_].append(e)

    # small constants
    deg_req = {v: len(out_req[v]) + len(in_req[v]) for v in V}
    M_deg = max(1, max(deg_req.values(), default=1))
    M_cap = max(1, len(V))  # loose global cap, tightened below by b[k]
    t_min = min(t_svc[e] for e in E_req) if E_req else 1.0
    eps = 1e-6

    m = Model("CARP_connected_linear_eid")

    # variables (eid-indexed)
    x = m.addVars(((e,k) for e in E_req for k in range(K_routes)), vtype=GRB.BINARY, name="x")  # service
    z = m.addVars(((e,k) for e in E_all for k in range(K_routes)), vtype=GRB.BINARY, name="z")  # deadhead
    u = m.addVars(range(K_routes), vtype=GRB.BINARY, name="u")                                   # route used

    y = m.addVars(((k,F) for k in range(K_routes) for F in F_vals), vtype=GRB.BINARY, name="y") # cap choice
    f = m.addVars(range(K_routes), vtype=GRB.CONTINUOUS, lb=0.0, name="f")                      # cap value

    s = m.addVars(((k,v) for k in range(K_routes) for v in V), vtype=GRB.BINARY, name="start")  # ≤1
    evar = m.addVars(((k,v) for k in range(K_routes) for v in V), vtype=GRB.BINARY, name="end") # ≤1

    a = m.addVars(((k,v) for k in range(K_routes) for v in V), vtype=GRB.BINARY, name="a_touch")
    r = m.addVars(((k,v) for k in range(K_routes) for v in V), vtype=GRB.BINARY, name="root")
    rho = m.addVars(((k,v) for k in range(K_routes) for v in V), vtype=GRB.CONTINUOUS, lb=0.0, name="rho")
    g = m.addVars(((e,k) for e in E_all for k in range(K_routes)), vtype=GRB.CONTINUOUS, lb=0.0, name="g")
    b = m.addVars(range(K_routes), vtype=GRB.CONTINUOUS, lb=0.0, name="b_touched")

    # activation
    m.addConstrs((x[e,k] <= u[k] for e in E_req for k in range(K_routes)), name="x_le_u")
    m.addConstrs((z[e,k] <= u[k] for e in E_all for k in range(K_routes)), name="z_le_u")

    # start/end limits
    m.addConstrs((quicksum(s[k,v] for v in V) <= u[k] for k in range(K_routes)), name="start_le_u")
    m.addConstrs((quicksum(evar[k,v] for v in V) <= u[k] for k in range(K_routes)), name="end_le_u")

    # cap choice
    m.addConstrs((quicksum(y[k,F] for F in F_vals) == u[k] for k in range(K_routes)), name="choose_cap_if_used")
    m.addConstrs((f[k] == quicksum(F * y[k,F] for F in F_vals) for k in range(K_routes)), name="f_equals_choice")
    Fmax = max(F_vals)
    m.addConstrs((f[k] <= Fmax * u[k] for k in range(K_routes)), name="f_zero_if_unused")

    # coverage
    if cover_equals_one:
        m.addConstrs((quicksum(x[e,k] for k in range(K_routes)) == 1 for e in E_req), name="coverage_eq1")
    else:
        m.addConstrs((quicksum(x[e,k] for k in range(K_routes)) >= 1 for e in E_req), name="coverage_ge1")

    # open-route balance (over eids)
    for k in range(K_routes):
        for v in V:
            m.addConstr(
                quicksum(x[e,k] for e in out_req[v]) + quicksum(z[e,k] for e in out_all[v]) -
                quicksum(x[e,k] for e in  in_req[v]) - quicksum(z[e,k] for e in  in_all[v])
                == s[k,v] - evar[k,v],
                name=f"flow_open[{k},{v}]"
            )

    # touch from serviced eids
    for k in range(K_routes):
        for v in V:
            m.addConstr(
                quicksum(x[e,k] for e in out_req[v]) + quicksum(x[e,k] for e in in_req[v])
                <= M_deg * a[k,v],
                name=f"touch_imp[{k},{v}]"
            )
            m.addConstr(a[k,v] <= u[k], name=f"a_le_u[{k},{v}]")
        m.addConstr(b[k] == quicksum(a[k,v] for v in V), name=f"b_def[{k}]")
        m.addConstr(quicksum(a[k,v] for v in V) >= u[k], name=f"a_ge_u[{k}]")

    # choose one root among touched nodes
    for k in range(K_routes):
        m.addConstr(quicksum(r[k,v] for v in V) == u[k], name=f"one_root[{k}]")
        m.addConstrs((r[k,v] <= a[k,v] for v in V), name=f"root_le_a[{k}]")

    # connectivity capacity (linear, no bilinear): both caps hold
    for k in range(K_routes):
        for e in E_all:
            # active-edge cap
            if e in set(E_req):
                m.addConstr(g[e,k] <= M_cap * (x[e,k] + z[e,k]), name=f"g_cap_active[{e},{k}]")
            else:
                m.addConstr(g[e,k] <= M_cap * z[e,k], name=f"g_cap_dead[{e},{k}]")
            # global per-route tightening by b[k]
            m.addConstr(g[e,k] <= b[k], name=f"g_cap_b[{e},{k}]")

    # allocate total b[k] to the unique root (linear)
    for k in range(K_routes):
        m.addConstr(quicksum(rho[k,v] for v in V) == b[k], name=f"rho_total[{k}]")
        m.addConstrs((rho[k,v] <= M_cap * r[k,v] for v in V), name=f"rho_at_root[{k}]")

    # connectivity balances (OUT - IN = rho - a)
    for k in range(K_routes):
        for v in V:
            m.addConstr(
                quicksum(g[e,k] for e in out_all[v]) -
                quicksum(g[e,k] for e in  in_all[v])
                == rho[k,v] - a[k,v],
                name=f"g_bal[{k},{v}]"
            )

    # time cap per route
    for k in range(K_routes):
        m.addConstr(
            quicksum(t_svc[e]  * x[e,k] for e in E_req) +
            quicksum(t_dead[e] * z[e,k] for e in E_all)
            <= f[k],
            name=f"time_cap[{k}]"
        )
        # linear arc-count bound
        m.addConstr(quicksum(x[e,k] for e in E_req) <= f[k] / t_min + eps, name=f"arc_count[{k}]")

    # symmetry breaking
    for k in range(K_routes - 1):
        m.addConstr(u[k] >= u[k+1], name=f"sym_u[{k}]")
        m.addConstr(f[k] >= f[k+1] - eps, name=f"sym_f[{k}]")

    # objective
    m.setObjective(
        quicksum(t_svc[e]  * x[e,k] for e in E_req for k in range(K_routes)) +
        alpha_dead * quicksum(t_dead[e] * z[e,k] for e in E_all for k in range(K_routes)) +
        route_penalty * quicksum(u[k] for k in range(K_routes)),
        GRB.MINIMIZE
    )

    # solver params: feasibility-first and more heuristics
    if time_limit is not None: m.Params.TimeLimit = float(time_limit)
    if threads: m.Params.Threads = int(threads)
    m.Params.MIPFocus   = int(mipfocus)  # keep your knob
    m.Params.Heuristics = float(heuristics)
    m.Params.RINS       = int(rins)
    m.Params.PumpPasses = int(pump_passes)
    m.Params.Cuts       = int(cuts)
    m.Params.Presolve   = int(presolve)
    if nodefile_start > 0.0:
        m.Params.NodefileStart = float(nodefile_start)
    m.Params.OutputFlag = 1

    m.optimize()
    return m, x, z, f, u, a, r, g


# ---------- Convert solution to memetic-like pickle ----------
def dump_memetic_like_pickle(path, model, x, fvar, A, eid_of):
    if model.SolCount == 0:
        raise RuntimeError("No solution")
    K_routes = len(fvar)
    routes_eids, used_all = [], set()
    for k in range(K_routes):
        arcs_k = []
        for (i, j) in A:
            if x[i, j, k].X > 0.5:
                eid = eid_of[(i, j)]
                arcs_k.append(eid)
                used_all.add(eid)
        if arcs_k:
            routes_eids.append(arcs_k)
    perm = sorted(list(used_all))  # unordered is fine for your current visualizer
    best_cost = float(model.ObjVal)
    with open(path, "wb") as f:
        pickle.dump((best_cost, routes_eids, perm), f)



def main():
    import argparse, csv, os
    import networkx as nx
    from collections import defaultdict
    # argparse as you wrote...
    ap = argparse.ArgumentParser()
    # Input
    ap.add_argument("--edges", default=None)
    ap.add_argument("--graph", default=None)
    ap.add_argument("--graph-format", choices=["gpickle","graphml","edgelist"], default="gpickle")
    ap.add_argument("--time-attr", default="travel_time_sec")
    ap.add_argument("--deadhead-attr", default="deadhead_time_sec")
    ap.add_argument("--cycle-attr", default="cycle_time_hr")
    ap.add_argument("--fallback-weight", default="weight")
    ap.add_argument("--default-time", type=float, default=60.0)
    ap.add_argument("--write-edges", default=None)

    # MIP controls
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--F", default="60,90,120,180")
    ap.add_argument("--alpha-dead", type=float, default=1.0)
    ap.add_argument("--route-penalty", type=float, default=20.0)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--mipfocus", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--require-max-cycle-min", type=float, default=180.0)
    ap.add_argument("--autoname", action="store_true")

    # Outputs
    ap.add_argument("--out-pkl", default="memetic_results_from_mip_ithaca.pkl")
    args = ap.parse_args()

    # (1) parse F
    F_vals = [float(x) for x in args.F.split(",") if x.strip()]
    Fmax = max(F_vals)

    # helpers
    def load_graph(path, fmt):
        import pickle
        import networkx as nx
        # normalize fmt by extension if needed
        if fmt is None:
            ext = (os.path.splitext(path)[1] or "").lstrip(".").lower()
            fmt = "gpickle" if ext in {"gpickle", "pkl", "pickle"} else ext

        if fmt == "gpickle":
            # Try NetworkX API if present (NX <3.0), else use pickle
            try:
                # NX 2.x
                from networkx.readwrite.gpickle import read_gpickle  # type: ignore
                return read_gpickle(path)
            except Exception:
                with open(path, "rb") as fh:
                    G = pickle.load(fh)
                # sanity
                if not hasattr(G, "nodes") or not hasattr(G, "edges"):
                    raise ValueError(f"{path} did not unpickle to a NetworkX graph")
                return G

        if fmt == "graphml":
            return nx.read_graphml(path)

        if fmt == "edgelist":
            # Minimal edgelist fallback; no attributes
            return nx.read_edgelist(path, create_using=nx.MultiDiGraph(), nodetype=int)

        raise ValueError(f"Unsupported graph format: {fmt}")


    def export_edges_csv_from_graph(out_csv, head, tail, t_svc, t_dead, req_flag):
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["eid","u","v","travel_time_min","deadhead_time_min","required"])
            for e in sorted(head.keys()):
                w.writerow([e, head[e], tail[e],
                            f"{t_svc[e]:.6f}", f"{t_dead[e]:.6f}", int(req_flag[e])])

    # --- decide output filenames ONCE (works for both branches) ---
    thr = args.require_max_cycle_min
    tag = f"c{int(round(thr))}m"

    if args.autoname:
        base_csv = (args.write_edges or "edges").rsplit(".", 1)
        edges_csv_path = f"{base_csv[0]}_{tag}.{base_csv[1] if len(base_csv)==2 else 'csv'}"
        base_pkl = (args.out_pkl or "memetic_results_from_mip.pkl").rsplit(".", 1)
        out_pkl_path = f"{base_pkl[0]}_{tag}.{base_pkl[1] if len(base_pkl)==2 else 'pkl'}"
    else:
        edges_csv_path = args.write_edges or "edges.csv"
        out_pkl_path = args.out_pkl

    # (2–4) build EIDs + convert units + mark required
    G = None
    if args.graph:
        G = load_graph(args.graph, args.graph_format)
        if not isinstance(G, nx.MultiDiGraph):
            MG = nx.MultiDiGraph()
            MG.add_nodes_from(G.nodes(data=True))
            for u, v, data in G.edges(data=True):
                MG.add_edge(u, v, **(data or {}))
            G = MG

        V = list(G.nodes())
        E_all, E_req = [], []
        head, tail, t_svc, t_dead = {}, {}, {}, {}
        req_flag = {}
        eid = 0

        for u, v, k, d in G.edges(keys=True, data=True):
            # times: seconds → minutes; use underscores (not hyphens)
            tt_sec = d.get(args.time_attr)
            dd_sec = d.get(args.deadhead_attr)
            if tt_sec is None:
                tt_sec = d.get(args.fallback_weight, args.default_time)
            if dd_sec is None:
                dd_sec = tt_sec

            tt_min = float(tt_sec) / 60.0
            dd_min = float(dd_sec) / 60.0

            cyc_hr = d.get(args.cycle_attr, None)
            cyc_min = float(cyc_hr) * 60.0 if cyc_hr is not None else None

            head[eid], tail[eid] = u, v
            t_svc[eid]  = max(1e-3, tt_min)
            t_dead[eid] = max(1e-3, dd_min)

            E_all.append(eid)
            is_required = (cyc_min is not None) and (cyc_min <= thr)
            req_flag[eid] = is_required
            if is_required:
                E_req.append(eid)
            eid += 1

        if args.write_edges:
            export_edges_csv_from_graph(edges_csv_path, head, tail, t_svc, t_dead, req_flag)

    elif args.edges:
        # CSV branch (expects columns: eid,u,v,travel_time_min,deadhead_time_min,required)
        Vset = set()
        head, tail, t_svc, t_dead, req_flag = {}, {}, {}, {}, {}
        E_all, E_req = [], []
        with open(args.edges, newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                e = int(r["eid"])
                u = int(r["u"]); v = int(r["v"])
                tt = float(r.get("travel_time_min", 1.0))
                dd = float(r.get("deadhead_time_min", tt))
                req = int(r.get("required", 1))
                head[e], tail[e] = u, v
                t_svc[e], t_dead[e] = tt, dd
                req_flag[e] = bool(req)
                E_all.append(e)
                if req: E_req.append(e)
                Vset.add(u); Vset.add(v)
        V = list(Vset)
        # Do not write CSV here; we are consuming one
    else:
        raise SystemExit("Provide --graph <file> (preferred) or --edges <edges.csv>.")

    # Capacity sanity
    total_service = sum(t_svc[e] for e in E_req)
    if total_service > args.K * Fmax + 1e-9:
        print(f"[warn] service {total_service:.1f} > K*Fmax {args.K*Fmax:.1f}")

    # Solve
    m, x, z, f, u, a, r, g = solve_carp_connected_nondepot_eid(
        V=V, E_req=E_req, E_all=E_all, head=head, tail=tail,
        t_svc=t_svc, t_dead=t_dead,
        K_routes=args.K, F_vals=F_vals,
        alpha_dead=args.alpha_dead, route_penalty=args.route_penalty,
        time_limit=args.time_limit, mipfocus=args.mipfocus, threads=args.threads,
    )

    status = m.Status
    print(f"[status] code={status} solcount={m.SolCount} bestbound={getattr(m, 'ObjBound', None)} obj={getattr(m, 'objVal', None)}")

    if m.SolCount == 0:
        # No incumbent. Don’t touch Var.X.
        # Optional: write the model and a snapshot for debugging.
        try: m.write("last_model.lp")
        except Exception: pass
        # You can also write the relaxation if useful:
        # m.Params.Method = 1  # dual simplex for LP solve
        # m.relax().write("last_relax.lp")
        raise SystemExit(
            "No feasible incumbent found within the time limit. Increase --time-limit "
            "or provide a warm start. Best bound shown above."
        )

    # Otherwise safe to read X
    routes_eid = {k: [e for e in E_req if x[e, k].X > 0.5] for k in range(args.K) if u[k].X > 0.5}
    routes_uv  = {k: [(head[e], tail[e]) for e in es] for k, es in routes_eid.items()}
    f_by_route = {k: f[k].X for k in range(args.K) if u[k].X > 0.5}

    out = {"routes_eid": routes_eid, "routes_uv": routes_uv,
        "f_by_route": f_by_route, "best_cost": m.objVal}

    os.makedirs(os.path.dirname(out_pkl_path) or ".", exist_ok=True)
    with open(out_pkl_path, "wb") as fh:
        pickle.dump(out, fh)

    written_csv = edges_csv_path if args.write_edges else "(no CSV written)"
    print(f"Wrote {written_csv} and {out_pkl_path}")




if __name__ == "__main__":
    main()
