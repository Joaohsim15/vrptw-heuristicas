"""
baseline.py
Heurística clássica de Clarke & Wright (savings), adaptada para VRPTW:
- parte de uma rota por cliente (0 -> c -> 0)
- calcula a "economia" de unir duas rotas pelas extremidades
- tenta as fusões em ordem decrescente de economia, só aceitando a fusão
  se ela continuar respeitando capacidade E as janelas de tempo de TODOS
  os clientes da rota resultante.

Usada como baseline (R4) e também como ponto de partida candidato para o
diferencial de warm-start (D2) no checkpoint 3.
"""
from __future__ import annotations
from pathlib import Path
from instance import VRPTWInstance, route_distance


def _route_is_time_feasible(route: list[int], inst: VRPTWInstance) -> bool:
    time = 0.0
    prev = 0
    for cust in route:
        arrival = time + inst.dist[prev, cust]
        start = max(arrival, inst.ready[cust])
        if start > inst.due[cust] + 1e-6:
            return False
        time = start + inst.service[cust]
        prev = cust
    return True


def _route_load(route: list[int], inst: VRPTWInstance) -> float:
    return sum(inst.demand[c] for c in route)


def clarke_wright(inst: VRPTWInstance) -> list[list[int]]:
    customers = list(range(1, inst.n_customers + 1))
    routes = {c: [c] for c in customers}          # rota de cada cliente (mutável)
    route_of = {c: c for c in customers}           # cliente -> id da rota a que pertence

    savings = []
    for i in customers:
        for j in customers:
            if i >= j:
                continue
            s = inst.dist[0, i] + inst.dist[0, j] - inst.dist[i, j]
            savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])

    for s, i, j in savings:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        route_i, route_j = routes[ri], routes[rj]
        # só pode fundir se i for o ÚLTIMO da rota_i e j for o PRIMEIRO da rota_j
        if route_i[-1] != i or route_j[0] != j:
            continue

        merged = route_i + route_j
        if _route_load(merged, inst) > inst.capacity + 1e-6:
            continue
        if not _route_is_time_feasible(merged, inst):
            continue

        # aceita a fusão
        for c in merged:
            route_of[c] = ri
        routes[ri] = merged
        del routes[rj]

    return list(routes.values())


def routes_to_perm(routes: list[list[int]]) -> list[int]:
    """Converte uma lista de rotas em um giant-tour (achatando), para servir
    de cromossomo inicial do GA / NSGA-II (usado no warm-start, D2)."""
    perm = []
    for r in routes:
        perm.extend(r)
    return perm


if __name__ == "__main__":
    from instance import load_solomon_instance, evaluate

    inst = load_solomon_instance(str(Path(__file__).resolve().parent.parent / "data" / "C101.txt"), n_customers=25)
    routes = clarke_wright(inst)
    total_dist = sum(route_distance(r, inst) for r in routes)
    print(f"[Baseline Clarke-Wright] rotas = {len(routes)} | distância total = {total_dist:.2f}")
    for k, r in enumerate(routes, 1):
        print(f"  Rota {k}: depósito -> {r} -> depósito  (carga={_route_load(r, inst):.0f})")

    perm = routes_to_perm(routes)
    res = evaluate(perm, inst)
    print(f"\nVerificação via evaluate(): f1={res['f1_distance']:.2f}  f2={res['f2_vehicles']}  f3={res['f3_load_imbalance']:.2f}")
