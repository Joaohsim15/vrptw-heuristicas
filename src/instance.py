"""
instance.py
Modelagem do problema VRPTW (Vehicle Routing Problem with Time Windows).

Representação de solução: "giant tour" — uma permutação dos clientes (1..n),
SEM marcadores de rota. Um procedimento de split percorre a permutação e
decide onde quebrar em rotas, respeitando capacidade do veículo e janelas
de tempo. Isso evita ter que evoluir o número de rotas explicitamente.

Restrições modeladas:
  - cada cliente visitado exatamente uma vez (garantido pela permutação)
  - capacidade do veículo Q não pode ser excedida em nenhuma rota
  - janela de tempo [ready, due] de cada cliente: chegadas antecipadas
    esperam até "ready"; chegadas após "due" são INFACTÍVEIS -> a rota é
    quebrada antes desse cliente (ele inicia uma nova rota)
  - toda rota começa e termina no depósito (nó 0)

Objetivos:
  f1 = distância total percorrida pela frota
  f2 = número de veículos (rotas) utilizados
  f3 = desbalanceamento de carga entre veículos (desvio-padrão da ocupação)
       -> diferencial ético (D6), usado no NSGA-II de 3 objetivos
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class VRPTWInstance:
    name: str
    capacity: float
    coords: np.ndarray      # (n+1, 2), índice 0 = depósito
    demand: np.ndarray      # (n+1,)
    ready: np.ndarray       # (n+1,)
    due: np.ndarray         # (n+1,)
    service: np.ndarray     # (n+1,)
    dist: np.ndarray        # (n+1, n+1) matriz de distâncias euclidianas

    @property
    def n_customers(self) -> int:
        return len(self.coords) - 1


def load_solomon_instance(path: str, n_customers: int | None = None) -> VRPTWInstance:
    """Lê um arquivo no formato Solomon (.txt) e retorna uma VRPTWInstance.
    n_customers: se informado, usa só os primeiros N clientes (útil para
    reduzir o custo computacional em testes rápidos)."""
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() != ""]

    name = lines[0]
    # lines[1] == 'VEHICLE', lines[2] == 'NUMBER     CAPACITY', lines[3] == '25 200'
    veh_line = lines[3].split()
    capacity = float(veh_line[1])

    # lines[4] == 'CUSTOMER', lines[5] == cabeçalho das colunas
    data_lines = lines[6:]

    coords, demand, ready, due, service = [], [], [], [], []
    for line in data_lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        _, x, y, q, e, l, s = parts[:7]
        coords.append((float(x), float(y)))
        demand.append(float(q))
        ready.append(float(e))
        due.append(float(l))
        service.append(float(s))

    if n_customers is not None:
        # mantém o depósito (índice 0) + os N primeiros clientes
        coords = coords[: n_customers + 1]
        demand = demand[: n_customers + 1]
        ready = ready[: n_customers + 1]
        due = due[: n_customers + 1]
        service = service[: n_customers + 1]

    coords = np.array(coords)
    demand = np.array(demand)
    ready = np.array(ready)
    due = np.array(due)
    service = np.array(service)

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    return VRPTWInstance(name, capacity, coords, demand, ready, due, service, dist)


def split_into_routes(perm: list[int], inst: VRPTWInstance) -> list[list[int]]:
    """Decodifica um giant-tour (permutação de clientes) em uma lista de rotas
    factíveis (capacidade + janelas de tempo), usando uma heurística de split
    sequencial (greedy): percorre a permutação e só quebra a rota quando
    adicionar o próximo cliente violaria capacidade OU a janela de tempo dele."""
    routes: list[list[int]] = []
    route: list[int] = []
    load = 0.0
    time = 0.0
    prev = 0  # depósito

    for cust in perm:
        q = inst.demand[cust]
        arrival = time + inst.dist[prev, cust]
        start_service = max(arrival, inst.ready[cust])

        fits_capacity = (load + q) <= inst.capacity + 1e-6
        fits_time = start_service <= inst.due[cust] + 1e-6

        if route and (not fits_capacity or not fits_time):
            # fecha a rota atual e começa uma nova a partir do depósito
            routes.append(route)
            route = []
            load = 0.0
            time = 0.0
            prev = 0
            arrival = time + inst.dist[prev, cust]
            start_service = max(arrival, inst.ready[cust])

        route.append(cust)
        load += q
        time = start_service + inst.service[cust]
        prev = cust

    if route:
        routes.append(route)
    return routes


def route_distance(route: list[int], inst: VRPTWInstance) -> float:
    d = inst.dist[0, route[0]]
    for a, b in zip(route[:-1], route[1:]):
        d += inst.dist[a, b]
    d += inst.dist[route[-1], 0]
    return d


def evaluate(perm: list[int], inst: VRPTWInstance) -> dict:
    """Decodifica a permutação e calcula f1 (distância total), f2 (nº de
    veículos) e f3 (desbalanceamento de carga = desvio-padrão das cargas)."""
    routes = split_into_routes(perm, inst)
    total_dist = sum(route_distance(r, inst) for r in routes)
    n_vehicles = len(routes)
    loads = [sum(inst.demand[c] for c in r) for r in routes]
    load_imbalance = float(np.std(loads)) if loads else 0.0
    return {
        "f1_distance": total_dist,
        "f2_vehicles": n_vehicles,
        "f3_load_imbalance": load_imbalance,
        "routes": routes,
        "loads": loads,
    }


if __name__ == "__main__":
    inst = load_solomon_instance("/home/claude/vrp_project/data/C101.txt", n_customers=25)
    print(f"Instância: {inst.name} | clientes: {inst.n_customers} | capacidade: {inst.capacity}")
    perm = list(range(1, inst.n_customers + 1))
    res = evaluate(perm, inst)
    print(f"f1 (distância) = {res['f1_distance']:.2f}")
    print(f"f2 (veículos)  = {res['f2_vehicles']}")
    print(f"f3 (desbal.)   = {res['f3_load_imbalance']:.2f}")
