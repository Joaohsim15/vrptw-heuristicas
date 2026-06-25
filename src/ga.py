"""
ga.py
Algoritmo Genético single-objective para o VRPTW (Checkpoint 1).

Representação: permutação dos clientes (giant-tour), decodificada em rotas
pelo split heurístico de instance.py.

Fitness (minimização): f1_distancia + LAMBDA_VEHICLE_PENALTY * f2_veiculos
(conforme a proposta: "no Checkpoint 1 a versão single-objective trata f1,
com f2 controlada por penalização").

Operadores:
  - crossover: Ordered Crossover (OX) -> preserva blocos de adjacência,
    adequado para problemas de roteamento.
  - mutação: shuffle de índices (troca posições) com baixa probabilidade.
  - seleção: torneio.
  - elitismo: o melhor indivíduo da geração é sempre preservado.
"""
from __future__ import annotations
import random
import numpy as np
from pathlib import Path
from deap import base, creator, tools

from instance import VRPTWInstance, load_solomon_instance, evaluate

LAMBDA_VEHICLE_PENALTY = 50.0  # custo "fixo" equivalente por veículo usado


def fitness_fn(perm0: list[int], inst: VRPTWInstance) -> float:
    perm = [c + 1 for c in perm0]  # 0-indexado (GA) -> ID real do cliente (1..n)
    res = evaluate(perm, inst)
    return res["f1_distance"] + LAMBDA_VEHICLE_PENALTY * res["f2_vehicles"]


def build_toolbox(inst: VRPTWInstance, seed_perm: list[int] | None = None):
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    n = inst.n_customers
    seed_perm0 = [c - 1 for c in seed_perm] if seed_perm is not None else None

    def init_individual():
        if seed_perm0 is not None and random.random() < 0.3:
            ind = list(seed_perm0)  # mantém a rota-base (usado no warm-start, D2)
        else:
            ind = list(range(n))
            random.shuffle(ind)
        return creator.Individual(ind)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", lambda ind: (fitness_fn(ind, inst),))
    return toolbox


def run_ga(inst: VRPTWInstance, seed: int, pop_size: int = 120, ngen: int = 150,
           cxpb: float = 0.8, mutpb: float = 0.2, seed_perm: list[int] | None = None,
           verbose: bool = False):
    random.seed(seed)
    np.random.seed(seed)

    toolbox = build_toolbox(inst, seed_perm=seed_perm)
    pop = toolbox.population(n=pop_size)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    history = []
    best = tools.selBest(pop, 1)[0]
    history.append(best.fitness.values[0])

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(pop, len(pop) - 1)  # -1 para abrir espaço ao elite
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                tools.cxOrdered(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for mut in offspring:
            if random.random() < mutpb:
                tools.mutShuffleIndexes(mut, indpb=0.05)
                del mut.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit

        offspring.append(toolbox.clone(best))  # elitismo
        pop[:] = offspring

        best = tools.selBest(pop, 1)[0]
        history.append(best.fitness.values[0])
        if verbose and gen % 25 == 0:
            print(f"  geração {gen:4d} | melhor fitness = {best.fitness.values[0]:.2f}")

    result = evaluate([c + 1 for c in best], inst)
    result["fitness"] = best.fitness.values[0]
    result["history"] = history
    result["best_perm"] = [c + 1 for c in best]
    return result


if __name__ == "__main__":
    inst = load_solomon_instance(str(Path(__file__).resolve().parent.parent / "data" / "C101.txt"), n_customers=25)
    print(f"Instância: {inst.name} ({inst.n_customers} clientes)\n")

    seeds = [1, 2, 3, 4, 5]
    print("Rodando GA single-objective com 5 sementes diferentes...\n")
    summaries = []
    for s in seeds:
        res = run_ga(inst, seed=s, pop_size=120, ngen=150)
        print(f"  seed={s} | f1(distância)={res['f1_distance']:.2f} | "
              f"f2(veículos)={res['f2_vehicles']} | fitness={res['fitness']:.2f}")
        summaries.append(res)

    f1s = [r["f1_distance"] for r in summaries]
    f2s = [r["f2_vehicles"] for r in summaries]
    print(f"\nResumo (5 sementes):")
    print(f"  f1 distância: média={np.mean(f1s):.2f}  desvio={np.std(f1s):.2f}  melhor={min(f1s):.2f}")
    print(f"  f2 veículos : média={np.mean(f2s):.2f}  desvio={np.std(f2s):.2f}  melhor={min(f2s)}")
