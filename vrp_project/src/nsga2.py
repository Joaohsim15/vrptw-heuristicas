"""
nsga2.py
Versão multiobjetivo do VRPTW usando NSGA-II (pymoo) - Checkpoint 2.

Objetivos otimizados simultaneamente:
  f1 = distância total da frota
  f2 = número de veículos utilizados

(f3 - desbalanceamento de carga - fica disponível e pode ser ativado como
3º objetivo; ver USE_F3_ETHICS abaixo. Por padrão usamos (f1,f2) conforme
a proposta, mantendo f3 como extensão opcional documentada.)

Representação: mesma do GA - permutação dos clientes (giant-tour), decodificada
pelo split heurístico de instance.py. Usamos os operadores de permutação
nativos do pymoo (Order Crossover + Inversion Mutation), próprios para
problemas de roteamento/sequenciamento.

Diferencial D2 (warm-start, Checkpoint 3): a população inicial pode ser
"semeada" com a solução do baseline Clarke-Wright e com a melhor solução do
GA single-objective, em vez de 100% aleatória — ver `seed_solutions`.
"""
from __future__ import annotations
import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.optimize import minimize

from instance import VRPTWInstance, load_solomon_instance, evaluate


class VRPTWProblem(Problem):
    """x é uma permutação de 0..n-1 (cliente real = x+1)."""

    def __init__(self, inst: VRPTWInstance, use_f3_ethics: bool = False):
        self.inst = inst
        self.use_f3_ethics = use_f3_ethics
        n_obj = 3 if use_f3_ethics else 2
        super().__init__(n_var=inst.n_customers, n_obj=n_obj, xl=0, xu=inst.n_customers - 1, vtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], self.n_obj))
        for i, x in enumerate(X):
            perm = [int(c) + 1 for c in x]
            res = evaluate(perm, self.inst)
            F[i, 0] = res["f1_distance"]
            F[i, 1] = res["f2_vehicles"]
            if self.use_f3_ethics:
                F[i, 2] = res["f3_load_imbalance"]
        out["F"] = F


class SeededPermutationSampling(Sampling):
    """Gera a população inicial: uma fração 'seed_frac' dos indivíduos parte
    das soluções fornecidas (warm-start, D2); o restante é aleatório
    (PermutationRandomSampling), preservando diversidade genética."""

    def __init__(self, seed_perms0: list[list[int]], seed_frac: float = 0.25):
        super().__init__()
        self.seed_perms0 = seed_perms0  # já em 0-index
        self.seed_frac = seed_frac
        self.random_sampling = PermutationRandomSampling()

    def _do(self, problem, n_samples, **kwargs):
        X = self.random_sampling._do(problem, n_samples, **kwargs)
        n_seed = int(n_samples * self.seed_frac)
        for i in range(min(n_seed, n_samples)):
            base = self.seed_perms0[i % len(self.seed_perms0)]
            X[i, :] = np.array(base)
        return X


def run_nsga2(inst: VRPTWInstance, seed: int = 1, pop_size: int = 120, n_gen: int = 150,
              seed_perms: list[list[int]] | None = None, use_f3_ethics: bool = False):
    """seed_perms: lista de soluções (permutações com IDs 1..n) usadas para
    warm-start; se None, população 100% aleatória (sem o diferencial D2)."""
    problem = VRPTWProblem(inst, use_f3_ethics=use_f3_ethics)

    if seed_perms:
        seed_perms0 = [[c - 1 for c in p] for p in seed_perms]
        sampling = SeededPermutationSampling(seed_perms0, seed_frac=0.25)
    else:
        sampling = PermutationRandomSampling()

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=OrderCrossover(),
        mutation=InversionMutation(),
        eliminate_duplicates=True,
    )

    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False, save_history=True)

    # histórico do hypervolume/“melhor f1” por geração, útil para mostrar a
    # velocidade de convergência (warm-start vs aleatório, Checkpoint 3)
    best_f1_history = [gen.opt.get("F")[:, 0].min() for gen in res.history]

    return res, best_f1_history


if __name__ == "__main__":
    inst = load_solomon_instance("/home/claude/vrp_project/data/C101.txt", n_customers=25)
    print(f"Instância: {inst.name} ({inst.n_customers} clientes)\n")
    print("Rodando NSGA-II (f1=distância, f2=veículos), população aleatória...")
    res, hist = run_nsga2(inst, seed=1, pop_size=120, n_gen=150)
    F = res.F
    print(f"Frente de Pareto com {len(F)} soluções não-dominadas.")
    order = np.argsort(F[:, 0])
    for f1, f2 in F[order][:10]:
        print(f"  f1(distância)={f1:7.2f}  f2(veículos)={f2:.0f}")
