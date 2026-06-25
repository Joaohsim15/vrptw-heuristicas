"""
Checkpoint 3 — "Projeto final: Pré-final"

Entrega exigida:
  - Versão integrada com pelo menos um diferencial
  - Diferencial escolhido: WARM-START (D2) — a população inicial do NSGA-II é
    parcialmente "semeada" com a solução do baseline Clarke-Wright e com a
    melhor solução do GA single-objective, em vez de ser 100% aleatória.
  - Artefato: este script (repositório executável)
  - + Relatório técnico (ver report/relatorio_tecnico.md)

Rode com:  python3 scripts/checkpoint3_pre_final.py
"""
import sys, csv
sys.path.insert(0, "/home/claude/vrp_project/src")
import numpy as np

from instance import load_solomon_instance, evaluate
from baseline import clarke_wright, routes_to_perm
from ga import run_ga
from nsga2 import run_nsga2
from utils import plot_convergence, plot_pareto_front

DATA = "/home/claude/vrp_project/data/C101.txt"
RESULTS = "/home/claude/vrp_project/results"
N_CUSTOMERS = 25

if __name__ == "__main__":
    inst = load_solomon_instance(DATA, n_customers=N_CUSTOMERS)
    print(f"Instância: {inst.name} | {inst.n_customers} clientes\n")

    # Soluções "boas" usadas como semente do warm-start
    cw_routes = clarke_wright(inst)
    cw_perm = routes_to_perm(cw_routes)
    cw_res = evaluate(cw_perm, inst)
    best_ga = run_ga(inst, seed=5, pop_size=120, ngen=150)  # melhor seed do Checkpoint 1
    seed_perms = [cw_perm, best_ga["best_perm"]]

    print("Rodando NSGA-II com população 100% aleatória (controle, sem diferencial)...")
    res_cold, hist_cold = run_nsga2(inst, seed=10, pop_size=120, n_gen=80)

    print("Rodando NSGA-II com warm-start (Clarke-Wright + melhor GA)...")
    res_warm, hist_warm = run_nsga2(inst, seed=10, pop_size=120, n_gen=80, seed_perms=seed_perms)

    print(f"\nAleatório  : f1 geração 0 = {hist_cold[0]:.1f}  ->  f1 geração 80 = {hist_cold[-1]:.1f}")
    print(f"Warm-start : f1 geração 0 = {hist_warm[0]:.1f}  ->  f1 geração 80 = {hist_warm[-1]:.1f}")
    print(f"Baseline   : f1 = {cw_res['f1_distance']:.1f}  f2 = {cw_res['f2_vehicles']}")

    plot_convergence(
        {"NSGA-II aleatório (sem diferencial)": hist_cold,
         "NSGA-II + warm-start (D2: Clarke-Wright + GA)": hist_warm},
        "Checkpoint 3 — Efeito do diferencial de warm-start na convergência",
        f"{RESULTS}/cp3_warmstart_convergence.png",
    )

    with open(f"{RESULTS}/cp3_warmstart_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["geracao", "f1_aleatorio", "f1_warmstart"])
        for g, (a, b) in enumerate(zip(hist_cold, hist_warm)):
            w.writerow([g, a, b])

    plot_pareto_front(
        {
            "NSGA-II aleatório": (res_cold.F[:, 0], res_cold.F[:, 1], dict(c="tab:gray", s=70, marker="o")),
            "NSGA-II + warm-start (D2)": (res_warm.F[:, 0], res_warm.F[:, 1], dict(c="tab:green", s=80, marker="D")),
            "Baseline Clarke-Wright": ([cw_res["f1_distance"]], [cw_res["f2_vehicles"]], dict(c="tab:red", s=110, marker="*")),
        },
        "Checkpoint 3 — Fronteira final: aleatório vs. warm-start (80 gerações)",
        f"{RESULTS}/cp3_pareto_warmstart.png",
    )

    print(f"\nArtefatos salvos em {RESULTS}/cp3_*  |  veja também report/relatorio_tecnico.md")
