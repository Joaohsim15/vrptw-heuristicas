"""
Checkpoint 2 — "Projeto final: versão multiobjetivo"

Entrega exigida:
  - Versão multiobjetivo (NSGA-II) otimizando (f1, f2)
  - Comparação com a versão single-objective (GA) + baseline
  - Fronteira de Pareto inicial

Pré-requisito: ter rodado checkpoint1_modelagem.py antes (reaproveita baseline e GA).
Rode com:  python3 scripts/checkpoint2_multiobjetivo.py
"""
import sys, csv
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # raiz do repositório
sys.path.insert(0, str(BASE_DIR / "src"))
import numpy as np

from instance import load_solomon_instance, evaluate
from baseline import clarke_wright, routes_to_perm
from ga import run_ga
from nsga2 import run_nsga2
from utils import plot_pareto_front

DATA = str(BASE_DIR / "data" / "C101.txt")
RESULTS = str(BASE_DIR / "results")
N_CUSTOMERS = 25
SEEDS = [1, 2, 3, 4, 5]

if __name__ == "__main__":
    inst = load_solomon_instance(DATA, n_customers=N_CUSTOMERS)
    print(f"Instância: {inst.name} | {inst.n_customers} clientes\n")

    # Recalcula baseline e GA (mesmos parâmetros do Checkpoint 1) para comparação
    cw_routes = clarke_wright(inst)
    cw_res = evaluate(routes_to_perm(cw_routes), inst)

    ga_results = [run_ga(inst, seed=s, pop_size=120, ngen=150) for s in SEEDS]
    f1s = [r["f1_distance"] for r in ga_results]
    f2s = [r["f2_vehicles"] for r in ga_results]

    # --- NSGA-II: fronteira de Pareto inicial ----------------------------
    print("NSGA-II (f1=distância, f2=veículos), 30 gerações — fronteira inicial")
    res_nsga, _ = run_nsga2(inst, seed=1, pop_size=150, n_gen=30)
    F = res_nsga.F
    print(f"  {len(F)} soluções não-dominadas:")
    for f1, f2 in F[np.argsort(F[:, 0])]:
        print(f"    f1={f1:.2f}  f2={f2:.0f}")

    with open(f"{RESULTS}/cp2_pareto_front.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["f1_distancia", "f2_veiculos"])
        for f1, f2 in F:
            w.writerow([f1, f2])

    plot_pareto_front(
        {
            "NSGA-II (fronteira inicial, 30 ger.)": (F[:, 0], F[:, 1], dict(c="tab:blue", s=70, marker="o")),
            "GA single-objective (5 sementes)": (f1s, f2s, dict(c="tab:orange", s=60, marker="^")),
            "Baseline Clarke-Wright": ([cw_res["f1_distance"]], [cw_res["f2_vehicles"]], dict(c="tab:red", s=110, marker="*")),
        },
        "Checkpoint 2 — Fronteira de Pareto inicial vs. single-objective vs. baseline",
        f"{RESULTS}/cp2_pareto_comparison.png",
    )

    print(f"\nArtefatos salvos em {RESULTS}/cp2_*")
