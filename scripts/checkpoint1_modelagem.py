"""
Checkpoint 1 — "Projeto final: modelagem completa"

Entrega exigida:
  - Modelagem completa do VRPTW em código
  - Uma metaheurística single-objective rodando (GA, objetivo = f1 com f2 penalizada)
  - Baseline (Clarke-Wright savings)
  - Análise inicial com 5 sementes

Rode com:  python3 scripts/checkpoint1_modelagem.py
"""
import sys, csv
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # raiz do repositório
sys.path.insert(0, str(BASE_DIR / "src"))
import numpy as np

from instance import load_solomon_instance, evaluate
from baseline import clarke_wright, routes_to_perm
from ga import run_ga
from utils import plot_convergence, plot_routes

DATA = str(BASE_DIR / "data" / "C101.txt")
RESULTS = str(BASE_DIR / "results")
N_CUSTOMERS = 25
SEEDS = [1, 2, 3, 4, 5]

if __name__ == "__main__":
    inst = load_solomon_instance(DATA, n_customers=N_CUSTOMERS)
    print(f"Instância: {inst.name} | {inst.n_customers} clientes | capacidade={inst.capacity}\n")

    # --- Baseline -------------------------------------------------------
    print("[1/2] Baseline Clarke-Wright (savings)")
    cw_routes = clarke_wright(inst)
    cw_perm = routes_to_perm(cw_routes)
    cw_res = evaluate(cw_perm, inst)
    print(f"  f1 (distância) = {cw_res['f1_distance']:.2f}")
    print(f"  f2 (veículos)  = {cw_res['f2_vehicles']}")
    plot_routes(cw_routes, inst,
                f"Baseline Clarke-Wright — dist={cw_res['f1_distance']:.1f}, veículos={cw_res['f2_vehicles']}",
                f"{RESULTS}/cp1_baseline_routes.png")

    # --- GA single-objective, 5 sementes ---------------------------------
    print("\n[2/2] GA single-objective (5 sementes)")
    ga_results = []
    for s in SEEDS:
        r = run_ga(inst, seed=s, pop_size=120, ngen=150)
        ga_results.append(r)
        print(f"  seed={s}: f1={r['f1_distance']:.2f}  f2={r['f2_vehicles']}  fitness={r['fitness']:.2f}")

    f1s = [r["f1_distance"] for r in ga_results]
    f2s = [r["f2_vehicles"] for r in ga_results]
    print(f"\nResumo GA (5 sementes): f1 média={np.mean(f1s):.2f} (±{np.std(f1s):.2f}) | "
          f"f2 média={np.mean(f2s):.2f} (±{np.std(f2s):.2f}) | melhor f1={min(f1s):.2f}")
    print(f"Baseline               : f1={cw_res['f1_distance']:.2f} | f2={cw_res['f2_vehicles']}")

    with open(f"{RESULTS}/cp1_ga_5seeds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "f1_distancia", "f2_veiculos", "f3_desbalanceamento", "fitness"])
        for s, r in zip(SEEDS, ga_results):
            w.writerow([s, r["f1_distance"], r["f2_vehicles"], r["f3_load_imbalance"], r["fitness"]])

    plot_convergence({f"GA seed={s}": r["history"] for s, r in zip(SEEDS, ga_results)},
                      "GA single-objective — convergência (5 sementes)",
                      f"{RESULTS}/cp1_ga_convergence.png")

    best_ga = min(ga_results, key=lambda r: r["fitness"])
    plot_routes(best_ga["routes"], inst,
                f"Melhor solução GA — dist={best_ga['f1_distance']:.1f}, veículos={best_ga['f2_vehicles']}",
                f"{RESULTS}/cp1_ga_best_routes.png")

    print(f"\nArtefatos salvos em {RESULTS}/cp1_*")
