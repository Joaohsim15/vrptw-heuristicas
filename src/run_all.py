"""
run_all.py
Roda de ponta a ponta os 3 checkpoints e salva todos os artefatos
(CSV + figuras) em results/, usados pelos notebooks e pelo relatório técnico.
"""
import sys, json, csv
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # raiz do repositório
sys.path.insert(0, str(BASE_DIR / "src"))
import numpy as np

from instance import load_solomon_instance, evaluate, route_distance
from baseline import clarke_wright, routes_to_perm
from ga import run_ga
from nsga2 import run_nsga2
from utils import plot_convergence, plot_pareto_front, plot_routes

RESULTS = str(BASE_DIR / "results")
DATA = str(BASE_DIR / "data" / "C101.txt")
N_CUSTOMERS = 25
SEEDS = [1, 2, 3, 4, 5]

inst = load_solomon_instance(DATA, n_customers=N_CUSTOMERS)
print(f"Instância: {inst.name} | {inst.n_customers} clientes | capacidade={inst.capacity}\n")

# ----------------------------------------------------------------------
# CHECKPOINT 1: modelagem completa + baseline + GA single-objective + 5 sementes
# ----------------------------------------------------------------------
print(">>> Checkpoint 1: baseline + GA (5 sementes)")
cw_routes = clarke_wright(inst)
cw_perm = routes_to_perm(cw_routes)
cw_res = evaluate(cw_perm, inst)
plot_routes(cw_routes, inst, f"Baseline Clarke-Wright — dist={cw_res['f1_distance']:.1f}, veículos={cw_res['f2_vehicles']}",
            f"{RESULTS}/cp1_baseline_routes.png")

ga_results = []
for s in SEEDS:
    r = run_ga(inst, seed=s, pop_size=120, ngen=150)
    ga_results.append(r)
    print(f"  seed={s}: f1={r['f1_distance']:.2f}  f2={r['f2_vehicles']}  fitness={r['fitness']:.2f}")

with open(f"{RESULTS}/cp1_ga_5seeds.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed", "f1_distancia", "f2_veiculos", "f3_desbalanceamento", "fitness"])
    for s, r in zip(SEEDS, ga_results):
        w.writerow([s, r["f1_distance"], r["f2_vehicles"], r["f3_load_imbalance"], r["fitness"]])

f1s = [r["f1_distance"] for r in ga_results]
f2s = [r["f2_vehicles"] for r in ga_results]
print(f"  Resumo GA: f1 média={np.mean(f1s):.2f} (±{np.std(f1s):.2f}) | f2 média={np.mean(f2s):.2f} (±{np.std(f2s):.2f})")
print(f"  Baseline : f1={cw_res['f1_distance']:.2f} | f2={cw_res['f2_vehicles']}")

plot_convergence({f"GA seed={s}": r["history"] for s, r in zip(SEEDS, ga_results)},
                  "GA single-objective — convergência (5 sementes)",
                  f"{RESULTS}/cp1_ga_convergence.png")

best_ga = min(ga_results, key=lambda r: r["fitness"])
plot_routes(best_ga["routes"], inst,
            f"Melhor solução GA — dist={best_ga['f1_distance']:.1f}, veículos={best_ga['f2_vehicles']}",
            f"{RESULTS}/cp1_ga_best_routes.png")

# ----------------------------------------------------------------------
# CHECKPOINT 2: versão multiobjetivo (NSGA-II) + comparação com single-obj + baseline
# ----------------------------------------------------------------------
print("\n>>> Checkpoint 2: NSGA-II multiobjetivo (fronteira de Pareto inicial)")
res_nsga, hist_nsga = run_nsga2(inst, seed=1, pop_size=150, n_gen=30)
F = res_nsga.F
print(f"  Fronteira inicial (30 gerações): {len(F)} soluções não-dominadas")
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

# ----------------------------------------------------------------------
# CHECKPOINT 3: versão integrada com diferencial D2 (warm-start)
# ----------------------------------------------------------------------
print("\n>>> Checkpoint 3: NSGA-II integrado com warm-start (D2) vs. aleatório")
seed_perms = [cw_perm, best_ga["best_perm"]]

res_cold, hist_cold = run_nsga2(inst, seed=10, pop_size=120, n_gen=80)
res_warm, hist_warm = run_nsga2(inst, seed=10, pop_size=120, n_gen=80, seed_perms=seed_perms)

print(f"  Aleatório  : f1 inicial={hist_cold[0]:.1f} -> f1 final (80 ger.)={hist_cold[-1]:.1f}")
print(f"  Warm-start : f1 inicial={hist_warm[0]:.1f} -> f1 final (80 ger.)={hist_warm[-1]:.1f}")

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

best_warm_idx = int(np.argmin(res_warm.F[:, 0]))
best_warm_perm = None  # (a melhor solução do NSGA-II fica em res_warm.X)
plot_pareto_front(
    {
        "NSGA-II aleatório": (res_cold.F[:, 0], res_cold.F[:, 1], dict(c="tab:gray", s=70, marker="o")),
        "NSGA-II + warm-start (D2)": (res_warm.F[:, 0], res_warm.F[:, 1], dict(c="tab:green", s=80, marker="D")),
        "Baseline Clarke-Wright": ([cw_res["f1_distance"]], [cw_res["f2_vehicles"]], dict(c="tab:red", s=110, marker="*")),
    },
    "Checkpoint 3 — Fronteira final: aleatório vs. warm-start (80 gerações)",
    f"{RESULTS}/cp3_pareto_warmstart.png",
)

summary = {
    "instance": inst.name,
    "n_customers": inst.n_customers,
    "baseline": {"f1": cw_res["f1_distance"], "f2": cw_res["f2_vehicles"]},
    "ga_5seeds": {"f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                  "f2_mean": float(np.mean(f2s)), "f2_std": float(np.std(f2s)),
                  "f1_best": float(min(f1s))},
    "nsga2_initial_front": F.tolist(),
    "warmstart_effect": {"f1_random_gen0": hist_cold[0], "f1_random_gen80": hist_cold[-1],
                          "f1_warmstart_gen0": hist_warm[0], "f1_warmstart_gen80": hist_warm[-1]},
}
with open(f"{RESULTS}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nTodos os artefatos foram salvos em:", RESULTS)
