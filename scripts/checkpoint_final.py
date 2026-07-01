"""
Checkpoint Final — "Projeto final: Entrega final"

Versão final integrada do projeto VRPTW multiobjetivo.

Executa a pipeline completa e calcula o Hypervolume (HV) como métrica
de qualidade da fronteira de Pareto — quanto MAIOR o HV, MELHOR a frente
(domina mais da região de interesse do espaço de objetivos).

Rode com:  python scripts/checkpoint_final.py
"""
import sys, csv, json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

import numpy as np
from pymoo.indicators.hv import HV

from instance import load_solomon_instance, evaluate
from baseline import clarke_wright, routes_to_perm
from ga import run_ga
from nsga2 import run_nsga2
from utils import plot_convergence, plot_pareto_front, plot_routes

DATA    = str(BASE_DIR / "data" / "C101.txt")
RESULTS = str(BASE_DIR / "results")
N_CUSTOMERS = 25

# Ponto de referência para cálculo do Hypervolume.
# Deve ser dominado por TODAS as soluções que queremos avaliar — usamos um
# valor claramente pior que qualquer solução observada nos experimentos.
# f1_ref = 750 (pior observado ~686), f2_ref = 12 (pior observado = 10).
HV_REF_POINT = np.array([750.0, 12.0])


def compute_hv(F: np.ndarray) -> float:
    """Calcula o hypervolume da frente de Pareto F (shape N x 2) em relação
    ao ponto de referência HV_REF_POINT. Retorna 0.0 se F for vazia."""
    if F is None or len(F) == 0:
        return 0.0
    ind = HV(ref_point=HV_REF_POINT)
    return float(ind(F))


if __name__ == "__main__":
    inst = load_solomon_instance(DATA, n_customers=N_CUSTOMERS)
    print(f"{'='*60}")
    print(f"  VRPTW — Entrega Final  |  {inst.name}  |  {inst.n_customers} clientes")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Baseline Clarke-Wright
    # ------------------------------------------------------------------
    print("[1/5] Baseline Clarke-Wright...")
    cw_routes = clarke_wright(inst)
    cw_perm   = routes_to_perm(cw_routes)
    cw_res    = evaluate(cw_perm, inst)
    print(f"      f1 = {cw_res['f1_distance']:.2f}  |  f2 = {cw_res['f2_vehicles']} veículos")
    plot_routes(cw_routes, inst,
                "Baseline Clarke-Wright — Rotas (C101, 25 clientes)",
                f"{RESULTS}/final_baseline_routes.png")

    # ------------------------------------------------------------------
    # 2. GA single-objective (5 sementes)
    # ------------------------------------------------------------------
    print("\n[2/5] GA single-objective (5 sementes)...")
    ga_results = []
    for s in range(1, 6):
        r = run_ga(inst, seed=s, pop_size=120, ngen=150, verbose=False)
        print(f"      seed {s}: f1 = {r['f1_distance']:.2f}  f2 = {r['f2_vehicles']}")
        ga_results.append(r)

    ga_f1s   = [r["f1_distance"] for r in ga_results]
    ga_f2s   = [r["f2_vehicles"] for r in ga_results]
    best_ga  = min(ga_results, key=lambda r: r["f1_distance"])
    print(f"      Média f1 = {np.mean(ga_f1s):.2f} ± {np.std(ga_f1s):.2f}"
          f"  |  melhor f1 = {best_ga['f1_distance']:.2f} (seed "
          f"{ga_results.index(best_ga)+1})")

    plot_routes(best_ga["routes"], inst,
                f"GA — Melhor rota (seed {ga_results.index(best_ga)+1}, C101, 25 clientes)",
                f"{RESULTS}/final_ga_best_routes.png")
    plot_convergence(
        {f"GA seed {i+1}": r["history"] for i, r in enumerate(ga_results)},
        "GA single-objective — Convergência (5 sementes)",
        f"{RESULTS}/final_ga_convergence.png",
    )

    # ------------------------------------------------------------------
    # 3. NSGA-II sem warm-start (controle)
    # ------------------------------------------------------------------
    print("\n[3/5] NSGA-II sem warm-start (controle, 80 gerações)...")
    res_cold, hist_cold = run_nsga2(inst, seed=10, pop_size=120, n_gen=80)
    F_cold = res_cold.F
    hv_cold = compute_hv(F_cold)
    print(f"      Soluções na frente: {len(F_cold)}")
    print(f"      f1 mín = {F_cold[:,0].min():.2f}  |  f2 mín = {F_cold[:,1].min():.0f}")
    print(f"      Hypervolume = {hv_cold:.2f}")

    # ------------------------------------------------------------------
    # 4. NSGA-II com warm-start (D2)
    # ------------------------------------------------------------------
    print("\n[4/5] NSGA-II + warm-start (D2, 80 gerações)...")
    seed_perms = [cw_perm, best_ga["best_perm"]]
    res_warm, hist_warm = run_nsga2(inst, seed=10, pop_size=120, n_gen=80,
                                    seed_perms=seed_perms)
    F_warm = res_warm.F
    hv_warm = compute_hv(F_warm)
    print(f"      Soluções na frente: {len(F_warm)}")
    print(f"      f1 mín = {F_warm[:,0].min():.2f}  |  f2 mín = {F_warm[:,1].min():.0f}")
    print(f"      Hypervolume = {hv_warm:.2f}")

    hv_gain_pct = (hv_warm - hv_cold) / hv_cold * 100 if hv_cold > 0 else float("inf")
    print(f"\n      Ganho de HV com warm-start: +{hv_gain_pct:.1f}%")

    # ------------------------------------------------------------------
    # 5. Hypervolume do Checkpoint 2 (frente com 30 gerações, dados salvos)
    # ------------------------------------------------------------------
    print("\n[5/5] Hypervolume da frente do Checkpoint 2 (30 gerações, dados históricos)...")
    cp2_F = np.array([[524.03, 7.0], [483.67, 9.0], [494.24, 8.0]])
    hv_cp2 = compute_hv(cp2_F)
    print(f"      Hypervolume CP2 (30 gen, sem warm-start) = {hv_cp2:.2f}")
    print(f"      Hypervolume CP3 (80 gen, sem warm-start) = {hv_cold:.2f}")
    print(f"      Hypervolume CP3 (80 gen, com warm-start) = {hv_warm:.2f}")

    # ------------------------------------------------------------------
    # Gráficos finais
    # ------------------------------------------------------------------
    plot_convergence(
        {"NSGA-II sem warm-start": hist_cold,
         "NSGA-II + warm-start (D2)": hist_warm},
        "Entrega Final — Convergência: warm-start vs. aleatório (80 gerações)",
        f"{RESULTS}/final_convergence.png",
    )

    plot_pareto_front(
        {
            "NSGA-II sem warm-start (80 gen)": (
                F_cold[:, 0], F_cold[:, 1], dict(c="tab:gray", s=70, marker="o")),
            "NSGA-II + warm-start (D2, 80 gen)": (
                F_warm[:, 0], F_warm[:, 1], dict(c="tab:green", s=80, marker="D")),
            "GA (melhor semente)": (
                [best_ga["f1_distance"]], [best_ga["f2_vehicles"]],
                dict(c="tab:blue", s=120, marker="^")),
            "Baseline Clarke-Wright": (
                [cw_res["f1_distance"]], [cw_res["f2_vehicles"]],
                dict(c="tab:red", s=110, marker="*")),
        },
        "Entrega Final — Fronteira de Pareto: comparação completa",
        f"{RESULTS}/final_pareto_comparison.png",
    )

    # ------------------------------------------------------------------
    # Salva CSVs
    # ------------------------------------------------------------------
    with open(f"{RESULTS}/final_pareto_warmstart.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metodo", "f1_distancia", "f2_veiculos"])
        for row in F_warm:
            w.writerow(["nsga2_warmstart", row[0], int(row[1])])
        for row in F_cold:
            w.writerow(["nsga2_aleatorio", row[0], int(row[1])])

    with open(f"{RESULTS}/final_ga_seeds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "f1_distancia", "f2_veiculos"])
        for i, r in enumerate(ga_results):
            w.writerow([i+1, r["f1_distance"], r["f2_vehicles"]])

    # ------------------------------------------------------------------
    # Atualiza summary.json
    # ------------------------------------------------------------------
    summary = {
        "instance": "C101",
        "n_customers": N_CUSTOMERS,
        "hv_reference_point": list(HV_REF_POINT),
        "baseline": {
            "f1": cw_res["f1_distance"],
            "f2": cw_res["f2_vehicles"],
        },
        "ga_5seeds": {
            "f1_mean": float(np.mean(ga_f1s)),
            "f1_std":  float(np.std(ga_f1s)),
            "f2_mean": float(np.mean(ga_f2s)),
            "f2_std":  float(np.std(ga_f2s)),
            "f1_best": best_ga["f1_distance"],
        },
        "nsga2_cp2_front_30gen": {
            "hv": hv_cp2,
            "n_solutions": int(len(cp2_F)),
        },
        "nsga2_cold_80gen": {
            "hv": hv_cold,
            "n_solutions": int(len(F_cold)),
            "f1_min": float(F_cold[:, 0].min()),
            "f2_min": int(F_cold[:, 1].min()),
        },
        "nsga2_warmstart_80gen": {
            "hv": hv_warm,
            "n_solutions": int(len(F_warm)),
            "f1_min": float(F_warm[:, 0].min()),
            "f2_min": int(F_warm[:, 1].min()),
            "hv_gain_vs_cold_pct": hv_gain_pct,
        },
    }

    with open(f"{RESULTS}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Resumo final
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  RESUMO FINAL")
    print(f"{'='*60}")
    print(f"  Baseline C&W     : f1={cw_res['f1_distance']:.1f}  f2={cw_res['f2_vehicles']}")
    print(f"  GA (média 5 sem) : f1={np.mean(ga_f1s):.1f} ± {np.std(ga_f1s):.1f}")
    print(f"  GA (melhor)      : f1={best_ga['f1_distance']:.1f}  f2={best_ga['f2_vehicles']}")
    print(f"  NSGA-II 30 gen   : HV={hv_cp2:.1f}")
    print(f"  NSGA-II 80 gen   : HV={hv_cold:.1f}  (sem warm-start)")
    print(f"  NSGA-II 80 gen   : HV={hv_warm:.1f}  (COM warm-start D2)  *** MELHOR ***")
    print(f"  Ganho do warm-start: +{hv_gain_pct:.1f}% em hypervolume")
    print(f"\n  Artefatos salvos em results/final_*")
    print(f"  summary.json atualizado")
    print(f"{'='*60}")
