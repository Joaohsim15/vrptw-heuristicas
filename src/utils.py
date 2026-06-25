"""utils.py - funções de plotagem usadas nos notebooks dos 3 checkpoints."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from instance import VRPTWInstance


def plot_convergence(histories: dict[str, list[float]], title: str, path: str):
    plt.figure(figsize=(7, 4.5))
    for label, hist in histories.items():
        plt.plot(hist, label=label, linewidth=2)
    plt.xlabel("Geração")
    plt.ylabel("Melhor fitness (distância + penalização)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_pareto_front(points: dict[str, tuple], title: str, path: str):
    """points: {label: (f1_array_or_value, f2_array_or_value, style)}"""
    plt.figure(figsize=(7, 5))
    for label, (f1, f2, style) in points.items():
        plt.scatter(f1, f2, label=label, **style)
    plt.xlabel("f1 — distância total")
    plt.ylabel("f2 — número de veículos")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_routes(routes: list[list[int]], inst: VRPTWInstance, title: str, path: str):
    plt.figure(figsize=(7, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(routes), 1)))
    depot = inst.coords[0]
    plt.scatter(*depot, c="black", marker="s", s=110, zorder=5, label="Depósito")
    for k, route in enumerate(routes):
        pts = [depot] + [inst.coords[c] for c in route] + [depot]
        xs, ys = zip(*pts)
        plt.plot(xs, ys, "-o", color=colors[k], linewidth=1.6, markersize=5,
                  label=f"Rota {k+1} ({len(route)} clientes)")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(fontsize=7, loc="best", ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
