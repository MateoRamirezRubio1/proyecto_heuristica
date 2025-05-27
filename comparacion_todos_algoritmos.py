import importlib, time, random, statistics, sys
from pathlib import Path
from collections import defaultdict
from scipy import stats

import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------
# 1)  Importar módulos con sus funciones principales
# ------------------------------------------------------------------
import solucion_tarea_1.code.greedy_determinista as mod_gd
import solucion_tarea_1.code.greedy_aleatorizado as mod_ga
import solucion_tarea_2.code.recocido_simulado as mod_sa
import solucion_tarea_3.busqueda_tabu as mod_ts
import solucion_tarea_4.code.brkga as mod_brk


# ----------------------------------------------------------
# 2)  Wrappers  →  (makespan, tiempo_seg)
# ----------------------------------------------------------
def run_GD():
    t0 = time.perf_counter()
    _, _, mk = mod_gd.build_greedy_minmax()
    return mk, time.perf_counter() - t0


def run_GA():
    random.seed()
    t0 = time.perf_counter()
    _, _, mk = mod_ga.build_greedy_minmax_random_order()
    return mk, time.perf_counter() - t0


def run_SA():
    random.seed()
    t0 = time.perf_counter()
    _, mk = mod_sa.recocido_simulado()
    return mk, time.perf_counter() - t0


def run_TS():
    random.seed()
    base, u_b, c_b = mod_ts.constructor_balanceado()
    hc, u_hc, c_hc, _ = mod_ts.hill_climber_FI(base, u_b, c_b)
    t0 = time.perf_counter()
    _, mk = mod_ts.tabu_search(hc, u_hc, c_hc)
    return mk, time.perf_counter() - t0


def run_BRKGA():
    random.seed()
    t0 = time.perf_counter()
    _, _, mk = mod_brk.brkga()
    return mk, time.perf_counter() - t0


ALGORITHMS = {
    "GD": (run_GD, 5),
    "GA": (run_GA, 20),
    "SA": (run_SA, 20),
    "TS": (run_TS, 20),
    "BRKGA": (run_BRKGA, 20),
}


# ----------------------------------------------------------
# 3)  Ejecutar y recoger resultados
# ----------------------------------------------------------
def main():
    resultados = defaultdict(list)  # {alg: [makespan, ...]}
    tiempos = defaultdict(list)  # {alg: [seg, ...]}

    for nombre, (func, n) in ALGORITHMS.items():
        print(f"\n→ {nombre}: {n} corridas")
        for i in range(1, n + 1):
            mk, t = func()
            resultados[nombre].append(mk)
            tiempos[nombre].append(t)
            print(f"  corrida {i:02d} | makespan {mk:.2f} | tiempo {t:.2f}s")

    # ------------------------------------------------------
    # 4)  Estadísticas descriptivas
    # ------------------------------------------------------
    filas = []
    mejor_global = min(min(v) for v in resultados.values())

    for alg in ALGORITHMS.keys():
        mks = resultados[alg]
        ts = tiempos[alg]
        fila = {
            "Algoritmo": alg,
            "N": len(mks),
            "Mejor": min(mks),
            "Promedio": round(statistics.mean(mks), 2),
            "DesvEst": round(statistics.stdev(mks), 2) if len(mks) > 1 else 0,
            "Mediana": statistics.median(mks),
            "Gap_%": round(
                (statistics.mean(mks) - mejor_global) / mejor_global * 100, 2
            ),
            "TiempoMed_s": round(statistics.mean(ts), 2),
        }
        filas.append(fila)

    tabla = pd.DataFrame(filas).set_index("Algoritmo")
    print("\n========== RESUMEN ==========")
    print(tabla)
    tabla.to_csv("resumen_algoritmos.csv")
    print("CSV guardado como 'resumen_algoritmos.csv'")

    # ------------------------------------------------------
    # 5)  Prueba global (ANOVA o Kruskal-Wallis)
    # ------------------------------------------------------
    muestras = [resultados[alg] for alg in ALGORITHMS.keys()]
    normal = all(stats.shapiro(m)[1] > 0.05 for m in muestras if len(m) > 3)

    if normal:
        f, p = stats.f_oneway(*muestras)
        print(f"\nANOVA: F = {f:.2f}, p = {p:.4f}")
    else:
        h, p = stats.kruskal(*muestras)
        print(f"\nKruskal-Wallis: H = {h:.2f}, p = {p:.4f}")

    # ------------------------------------------------------
    # 6)  Gráficos sencillos
    # ------------------------------------------------------
    df = pd.DataFrame(
        [(alg, mk) for alg, lst in resultados.items() for mk in lst],
        columns=["Algoritmo", "Makespan"],
    )

    # box-plot
    plt.figure(figsize=(8, 5))
    df.boxplot(column="Makespan", by="Algoritmo", grid=False)
    plt.suptitle("")
    plt.title("Distribución de Makespan")
    plt.ylabel("Makespan (s)")
    plt.tight_layout()
    plt.savefig("boxplot_makespan.png", dpi=120)
    print("Box-plot guardado en 'boxplot_makespan.png'")

    # scatter tiempo-calidad
    plt.figure(figsize=(6, 4))
    for alg in ALGORITHMS.keys():
        plt.scatter(tabla.loc[alg, "TiempoMed_s"], tabla.loc[alg, "Promedio"], s=70)
        plt.text(
            tabla.loc[alg, "TiempoMed_s"] * 1.02,
            tabla.loc[alg, "Promedio"],
            alg,
            va="center",
            fontsize=9,
        )
    plt.xlabel("Tiempo medio (s)")
    plt.ylabel("Makespan medio (s)")
    plt.title("Trade-off tiempo vs calidad")
    plt.tight_layout()
    plt.savefig("scatter_tiempo_calidad.png", dpi=120)
    print("Scatter guardado en 'scatter_tiempo_calidad.png'")


if __name__ == "__main__":
    main()
