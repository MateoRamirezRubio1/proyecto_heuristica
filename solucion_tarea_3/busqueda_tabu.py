import random, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0)  Datos
# ------------------------------------------------------------------
from lectura_datos import (
    pedidos,
    zonas,
    salidas,
    s,
    tiempo,
    n_sal,
    nombre_archivo_data,
)

# pre-cálculos útiles
zona_de_salida = {k: next(z for z in zonas if s[(z, k)] == 1) for k in salidas}
salidas_por_zona = {z: [k for k in salidas if zona_de_salida[k] == z] for z in zonas}


# ------------------------------------------------------------------
# 1)  Funciones básicas y verificador
# ------------------------------------------------------------------
def es_factible(sol):
    """
    • Cada pedido tiene una salida.
    • Para cada zona z,   #salidas_distintas_utilizadas <= n_sal[z].
    """
    assert len(sol) == len(pedidos)
    usadas = {z: set() for z in zonas}
    for i, k in sol.items():
        usadas[zona_de_salida[k]].add(k)
    return all(len(usadas[z]) <= int(n_sal.loc[z]) for z in zonas)


def calcular_w_j(sol):
    w = {z: 0.0 for z in zonas}
    for i, k in sol.items():
        z = zona_de_salida[k]
        w[z] += tiempo[(i, k)]
    return w


def costo_sol(sol):
    return max(calcular_w_j(sol).values())


def verify_solution(sol, tag=""):
    assert es_factible(sol), f"❌ Solución infactible ({tag})"
    w = calcular_w_j(sol)
    assert abs(max(w.values()) - costo_sol(sol)) < 1e-6, "FO incorrecta"
    return w


# ------------------------------------------------------------------
# 2)  CONSTRUCTIVO GREEDY (factible)
# ------------------------------------------------------------------
def greedy_factible():
    """
    Asigna cada pedido a la salida más rápido **que aún cabe**
    (no supera el cupo de salidas distintas en su zona).
    """
    sol, usadas = {}, {z: set() for z in zonas}
    for i in random.sample(pedidos, k=len(pedidos)):  # orden aleatorio
        mejor_k, mejor_t = None, float("inf")
        for k in salidas_por_zona[zona_de_salida[salidas[0]]]:  # placeholder
            pass

        for k in salidas:
            z = zona_de_salida[k]
            extra = 0 if k in usadas[z] else 1
            if len(usadas[z]) + extra > int(n_sal.loc[z]):
                continue
            t = tiempo[(i, k)]
            if t < mejor_t:
                mejor_k, mejor_t = k, t
        if mejor_k is None:
            raise RuntimeError("No se halló asignación factible (revisar datos)")
        sol[i] = mejor_k
        usadas[zona_de_salida[mejor_k]].add(mejor_k)
    return sol


# ------------------------------------------------------------------
# 3)  Vecindarios
# ------------------------------------------------------------------
def vecino_relocate(sol):
    p = random.choice(pedidos)
    k_old = sol[p]
    k_new = random.choice(salidas)
    while k_new == k_old:
        k_new = random.choice(salidas)
    v = sol.copy()
    v[p] = k_new
    return v, ("R", p, k_new)


def vecino_swap(sol):
    p1, p2 = random.sample(pedidos, 2)
    while sol[p1] == sol[p2]:
        p1, p2 = random.sample(pedidos, 2)
    v = sol.copy()
    v[p1], v[p2] = v[p2], v[p1]
    return v, ("S", min(p1, p2), max(p1, p2))


# ------------------------------------------------------------------
# 4)  Hill-Climber FI  (solo Relocate)
# ------------------------------------------------------------------
def hill_climber_FI(sol0, max_no_imp=4000):
    sol, best_cost = sol0.copy(), costo_sol(sol0)
    no_imp = 0
    while no_imp < max_no_imp:
        improved = False
        for _ in range(len(pedidos) * 3):
            v, _ = vecino_relocate(sol)
            if not es_factible(v):
                continue
            c_v = costo_sol(v)
            if c_v < best_cost:
                sol, best_cost, improved = v, c_v, True
                break
        no_imp = 0 if improved else no_imp + 1
    return sol, best_cost


# ------------------------------------------------------------------
# 5)  Búsqueda Tabú  (Relocate + Swap)
# ------------------------------------------------------------------
def tabu_search(sol0, tenure=7, max_iter=20000, max_no_imp=60):
    sol = sol0.copy()
    best_sol, best_cost = sol, costo_sol(sol)
    tabu = {}
    it = no_imp = 0

    while it < max_iter:
        it += 1
        best_neighbor, best_val, best_attr = None, math.inf, None

        for gen_vecino in (vecino_relocate, vecino_swap):
            for _ in range(len(pedidos)):
                v, attr = gen_vecino(sol)
                if not es_factible(v):
                    continue
                if tabu.get(attr, -1) > it and costo_sol(v) >= best_cost:
                    continue
                c_v = costo_sol(v)
                if c_v < best_val:
                    best_neighbor, best_val, best_attr = v, c_v, attr

        if best_neighbor is None:
            break

        sol = best_neighbor
        tabu[best_attr] = it + tenure

        if best_val < best_cost:
            best_sol, best_cost, no_imp = sol.copy(), best_val, 0
        else:
            no_imp += 1
            if no_imp >= max_no_imp:  # diversificación
                for _ in range(max(1, int(0.15 * len(pedidos)))):
                    sol, _ = vecino_relocate(sol)
                    while not es_factible(sol):
                        sol, _ = vecino_relocate(sol)
                no_imp = 0

    return best_sol, best_cost


# ------------------------------------------------------------------
# 6)  Gráfico de barras
# ------------------------------------------------------------------
def guardar_barras(w, fname):
    z_, t_ = zip(*sorted(w.items()))
    plt.figure(figsize=(8, 4))
    plt.bar(z_, t_)
    plt.title("Distribución de tiempos por zona")
    plt.xlabel("Zona")
    plt.ylabel("Tiempo total")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Gráfico guardado:", fname)


# ------------------------------------------------------------------
# 7)  main()
# ------------------------------------------------------------------
def main():
    random.seed(2025)

    # Constructivo
    base = greedy_factible()
    c_base = costo_sol(base)
    verify_solution(base, "base")

    # Hill-Climber
    t0 = time.perf_counter()
    hc, c_hc = hill_climber_FI(base)
    t_hc = time.perf_counter() - t0
    verify_solution(hc, "HC")

    # Tabu Search
    t1 = time.perf_counter()
    ts, c_ts = tabu_search(hc)
    t_ts = time.perf_counter() - t1
    w_ts = verify_solution(ts, "TS")

    # Resumen
    print("\n=== COMPARATIVO (Ent. 3) ===")
    print(f"Constructivo  : {c_base:.2f}")
    print(f"Hill-Climber  : {c_hc:.2f}   ({t_hc:.1f}s)")
    print(f"Tabu Search   : {c_ts:.2f}   ({t_ts:.1f}s)")
    print(f"Mejora HC vs base = {(c_base-c_hc)/c_base*100:5.2f}%")
    print(f"Mejora TS vs base = {(c_base-c_ts)/c_base*100:5.2f}%")

    print("\nTiempos por zona (6 decimales):")
    for z, w in sorted(w_ts.items()):
        print(f"  {z}: {w:.6f}")
    print("Máximo =", max(w_ts.values()))

    guardar_barras(w_ts, f"BalanceTabu_{nombre_archivo_data}.png")


if __name__ == "__main__":
    main()
