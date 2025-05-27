import random, time, math
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0)  Datos (mismo módulo de la tarea 2)
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

# -------- pre-cálculos --------
zona_de_salida = {k: next(z for z in zonas if s[(z, k)] == 1) for k in salidas}
salidas_por_zona = {z: [k for k in salidas if zona_de_salida[k] == z] for z in zonas}


# ------------------------------------------------------------------
# 1)  Funciones objetivo y cargas
# ------------------------------------------------------------------
def calcular_w_j(sol):
    """Devuelve dict  zona -> carga total (suma de tiempos)."""
    w = {z: 0.0 for z in zonas}
    for i, k in sol.items():
        w[zona_de_salida[k]] += tiempo[(i, k)]
    return w


def costo_sol(sol):
    """Makespan = máx carga de zona."""
    return max(calcular_w_j(sol).values())


# ------------------------------------------------------------------
# 2)  Constructor factible y estructuras auxiliares
# ------------------------------------------------------------------
def constructor_balanceado():
    """
    Asigna pedidos a zonas respetando el cupo de salidas
    y selecciona en la zona el chute más rápido.
    Devuelve:
        sol …  dict pedido -> chute
        use_set[z] … set de chutes usados por zona
        cnt[k] … nº pedidos ya asignados al chute
    """
    capacidad = {z: int(n_sal.loc[z]) for z in zonas}
    use_set = {z: set() for z in zonas}
    cnt = {k: 0 for k in salidas}
    sol = {}

    pedidos_shuf = pedidos[:]
    random.shuffle(pedidos_shuf)

    for i in pedidos_shuf:
        # zona con más capacidad libre
        z_libres = [z for z in zonas if capacidad[z] > 0]
        z_sel = (
            random.choice(z_libres)
            if len(z_libres) == 1
            else max(z_libres, key=lambda z: capacidad[z])
        )

        # mejor chute dentro de la zona
        k_best = min(salidas_por_zona[z_sel], key=lambda k: tiempo[(i, k)])

        sol[i] = k_best
        cnt[k_best] += 1
        if k_best not in use_set[z_sel]:
            use_set[z_sel].add(k_best)
            capacidad[z_sel] -= 1  # ocupamos una salida “distinta”
    return sol, use_set, cnt


# ------------------------------------------------------------------
# 3)  Vecindarios con control de factibilidad integrado
# ------------------------------------------------------------------
def vecino_relocate(sol, use_set, cnt):
    """Devuelve (nuevo_sol, (p,k_new))  o  (None,None) si no halló movimiento factible."""
    for _ in range(100):  # intenta hasta 100 veces
        p = random.choice(pedidos)
        k_old = sol[p]
        k_new = random.choice(salidas)
        if k_new == k_old:
            continue
        z_new = zona_de_salida[k_new]

        # ¿añadiría un chute distinto que desborda el cupo?
        if cnt[k_new] == 0 and len(use_set[z_new]) >= int(n_sal.loc[z_new]):
            continue  # no factible

        # --- aplicar movimiento ---
        sol2 = sol.copy()
        sol2[p] = k_new

        # actualizar estructuras (copia barata)
        use2 = {z: s.copy() for z, s in use_set.items()}
        cnt2 = cnt.copy()

        z_old = zona_de_salida[k_old]
        cnt2[k_old] -= 1
        if cnt2[k_old] == 0:
            use2[z_old].remove(k_old)
        cnt2[k_new] += 1
        use2[z_new].add(k_new)

        return sol2, (p, k_new), use2, cnt2
    return None, None, None, None  # no factible


def vecino_swap(sol):
    """Intercambia dos pedidos (si tienen chutes distintos). Siempre factible."""
    p1, p2 = random.sample(pedidos, 2)
    while sol[p1] == sol[p2]:
        p1, p2 = random.sample(pedidos, 2)
    v = sol.copy()
    v[p1], v[p2] = v[p2], v[p1]
    return v, (min(p1, p2), max(p1, p2))


# ------------------------------------------------------------------
# 4)  Hill-Climber FI  (sólo Relocate)
# ------------------------------------------------------------------
def hill_climber_FI(sol0, use_set0, cnt0, max_no_imp=4000):
    sol, use_set, cnt = sol0, use_set0, cnt0
    best_cost = costo_sol(sol)
    no_imp = 0

    while no_imp < max_no_imp:
        improved = False
        v, attr, use2, cnt2 = vecino_relocate(sol, use_set, cnt)
        if v is not None:  # movimiento factible hallado
            c_v = costo_sol(v)
            if c_v < best_cost:  # FIRST-IMPROVEMENT
                sol, use_set, cnt = v, use2, cnt2
                best_cost, improved = c_v, True
        no_imp = 0 if improved else no_imp + 1
    return sol, use_set, cnt, best_cost


# ------------------------------------------------------------------
# 5)  Búsqueda Tabú  (Relocate + Swap)
# ------------------------------------------------------------------
def tabu_search(sol0, use_set0, cnt0, tenure=7, max_iter=15000, max_no_imp=60):
    sol, use_set, cnt = sol0, use_set0, cnt0
    best_sol, best_cost = sol, costo_sol(sol)
    tabu = {}
    it = no_imp = 0

    while it < max_iter:
        it += 1
        best_neighbor, best_val, best_attr = None, math.inf, None
        best_use, best_cnt = None, None

        # ---------- Relocate ----------
        v, attr, use2, cnt2 = vecino_relocate(sol, use_set, cnt)
        if v is not None:
            c_v = costo_sol(v)
            if c_v < best_val:
                best_neighbor, best_val, best_attr = v, c_v, ("R",) + attr
                best_use, best_cnt = use2, cnt2

        # ---------- Swap ----------
        for _ in range(len(pedidos)):
            v_sw, attr_sw = vecino_swap(sol)
            if tabu.get(("S",) + attr_sw, -1) > it and costo_sol(v_sw) >= best_cost:
                continue
            c_sw = costo_sol(v_sw)
            if c_sw < best_val:
                best_neighbor, best_val = v_sw, c_sw
                best_attr = ("S",) + attr_sw
                # swap no cambia use_set ni cnt
                best_use, best_cnt = use_set, cnt

        if best_neighbor is None:
            break  # no movimiento factible encontrado

        sol, use_set, cnt = best_neighbor, best_use, best_cnt
        tabu[best_attr] = it + tenure

        if best_val < best_cost:
            best_sol, best_cost, no_imp = sol, best_val, 0
        else:
            no_imp += 1
            if no_imp >= max_no_imp:  # diversificación muy ligera
                sol, use_set, cnt = best_sol, use_set.copy(), cnt.copy()
                no_imp = 0

    return best_sol, best_cost


# ------------------------------------------------------------------
# 6)  Gráfico de barras
# ------------------------------------------------------------------
def guardar_barras(w, fname):
    zs, ts = zip(*sorted(w.items()))
    plt.figure(figsize=(8, 4))
    plt.bar(zs, ts)
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

    # Constructivo factible
    base, use_base, cnt_base = constructor_balanceado()
    c_base = costo_sol(base)

    # Hill-Climber
    t0 = time.perf_counter()
    hc, use_hc, cnt_hc, c_hc = hill_climber_FI(base, use_base, cnt_base)
    t_hc = time.perf_counter() - t0

    # Tabu Search
    t1 = time.perf_counter()
    ts, c_ts = tabu_search(hc, use_hc, cnt_hc)
    t_ts = time.perf_counter() - t1
    w_ts = calcular_w_j(ts)

    # Resumen
    print("\n=== COMPARATIVO (Entrega 3) ===")
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
    import time

    main()
