import random
import pandas as pd
import matplotlib.pyplot as plt
from solucion_tarea_1.code.lectura_datos import (
    pedidos,
    salidas,
    zonas,
    s,
    n_sal,
    tiempo,
    nombre_archivo_data,
)


def get_zone_of(k: str) -> str:
    """
    Determina la zona 'j' a la que pertenece la salida 'k' (s[(j,k)] == 1).
    Retorna None si no encuentra coincidencia.
    """
    for j in zonas:
        if s[(j, k)] == 1:
            return j
    return None


def build_greedy_minmax_random_order() -> tuple[dict, dict, float]:
    """
    Construye la solución asignando los pedidos en un orden aleatorio (shuffle),
    y manteniendo la regla local de minimización de max(W_j) en cada asignación.

    Retorna:
        X        -> dict con la asignación: X[i] = k (pedido i va a salida k)
        W        -> dict con la carga final W[j] en cada zona j
        max_time -> valor final de max(W[j])
    """

    # 1) Copiar la lista de pedidos y barajarla al azar
    pedidos_aleatorios = pedidos[:]
    random.shuffle(pedidos_aleatorios)  # Mezcla los pedidos

    # 2)
    X = {}  # Asignación final: X[pedido] = salida
    used_positions = set()  # Salidas ocupadas
    zone_usage = {j: 0 for j in zonas}  # Cuántos pedidos en cada zona
    W = {j: 0.0 for j in zonas}  # Carga total en cada zona

    # 3) Asignar cada pedido, minimizando el incremento en max(W[j])
    for i in pedidos_aleatorios:
        best_k = None
        best_valor = float("inf")

        # Intentar todas las salidas k disponibles
        for k in salidas:
            if k in used_positions:
                continue  # Esa salida ya está ocupada

            zona_k = get_zone_of(k)
            if zona_k is None:
                continue

            # Verificar si la zona_k aún tiene posiciones disponibles
            if zone_usage[zona_k] < n_sal[zona_k]:
                # Asignación temporal
                old_w = W[zona_k]
                W[zona_k] += tiempo[(i, k)]

                new_max = max(W.values())

                # Si mejora el max(W[j]) local, guardamos la opción
                if new_max < best_valor:
                    best_valor = new_max
                    best_k = k

                # Revertir
                W[zona_k] = old_w

        # Si encontramos una salida viable, la asignamos en firme
        if best_k is not None:
            X[i] = best_k
            used_positions.add(best_k)
            zona_best = get_zone_of(best_k)
            W[zona_best] += tiempo[(i, best_k)]
            zone_usage[zona_best] += 1
        else:
            pass

    max_time = max(W.values()) if len(W) > 0 else 0.0
    return X, W, max_time


def main():
    """
    Función principal que:
    1) Ejecuta build_greedy_minmax_random_order() para obtener la asignación
       con un orden aleatorio de pedidos.
    2) Genera un archivo Excel "Solucion_GreedyRandom.xlsx"
    3) Genera un gráfico de barras con las cargas finales.
    """

    # 1) Ejecutar el método con orden aleatorio
    X, W, max_time = build_greedy_minmax_random_order()

    # 2) Construir DataFrames para Excel
    if len(W) > 0:
        zona_max = max(W, key=W.get)
        maximo_carga = W[zona_max]
    else:
        zona_max = None
        maximo_carga = 0.0

    # Hoja Resumen
    df_resumen = pd.DataFrame(
        {
            "Instancia": [nombre_archivo_data],
            "Zona": [zona_max],
            "Maximo": [maximo_carga],
        }
    )

    # Hoja Solucion
    sol_rows = []
    for i in pedidos:
        salida_asignada = X[i] if i in X else None
        sol_rows.append([i, salida_asignada])

    df_sol = pd.DataFrame(sol_rows, columns=["Pedido", "Salida"])

    # Hoja Metricas
    df_metr = pd.DataFrame({"Zona": list(W.keys()), "Tiempo": list(W.values())})

    # 3) Guardar el Excel
    with pd.ExcelWriter(
        f"../soluciones_plantilla_excel/Solucion_Greedy_Random_{nombre_archivo_data[:7]}.xlsx",
        engine="openpyxl",
    ) as writer:
        df_resumen.to_excel(writer, sheet_name="Resumen", index=False)
        df_sol.to_excel(writer, sheet_name="Solucion", index=False)
        df_metr.to_excel(writer, sheet_name="Metricas", index=False)

    # 4) Crear gráfico de barras y guardarlo
    zonas_list = list(W.keys())
    cargas_list = [W[j] for j in zonas_list]

    plt.figure(figsize=(8, 4))
    plt.bar(zonas_list, cargas_list)
    plt.title("Cargas finales por zona (Greedy Aleatorio)")
    plt.xlabel("Zonas")
    plt.ylabel("Tiempo Acumulado")
    plt.savefig(
        f"../grafico_solucion/balance_greedy_random_{nombre_archivo_data[:7]}.png",
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    # Si se desea tener siempre los mismos resultados, se pudede fijar una semilla: random.seed(12345)
    main()
