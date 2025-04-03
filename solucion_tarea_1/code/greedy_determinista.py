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
    Dada una salida k, retorna la zona j a la cual pertenece (s[(j,k)] == 1).
    Si no pertenece a ninguna, retorna None.
    """
    for j in zonas:
        if s[(j, k)] == 1:
            return j
    return None


def build_greedy_minmax() -> tuple[dict, dict, float]:
    """
    Construye la solución asignando cada pedido a la salida que menos incremente max(W_j).

    Retorna:
        X        -> dict con la asignación: X[i] = k
        W        -> dict con la carga final en cada zona
        max_time -> valor max(W[j]) al finalizar
    """

    # 1)
    X = {}  # Asignación: X[pedido] = salida
    used_positions = set()  # Salidas ocupadas
    zone_usage = {j: 0 for j in zonas}  # Cuántos pedidos se asignaron en cada zona
    W = {j: 0.0 for j in zonas}  # Carga total en cada zona

    # 2) Definir función para el tiempo promedio de un pedido i
    def tiempo_promedio_pedido(i: str) -> float:
        total = 0.0
        for k in salidas:
            total += tiempo[(i, k)]
        return total / len(salidas)

    # 3) Ordenar los pedidos de mayor a menor tiempo promedio
    pedidos_ordenados = sorted(pedidos, key=tiempo_promedio_pedido, reverse=True)

    # 4) Asignar cada pedido según la lógica greedily minimizando max(W_j)
    for i in pedidos_ordenados:
        best_k = None
        best_valor = float("inf")

        for k in salidas:
            # Verificar que no esté ocupada
            if k in used_positions:
                continue

            # Determinar zona a la que pertenece k
            zona_k = get_zone_of(k)
            if zona_k is None:
                continue

            # Verificar si la zona_k aún tiene cupo de posiciones
            if zone_usage[zona_k] < n_sal[zona_k]:
                # Asignación temporal
                old_w = W[zona_k]
                W[zona_k] += tiempo[(i, k)]

                new_max = max(W.values())

                # Si mejora, guardamos la opción
                if new_max < best_valor:
                    best_valor = new_max
                    best_k = k

                # Revertir
                W[zona_k] = old_w

        # Al terminar de probar todas las salidas, asignar la mejor hallada
        if best_k is not None:
            X[i] = best_k
            used_positions.add(best_k)
            zona_best = get_zone_of(best_k)
            W[zona_best] += tiempo[(i, best_k)]
            zone_usage[zona_best] += 1
        else:
            # No encontró salida factible, el pedido se queda sin asignar
            pass

    max_time = max(W.values())
    return X, W, max_time


def main():
    """
    Función principal que:
    1) Ejecuta el método constructivo build_greedy_minmax().
    2) Genera el archivo Excel
    3) Crea un gráfico de barras con las cargas finales.
    """

    # Ejecutar método constructivo
    X, W, max_time = build_greedy_minmax()

    # Preparar datos para el Excel
    # 1) Hoja Resumen
    zona_max = max(W, key=W.get)  # zona con mayor carga

    df_resumen = pd.DataFrame(
        {
            "Instancia": [nombre_archivo_data],
            "Zona": [zona_max],
            "Maximo": [W[zona_max]],
        }
    )

    # 2) Hoja Solucion
    sol_rows = []
    for i in pedidos:
        salida_asignada = X[i] if i in X else None
        sol_rows.append([i, salida_asignada])

    df_sol = pd.DataFrame(sol_rows, columns=["Pedido", "Salida"])

    # 3) Hoja Metricas
    df_metr = pd.DataFrame({"Zona": list(W.keys()), "Tiempo": list(W.values())})

    # Crear Excel y guardar las tres hojas
    with pd.ExcelWriter(
        f"../soluciones_plantilla_excel/Solucion_Greedy_{nombre_archivo_data[:7]}.xlsx",
        engine="openpyxl",
    ) as writer:
        df_resumen.to_excel(writer, sheet_name="Resumen", index=False)
        df_sol.to_excel(writer, sheet_name="Solucion", index=False)
        df_metr.to_excel(writer, sheet_name="Metricas", index=False)

    # Crear y guardar el gráfico de barras con las cargas finales
    zonas_list = list(W.keys())
    cargas_list = [W[j] for j in zonas_list]

    plt.figure(figsize=(8, 4))
    plt.bar(zonas_list, cargas_list)
    plt.title("Cargas finales por zona (Greedy Determinista)")
    plt.xlabel("Zonas")
    plt.ylabel("Tiempo Acumulado")
    plt.savefig(
        f"../grafico_solucion/balance_greedy_{nombre_archivo_data[:7]}.png",
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
