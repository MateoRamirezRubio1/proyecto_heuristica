import random
import math
import pandas as pd
import matplotlib.pyplot as plt

from solucion_tarea_2.code.lectura_datos import (
    pedidos,
    zonas,
    salidas,
    s,  # s[(j,k)] = 1 si la salida k pertenece a la zona j
    tiempo,  # tiempo[(i,k)] = tiempo total para pedido i si se asigna a salida k
    n_sal,  # n_sal (Series con # de salidas en cada zona)
    nombre_archivo_data,  # Nombre del archivo de datos
)


def generar_solucion_inicial():
    """
    Genera una solución inicial simple y aleatoria.
    Cada pedido se asigna a una salida distinta.
    """
    pedidos_copia = pedidos[:]
    salidas_copia = salidas[:]
    random.shuffle(pedidos_copia)
    random.shuffle(salidas_copia)

    solucion = {}
    for idx, pedido in enumerate(pedidos_copia):
        solucion[pedido] = salidas_copia[idx]
    return solucion


def es_factible(solucion):
    """
    Verifica que la asignación cumpla con:
     - No exceder el número de salidas disponibles en cada zona.
     - (Asume también que no se repite la misma salida para dos pedidos ya que está
       forzado en generar_solucion_inicial y generar_vecino).
    """
    contador_zona = {z: 0 for z in zonas}

    # Contar cuántos pedidos van a cada zona
    for pedido_i, salida_k in solucion.items():
        for z in zonas:
            if s[(z, salida_k)] == 1:
                contador_zona[z] += 1
                break

    # Verificar no superar n_sal[z]
    for z in zonas:
        if contador_zona[z] > int(n_sal.loc[z]):
            return False
    return True


def calcular_w_j(solucion):
    """
    Calcula el tiempo total W_j en cada zona j.
    Retorna un diccionario w[j].
    """
    w_zonas = {z: 0.0 for z in zonas}
    for pedido_i, salida_k in solucion.items():
        for z in zonas:
            if s[(z, salida_k)] == 1:
                w_zonas[z] += tiempo[(pedido_i, salida_k)]
                break
    return w_zonas


def costo_sol(solucion, modo="max"):
    """
    Devuelve el costo de la solución según el modo:
      - "max": max(W_j)
      - "diff": max(W_j) - min(W_j)
      - "avg": promedio(W_j)
    Por defecto, se asume "max".
    """
    w = calcular_w_j(solucion)
    valores = list(w.values())

    if modo == "max":
        return max(valores)
    elif modo == "diff":
        return max(valores) - min(valores)
    elif modo == "avg":
        return sum(valores) / len(valores)
    else:
        return max(valores)


def generar_vecino(solucion, modo="swap"):
    """
    Genera un vecino de la solución:
     - "swap": intercambia las salidas de dos pedidos.
     - "move": mueve un pedido a otra salida distinta.
    """
    vecino = solucion.copy()

    if modo == "swap":
        p1, p2 = random.sample(pedidos, 2)
        vecino[p1], vecino[p2] = vecino[p2], vecino[p1]
    else:  # "move"
        p = random.choice(pedidos)
        salida_actual = vecino[p]
        salida_nueva = random.choice(salidas)
        while salida_nueva == salida_actual:
            salida_nueva = random.choice(salidas)
        vecino[p] = salida_nueva

    return vecino


def recocido_simulado(
    T0=1000.0,
    Tmin=1.0,
    alpha=0.95,
    iter_por_temp=200,
    modo_vecindad="mix",
    objetivo="max",
):
    """
    Implementación de Recocido Simulado para el problema PTL,
    minimizando la métrica dada por 'costo_sol(..., objetivo)'.

    - T0, Tmin: temperatura inicial y mínima.
    - alpha: factor de enfriamiento.
    - iter_por_temp: número de vecinos probados por nivel de temperatura.
    - modo_vecindad: "swap", "move" o "mix" (ambos).
    - objetivo: "max", "diff", o "avg".

    Retorna (mejor_sol, mejor_costo).
    """
    sol_actual = generar_solucion_inicial()
    while not es_factible(sol_actual):
        sol_actual = generar_solucion_inicial()

    costo_actual = costo_sol(sol_actual, modo=objetivo)
    mejor_sol = sol_actual.copy()
    mejor_costo = costo_actual

    T = T0

    # Contadores de vecinos aceptados
    accepted_swaps = 0
    accepted_moves = 0

    # Bucle principal del recocido
    while T > Tmin:
        for _ in range(iter_por_temp):
            if modo_vecindad == "mix":
                op = random.choice(["swap", "move"])
            else:
                op = modo_vecindad

            vecino = generar_vecino(sol_actual, op)
            if not es_factible(vecino):
                # Si no cumple restricciones, se descarta
                continue

            costo_vecino = costo_sol(vecino, modo=objetivo)
            delta = costo_vecino - costo_actual

            if delta < 0:
                # Se acepta si es mejor
                sol_actual = vecino
                costo_actual = costo_vecino
                if op == "swap":
                    accepted_swaps += 1
                else:
                    accepted_moves += 1
            else:
                # Aceptar con prob. exp(-delta / T)
                prob_aceptar = math.exp(-delta / T)
                if random.random() < prob_aceptar:
                    sol_actual = vecino
                    costo_actual = costo_vecino
                    if op == "swap":
                        accepted_swaps += 1
                    else:
                        accepted_moves += 1

            # Actualizar mejor global
            if costo_actual < mejor_costo:
                mejor_sol = sol_actual.copy()
                mejor_costo = costo_actual

        T *= alpha

    # print(f"Se aceptaron {accepted_swaps} swaps y {accepted_moves} moves en total.")
    return mejor_sol, mejor_costo


def exportar_solucion_excel(
    solucion, w_zonas, max_w, nombre_archivo="SolucionRecocido.xlsx"
):
    """
    Exporta la solución al formato Excel
    Guarda el archivo en la carpeta ../soluciones_plantilla_excel/.
    """
    # Determinar la zona con la carga máxima
    if w_zonas:
        zona_mas_cargada = max(w_zonas, key=w_zonas.get)
    else:
        zona_mas_cargada = None

    # DataFrame "Resumen"
    df_resumen = pd.DataFrame(
        {
            "Instancia": [nombre_archivo_data],
            "Zona": [zona_mas_cargada],
            "Maximo": [max_w],
        }
    )

    # DataFrame "Solucion"
    filas_solucion = []
    for pedido_i in pedidos:
        if pedido_i in solucion:
            filas_solucion.append([pedido_i, solucion[pedido_i]])
        else:
            filas_solucion.append([pedido_i, None])

    df_sol = pd.DataFrame(filas_solucion, columns=["Pedido", "Salida"])

    # DataFrame "Metricas"
    df_metricas = pd.DataFrame(
        {"Zona": list(w_zonas.keys()), "Tiempo": list(w_zonas.values())}
    )

    ruta_excel = f"../soluciones_plantilla_excel/{nombre_archivo}"
    with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
        df_resumen.to_excel(writer, sheet_name="Resumen", index=False)
        df_sol.to_excel(writer, sheet_name="Solucion", index=False)
        df_metricas.to_excel(writer, sheet_name="Metricas", index=False)

    print(f"Archivo Excel guardado en: {ruta_excel}")


def guardar_grafico_barras(w_zonas, nombre_imagen="GraficoRecocido.png"):
    """
    Dibuja y guarda un gráfico de barras con las cargas (W_j) por zona.
    Se almacena en ../grafico_solucion/
    """
    lista_zonas = list(w_zonas.keys())
    lista_tiempos = [w_zonas[z] for z in lista_zonas]

    plt.figure(figsize=(8, 4))
    plt.bar(lista_zonas, lista_tiempos)
    plt.title("Distribución de Tiempos por Zona (Recocido Simulado)")
    plt.xlabel("Zonas")
    plt.ylabel("Tiempo Total")
    ruta_grafico = f"../grafico_solucion/{nombre_imagen}"
    plt.savefig(ruta_grafico, bbox_inches="tight")
    plt.close()
    print(f"Gráfico de barras guardado en: {ruta_grafico}")


def main():
    """
    Función principal:
      1) Corre el recocido simulado para minimizar W_max.
      2) Verifica la factibilidad y el costo final.
      3) Exporta la solución en Excel.
      4) Guarda el gráfico de barras con la distribución final.
    """
    # Ejecutar recocido
    mejor_sol, mejor_costo = recocido_simulado(
        T0=2000.0,
        Tmin=5.0,
        alpha=0.90,
        iter_por_temp=300,
        modo_vecindad="mix",
        objetivo="max",
    )

    # # Reporte en consola
    # print(f"Mejor Costo (W_max): {mejor_costo:.3f}")

    # # Verificar factibilidad
    # factible_final = es_factible(mejor_sol)
    # print("¿Cumple restricciones?", factible_final)

    # # Calcular tiempos por zona y confirmar que coincide con el costo
    w_zonas = calcular_w_j(mejor_sol)
    # w_max_manual = max(w_zonas.values()) if w_zonas else 0
    # print("Tiempos por zona:", w_zonas)
    # print("W_max manual:", w_max_manual)
    # if abs(w_max_manual - mejor_costo) < 1e-9:
    #     print("Coincide con el costo reportado (OK).")
    # else:
    #     print("ALERTA: El costo y W_max difieren.")

    # Exportar resultados a Excel
    nombre_excel = f"SolucionRecocido_{nombre_archivo_data[:7]}_{nombre_archivo_data.rsplit('_', 1)[-1]}.xlsx"
    exportar_solucion_excel(mejor_sol, w_zonas, mejor_costo, nombre_excel)

    # Guardar el gráfico de barras
    nombre_png = f"BalanceRecocido_{nombre_archivo_data[:7]}_{nombre_archivo_data.rsplit('_', 1)[-1]}.png"

    guardar_grafico_barras(w_zonas, nombre_png)


if __name__ == "__main__":
    main()
