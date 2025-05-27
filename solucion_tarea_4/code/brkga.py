import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------
#  1. Cargar los datos del proyecto
# -------------------------------------------------------------
from lectura_datos import (
    pedidos,
    zonas,
    salidas,
    s,  # s[(j,k)] = 1 si la salida k pertenece a la zona j
    tiempo,  # tiempo[(i,k)] = tiempo total para pedido i si se asigna a salida k
    n_sal,  # n_sal (Series con # de salidas en cada zona)
    nombre_archivo_data,  # Nombre del archivo de datos
    tiempo,
)

# salida → zona (única)
zona_de_salida = {k: z for z in zonas for k in salidas if s[(z, k)] == 1}


# -------------------------------------------------------------
#  2. Verificador interno de factibilidad
# -------------------------------------------------------------
def es_factible(asigna: dict) -> bool:
    """
    Comprueba las restricciones fundamentales del problema:
      R-1  cada pedido está asignado a EXACTAMENTE una salida válida
      R-2  la salida existe en la lista oficial
      R-3  la salida pertenece a una (y solo una) zona declarada
    Devuelve True si todo es válido, False en caso contrario.
    """
    # R-1 mismo conjunto de pedidos y biyección pedido→salida
    if set(asigna.keys()) != set(pedidos):
        return False
    if len(asigna) != len(pedidos):
        return False

    # R-2 salidas válidas
    if any(k not in salidas for k in asigna.values()):
        return False

    # R-3 cada salida asignada está mapeada a UNA zona
    if any(k not in zona_de_salida for k in asigna.values()):
        return False

    return True


# -------------------------------------------------------------
#  3. Decodificador  (random-keys → solución factible)
# -------------------------------------------------------------
def decode(keys):
    """
    Construye una asignación factible siguiendo el orden de prioridad
    indicado en las claves aleatorias.
    Retorna: asignacion, carga_por_zona, makespan
    """
    orden = [p for _, p in sorted(zip(keys, pedidos))]
    carga = {z: 0.0 for z in zonas}
    asigna = {}

    for p in orden:
        mejor_k, mejor_val = None, float("inf")
        for k in salidas:
            z = zona_de_salida[k]
            nueva = carga[z] + tiempo[(p, k)]
            coste = max(nueva, max(carga.values()))
            if coste < mejor_val:
                mejor_val, mejor_k = coste, k

        asigna[p] = mejor_k
        carga[zona_de_salida[mejor_k]] += tiempo[(p, mejor_k)]

    makespan = max(carga.values())
    return asigna, carga, makespan


# -------------------------------------------------------------
#  4. BRKGA  (plantilla cuaderno Colab)
# -------------------------------------------------------------
def brkga(
    generaciones=1000,
    tam_pob=100,
    frac_elite=0.20,
    frac_mut=0.10,
    rho=0.7,
    seed=42,
    sin_mejora_max=100,
):
    random.seed(seed)
    np.random.seed(seed)

    n_ped = len(pedidos)
    n_elite = int(tam_pob * frac_elite)
    n_mut = int(tam_pob * frac_mut)

    # 4.1 población inicial
    poblacion = [np.random.rand(n_ped) for _ in range(tam_pob)]
    datos = [decode(ind) for ind in poblacion]
    mejor_sol = min(datos, key=lambda x: x[2])
    sin_mejora = 0

    # 4.2 ciclo evolutivo
    for g in range(generaciones):
        # selección
        pares = sorted(zip([d[2] for d in datos], poblacion), key=lambda x: x[0])
        elite = [ind.copy() for (_, ind) in pares[:n_elite]]
        no_elite = [ind for (_, ind) in pares[n_elite:]]

        # nueva población
        nueva_pob = elite.copy()

        # cruces sesgados
        while len(nueva_pob) < tam_pob - n_mut:
            padre_e = random.choice(elite)
            padre_n = random.choice(no_elite)
            mask = np.random.rand(n_ped) < rho
            hijo = np.where(mask, padre_e, padre_n)
            nueva_pob.append(hijo)

        # mutantes
        nueva_pob.extend(np.random.rand(n_mut, n_ped))

        # evaluación
        poblacion = nueva_pob
        datos = [decode(ind) for ind in poblacion]
        actual = min(datos, key=lambda x: x[2])

        # actualización del mejor global
        if actual[2] < mejor_sol[2] - 1e-9:
            mejor_sol = actual
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= sin_mejora_max:
                print(f"· Convergencia temprana en la generación {g}.")
                break

    asigna, carga_z, makespan = mejor_sol

    # verificación interna
    assert es_factible(asigna), "¡Solución NO factible!"
    print("✓ Verificación pasada: la solución es factible.")

    return asigna, carga_z, makespan


def calcular_w_zonas(asigna):
    w = {z: 0.0 for z in zonas}
    for p, k in asigna.items():
        z = zona_de_salida[k]
        w[z] += tiempo[(p, k)]
    return w


def exportar_solucion_excel(asigna, w_z, w_max, nombre_archivo):
    Path("../soluciones_plantilla_excel").mkdir(exist_ok=True, parents=True)
    zona_max = max(w_z, key=w_z.get)
    df_resumen = pd.DataFrame(
        {"Instancia": [nombre_archivo_data], "Zona": [zona_max], "Maximo": [w_max]}
    )
    df_sol = pd.DataFrame(
        [(p, asigna[p]) for p in pedidos], columns=["Pedido", "Salida"]
    )
    df_met = pd.DataFrame({"Zona": list(w_z.keys()), "Tiempo": list(w_z.values())})
    ruta = f"../soluciones_plantilla_excel/{nombre_archivo}"
    with pd.ExcelWriter(ruta, engine="openpyxl") as wr:
        df_resumen.to_excel(wr, sheet_name="Resumen", index=False)
        df_sol.to_excel(wr, sheet_name="Solucion", index=False)
        df_met.to_excel(wr, sheet_name="Metricas", index=False)
    print(f"Archivo Excel guardado en: {ruta}")


def guardar_grafico_barras(w_z, nombre_png):
    Path("../grafico_solucion").mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(8, 4))
    plt.bar(list(w_z.keys()), list(w_z.values()))
    plt.title("Distribución de Tiempos por Zona (BRKGA)")
    plt.xlabel("Zonas")
    plt.ylabel("Tiempo Total")
    ruta = f"../grafico_solucion/{nombre_png}"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"Gráfico de barras guardado en: {ruta}")


# -------------------------------------------------------------
#  6. Ejecución principal
# -------------------------------------------------------------
if __name__ == "__main__":
    asigna, carga_z, makespan = brkga()

    print("\n=========== RESULTADOS ===========")
    print(f"Makespan óptimo hallado : {makespan:,.2f} s\n")
    print("Cargas por zona:")
    for z, c in sorted(carga_z.items()):
        print(f"  Zona {z:<4} → {c:,.2f} s   ({c/makespan:5.2%})")

    # --- Exportar a Excel / PNG ---
    w_z = calcular_w_zonas(asigna)
    nombre_excel = (
        f"SolucionBRKGA_{nombre_archivo_data[:7]}_"
        f"{nombre_archivo_data.rsplit('_',1)[-1]}.xlsx"
    )
    exportar_solucion_excel(asigna, w_z, makespan, nombre_excel)

    nombre_png = (
        f"BalanceBRKGA_{nombre_archivo_data[:7]}_"
        f"{nombre_archivo_data.rsplit('_',1)[-1]}.png"
    )
    guardar_grafico_barras(w_z, nombre_png)
