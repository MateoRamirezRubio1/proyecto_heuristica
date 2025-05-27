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
    Construye una asignación factible a partir de un vector de claves aleatorias.

    Parámetros:
      keys (list o np.ndarray): lista de valores en [0,1), uno por pedido,
                                que determinan el orden de prioridad.

    Retorna:
      asigna (dict): mapeo pedido → salida elegida
      carga (dict):  carga acumulada por zona
      makespan (float): máximo de las cargas zonales
    """
    # 1) Asociar cada pedido con su clave
    pares = []
    for i in range(len(pedidos)):
        # pares será lista de tuplas (clave, pedido)
        pares.append((keys[i], pedidos[i]))

    # 2) Ordenar por clave ascendente
    #    Esto nos da la secuencia en que asignaremos los pedidos
    pares.sort(key=lambda tupla: tupla[0])

    # 3) Extraer la lista ordenada de pedidos
    orden = []
    for clave, pedido in pares:
        orden.append(pedido)

    # 4) Inicializar la carga de cada zona a cero
    carga = {}
    for z in zonas:
        carga[z] = 0.0

    # 5) Preparar el diccionario de asignaciones
    asigna = {}

    # 6) Recorrer cada pedido según la prioridad establecida
    for p in orden:
        mejor_k = None  # guardará la mejor salida para p
        mejor_valor = float("inf")  # guardar el menor makespan provisional

        # 6.1) Probar todas las salidas disponibles
        for k in salidas:
            # Identificar zona de la salida k
            z = zona_de_salida[k]

            # Calcular la carga si asignamos p a k
            carga_nueva = carga[z] + tiempo[(p, k)]

            # Calcular el makespan provisional:
            # es el máximo entre la nueva carga y las cargas actuales
            max_actual = None
            for z2 in carga:
                if max_actual is None or carga[z2] > max_actual:
                    max_actual = carga[z2]
            if carga_nueva > max_actual:
                coste = carga_nueva
            else:
                coste = max_actual

            # Si este coste es mejor, recordar k como candidato
            if coste < mejor_valor:
                mejor_valor = coste
                mejor_k = k

        # 7) Asignar el pedido p a la mejor salida encontrada
        asigna[p] = mejor_k

        # 8) Actualizar la carga de la zona correspondiente
        z_sel = zona_de_salida[mejor_k]
        carga[z_sel] = carga[z_sel] + tiempo[(p, mejor_k)]

    # 9) Determinar el makespan final (la zona más cargada)
    makespan = None
    for z in carga:
        if makespan is None or carga[z] > makespan:
            makespan = carga[z]

    # 10) Devolver la asignación completa, las cargas y el makespan
    return asigna, carga, makespan


# -------------------------------------------------------------
#  4. BRKGA
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
    """
    Biased Random-Key Genetic Algorithm (BRKGA):
    – generaciones: número máximo de iteraciones
    – tam_pob: tamaño de la población
    – frac_elite: fracción de la población que es élite
    – frac_mut: fracción de la población que serán mutantes
    – rho: probabilidad de heredar el gen del padre élite en el cruce
    – seed: semilla para aleatoriedad reproducible
    – sin_mejora_max: generaciones consecutivas sin mejora para detenerse
    Devuelve: (asignación, carga_por_zona, makespan)
    """
    # 1) Fijar semilla para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)

    # 2) Preparar tamaños auxiliares
    n_ped = len(pedidos)  # número de pedidos
    n_elite = int(tam_pob * frac_elite)  # cuántos pasan a élite
    n_mut = int(tam_pob * frac_mut)  # cuántos mutantes

    # 3) Generar población inicial
    poblacion = []
    for _ in range(tam_pob):
        # Cada individuo es un vector de n_ped claves aleatorias [0,1)
        individuo = np.random.rand(n_ped)
        poblacion.append(individuo)

    # Decodificar cada individuo y guardar (asig, carga_zona, makespan)
    datos = []
    for ind in poblacion:
        solucion = decode(ind)  # decodificador existente
        datos.append(solucion)

    # Encontrar la mejor solución inicial
    mejor_sol = datos[0]
    for d in datos:
        if d[2] < mejor_sol[2]:
            mejor_sol = d

    sin_mejora = 0  # contador de generaciones sin mejora

    # 4) Bucle evolutivo
    for g in range(generaciones):
        # 4.1) Selección elitista
        # Construir lista de pares (fitness, individuo)
        pares = []
        for i in range(len(datos)):
            fitness = datos[i][2]  # makespan de la i-ésima solución
            indiv = poblacion[i]
            pares.append((fitness, indiv))
        # Ordenar de menor a mayor makespan
        pares.sort(key=lambda x: x[0])

        # Separar élite y no-élite
        elite = []
        no_elite = []
        for i in range(len(pares)):
            if i < n_elite:
                # Se copia el individuo para no compartir memoria
                elite.append(pares[i][1].copy())
            else:
                no_elite.append(pares[i][1])

        # 4.2) Crear nueva población
        nueva_pob = []
        # 4.2.1) Copiar élite directamente
        for ind in elite:
            nueva_pob.append(ind.copy())

        # 4.2.2) Cruces sesgados (biased crossover)
        while len(nueva_pob) < tam_pob - n_mut:
            padre_e = random.choice(elite)
            padre_n = random.choice(no_elite)
            # Máscara booleana: True con prob rho
            mask = np.random.rand(n_ped) < rho
            # Heredar gen a gen según la máscara
            hijo = np.where(mask, padre_e, padre_n)
            nueva_pob.append(hijo)

        # 4.2.3) Añadir mutantes puros
        mutantes = np.random.rand(n_mut, n_ped)
        for i in range(n_mut):
            nueva_pob.append(mutantes[i])

        # 4.3) Evaluar nueva población
        poblacion = nueva_pob
        datos = []
        for ind in poblacion:
            datos.append(decode(ind))

        # 4.4) Encontrar el mejor de la generación
        actual = datos[0]
        for d in datos:
            if d[2] < actual[2]:
                actual = d

        # 4.5) Actualizar mejor global o contar sin mejora
        if actual[2] < mejor_sol[2] - 1e-9:
            mejor_sol = actual
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= sin_mejora_max:
                print("· Convergencia temprana en generación", g)
                break

    # 5) Desempaquetar y devolver la mejor solución encontrada
    asigna, carga_z, makespan = mejor_sol
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
