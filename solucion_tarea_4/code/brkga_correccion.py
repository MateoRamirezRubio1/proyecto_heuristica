import random
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  1. Cargar datos de la instancia
# ----------------------------------------------------------------------
from lectura_datos import (
    pedidos,  # lista de pedidos P
    salidas,  # lista de salidas K
    zonas,  # lista de zonas Z
    s,  # dict s[(z,k)] = 1 si salida k pertenece a zona z
    tiempo,  # dict tiempo[(p,k)] = tiempo en segundos para servir p por k
    nombre_archivo_data,  # nombre del archivo de datos (p.ej. "Data_40_heterogéneas.xlsx")
)

# Construir mapa salida → zona (única) a partir de s
zona_de_salida = {k: z for z in zonas for k in salidas if s[(z, k)] == 1}


# ----------------------------------------------------------------------
#  2. Función de verificación de factibilidad
# ----------------------------------------------------------------------
def es_factible(asigna: dict) -> bool:
    """
    Comprueba las 4 restricciones:
      R1) cada pedido aparece EXACTAMENTE una vez
      R2) toda salida asignada existe en el catálogo
      R3) cada salida pertenece a una zona válida
      R4) ninguna salida se reutiliza (bijección pedido↔salida)
    """
    # R1: conjunto de pedidos mapeados coincide con pedidos
    if set(asigna.keys()) != set(pedidos) or len(asigna) != len(pedidos):
        return False
    # R4: salidas únicas
    if len(set(asigna.values())) != len(asigna.values()):
        return False
    # R2: cada salida asignada está en la lista oficial
    if any(k not in salidas for k in asigna.values()):
        return False
    # R3: cada salida asignada está mapeada a una zona
    if any(k not in zona_de_salida for k in asigna.values()):
        return False
    return True


# ----------------------------------------------------------------------
#  3. Decodificador heurístico (random-keys → solución factible)
# ----------------------------------------------------------------------
def decode(keys):
    """
    1) Ordena los pedidos por prioridad ascendente (clave aleatoria).
    2) Para cada pedido en ese orden, elige la salida LIBRE que menos aumenta
       el makespan provisional (heurística best-fit min-max).
    3) Cada salida solo puede atender un pedido (1 salida = 1 pedido).
    4) Devuelve la asignación, la carga por zona y el makespan final.
    """
    # Paso 1: crear lista de pares (clave, pedido)
    pares = []
    for i in range(len(keys)):
        valor = keys[i]
        pedido = pedidos[i]
        pares.append((valor, pedido))
    # Ordenar la lista de pares por la clave (posición 0 de la tupla)
    pares.sort(key=lambda tupla: tupla[0])

    # Extraer el orden de pedidos ya ordenado
    orden = []
    for tupla in pares:
        # tupla[1] es el pedido
        orden.append(tupla[1])

    # Paso 2: inicializar cargas zonales a cero
    carga = {}
    for z in zonas:
        carga[z] = 0.0

    # Lista (conjunto) de salidas aún libres
    libres = set()
    for k in salidas:
        libres.add(k)

    # Diccionario para la asignación final
    asigna = {}

    # Recorrer cada pedido según la prioridad establecida
    for p in orden:
        mejor_salida = None
        mejor_makespan = float("inf")

        # Probar cada salida que aún esté libre
        for k in libres:
            zona = zona_de_salida[k]
            # calcular la carga de esa zona si asignamos p a k
            carga_nueva = carga[zona] + tiempo[(p, k)]
            # determinar el makespan provisional:
            # es el máximo entre carga_nueva y las cargas actuales de otras zonas
            provisional = carga_nueva
            for z2 in carga:
                if carga[z2] > provisional:
                    provisional = carga[z2]

            # quedarse con la salida que dé el menor makespan provisional
            if provisional < mejor_makespan:
                mejor_makespan = provisional
                mejor_salida = k

        # Si no se encontró ninguna salida libre, la instancia es inviable
        if mejor_salida is None:
            raise ValueError("No quedan salidas libres para asignar el pedido.")

        # Paso 3: asignar y actualizar estructuras
        asigna[p] = mejor_salida
        # quitar esa salida de las disponibles
        libres.remove(mejor_salida)
        # actualizar la carga de su zona
        zona_seleccionada = zona_de_salida[mejor_salida]
        carga[zona_seleccionada] = carga[zona_seleccionada] + tiempo[(p, mejor_salida)]

    # Paso 4: calcular el makespan final (zona con mayor carga)
    makespan = None
    for z in carga:
        if makespan is None or carga[z] > makespan:
            makespan = carga[z]

    return asigna, carga, makespan


# ----------------------------------------------------------------------
#  4. Búsqueda local de intercambio ("swap") en asignaciones
# ----------------------------------------------------------------------
def mejora_swap(asigna: dict, carga: dict, mk: float, iter_max: int = 300):
    """
    Intenta hasta iter_max intercambios aleatorios entre dos pedidos:
      - Calcula el makespan si se intercambian sus salidas.
      - Si el nuevo makespan es menor, acepta el intercambio (first-improve).
    Devuelve la asignación, las cargas y el makespan actualizados.
    """
    # 1) Convertir el iterable de pedidos en una lista para acceder por índice
    pedidos_lista = []
    for p in pedidos:
        pedidos_lista.append(p)

    # 2) Iterar hasta iter_max intentos de swap
    for _ in range(iter_max):
        # 2.1) Elegir dos índices distintos al azar
        idx1 = random.randint(0, len(pedidos_lista) - 1)
        idx2 = random.randint(0, len(pedidos_lista) - 1)
        # Asegurar que no sean iguales
        while idx2 == idx1:
            idx2 = random.randint(0, len(pedidos_lista) - 1)

        # 2.2) Obtener los pedidos y sus salidas actuales
        p1 = pedidos_lista[idx1]
        p2 = pedidos_lista[idx2]
        k1 = asigna[p1]
        k2 = asigna[p2]

        # Si ambos pedidos ya usan la misma salida, saltar
        if k1 == k2:
            continue

        # 2.3) Determinar las zonas de cada salida
        z1 = zona_de_salida[k1]
        z2 = zona_de_salida[k2]

        # 2.4) Copiar las cargas actuales para simular el swap
        carga_tmp = {}
        for z in carga:
            carga_tmp[z] = carga[z]

        # 2.5) Ajustar cargas tras el hipotético intercambio
        # Pedido p2 en salida k1 y p1 en salida k2
        carga_tmp[z1] = carga_tmp[z1] + tiempo[(p2, k1)] - tiempo[(p1, k1)]
        carga_tmp[z2] = carga_tmp[z2] + tiempo[(p1, k2)] - tiempo[(p2, k2)]

        # 2.6) Calcular el nuevo makespan
        new_mk = None
        for z in carga_tmp:
            if new_mk is None or carga_tmp[z] > new_mk:
                new_mk = carga_tmp[z]

        # 2.7) Si mejora, aceptar el swap
        if new_mk < mk:
            # Intercambiar las salidas de p1 y p2
            asigna[p1] = k2
            asigna[p2] = k1
            # Actualizar cargas y makespan actuales
            carga = carga_tmp
            mk = new_mk

    # 3) Devolver los valores posiblemente mejorados
    return asigna, carga, mk


# ----------------------------------------------------------------------
#  5. BRKGA + Memética en élite
# ----------------------------------------------------------------------
def brkga(
    generaciones: int = 1000,
    tam_pob: int = 100,
    frac_elite: float = 0.20,
    frac_mut: float = 0.10,
    rho: float = 0.7,
    sin_mejora_max: int = 100,
    seed: int = 42,
):
    """
    Biased Random-Key Genetic Algorithm con refinamiento memético:
      - generaciones: máximo de iteraciones evolutivas
      - tam_pob: número de individuos en la población
      - frac_elite: proporción de individuos élite que se conservan
      - frac_mut: proporción de mutantes nuevos en cada generación
      - rho: probabilidad de heredar el gen del padre élite en el cruce
      - sin_mejora_max: generaciones consecutivas sin mejora para detenerse
      - seed: semilla para reproducibilidad
    Retorna:
      asig  (dict): asignación pedido→salida de la mejor solución
      car   (dict): cargas por zona de esa solución
      mk    (float): makespan (valor objetivo) de esa solución
    """
    # 1) Fijar semilla para librerías random y numpy
    random.seed(seed)
    np.random.seed(seed)

    # 2) Calcular tamaños dependientes de tam_pob
    n_ped = len(pedidos)  # número de pedidos
    n_elite = int(tam_pob * frac_elite)  # cuántos van a élite
    n_mut = int(tam_pob * frac_mut)  # cuántos serán mutantes

    # 3) Generar población inicial: lista de vectores de claves
    poblacion = []
    for _ in range(tam_pob):
        vec = np.random.rand(n_ped)  # vector de claves [0,1)
        poblacion.append(vec)

    # 4) Evaluar toda la población con el decodificador
    datos = []
    for ind in poblacion:
        solucion = decode(ind)  # (asig, carga, makespan)
        datos.append(solucion)

    # 5) Identificar la mejor solución inicial
    mejor_sol = datos[0]
    for sol in datos:
        if sol[2] < mejor_sol[2]:
            mejor_sol = sol
    sin_mejora = 0

    # 6) Bucle evolutivo principal
    for gen in range(generaciones):
        # --- 6.1) Selección por fitness ---
        # Construir lista de tuplas (fitness, individuo)
        pares = []
        for i in range(len(poblacion)):
            fitness = datos[i][2]  # makespan del i-ésimo
            indiv = poblacion[i]
            pares.append((fitness, indiv))
        # Ordenar de menor a mayor fitness
        pares.sort(key=lambda tupla: tupla[0])

        # Extraer élite y no-élite
        elite_ind = []
        no_elite_ind = []
        for i in range(len(pares)):
            if i < n_elite:
                # Copia profunda para no compartir la misma matriz
                elite_ind.append(pares[i][1].copy())
            else:
                no_elite_ind.append(pares[i][1])

        # --- 6.2) Formación de nueva población ---
        nueva = []
        # 6.2.1) Copiar élite
        for ind in elite_ind:
            nueva.append(ind.copy())

        # 6.2.2) Cruce sesgado (biased crossover)
        while len(nueva) < tam_pob - n_mut:
            # Elegir un padre élite y uno no-élite al azar
            padre_e = elite_ind[random.randint(0, len(elite_ind) - 1)]
            padre_n = no_elite_ind[random.randint(0, len(no_elite_ind) - 1)]
            # Crear máscara booleana
            mask = np.random.rand(n_ped) < rho
            # Generar hijo mezclando genes
            hijo = np.where(mask, padre_e, padre_n)
            nueva.append(hijo)

        # 6.2.3) Añadir mutantes puramente aleatorios
        for _ in range(n_mut):
            mutante = np.random.rand(n_ped)
            nueva.append(mutante)

        # --- 6.3) Evaluar nueva población ---
        poblacion = nueva
        datos = []
        for ind in poblacion:
            datos.append(decode(ind))

        # --- 6.4) Memética: refinamiento de la élite por swaps ---
        for i in range(n_elite):
            asigna_i, carga_i, mk_i = datos[i]
            # Aplicar búsqueda local de intercambio
            asigna_i, carga_i, mk_i = mejora_swap(asigna_i, carga_i, mk_i, iter_max=150)
            datos[i] = (asigna_i, carga_i, mk_i)

        # --- 6.5) Actualizar la mejor solución global ---
        actual_mejor = datos[0]
        for sol in datos:
            if sol[2] < actual_mejor[2]:
                actual_mejor = sol

        if actual_mejor[2] < mejor_sol[2] - 1e-9:
            mejor_sol = actual_mejor
            sin_mejora = 0
        else:
            sin_mejora += 1
            # Detener si no hay mejora tras sin_mejora_max generaciones
            if sin_mejora >= sin_mejora_max:
                print(f"Convergencia en generación {gen}")
                break

    # 7) Desempaquetar y retornar la mejor solución hallada
    asig, car, mk = mejor_sol
    assert es_factible(asig), "Solución no factible"
    return asig, car, mk


# ----------------------------------------------------------------------
#  6. Funciones de salida: Excel y PNG
# ----------------------------------------------------------------------
def exportar_excel(asig, car, mk):
    """
    Crea carpeta 'resultados_excel', genera un .xlsx con tres hojas:
      - Resumen (makespan)
      - Asignacion (pedido→salida)
      - Carga_Zona (tiempos por zona)
    """
    Path("resultados_excel").mkdir(exist_ok=True)
    nombre = nombre_archivo_data.replace(".xlsx", "")
    ruta = Path("resultados_excel") / f"BRKGA_{nombre}.xlsx"

    # Hoja Resumen
    df_res = pd.DataFrame({"Instancia": [nombre_archivo_data], "Makespan": [mk]})
    # Hoja Asignación
    df_asg = pd.DataFrame(asig.items(), columns=["Pedido", "Salida"])
    # Hoja Carga por zona
    df_car = pd.DataFrame({"Zona": list(car.keys()), "Tiempo": list(car.values())})

    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        df_res.to_excel(writer, sheet_name="Resumen", index=False)
        df_asg.to_excel(writer, sheet_name="Asignacion", index=False)
        df_car.to_excel(writer, sheet_name="Carga_Zona", index=False)

    print("Excel guardado en:", ruta)


def guardar_png(car):
    """
    Crea carpeta 'resultados_png' y guarda un gráfico de barras
    con la carga de cada zona en un archivo .png.
    """
    Path("resultados_png").mkdir(exist_ok=True)
    nombre = nombre_archivo_data.replace(".xlsx", "")
    ruta = Path("resultados_png") / f"BRKGA_barras_{nombre}.png"

    plt.figure(figsize=(8, 4))
    plt.bar(car.keys(), car.values())
    plt.axhline(
        max(car.values()),
        ls="--",
        color="red",
        label=f"Makespan = {max(car.values()):.1f} s",
    )
    plt.title("Carga por zona — BRKGA + Memético")
    plt.xlabel("Zona")
    plt.ylabel("Tiempo total (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()

    print("PNG guardado en:", ruta)


# ----------------------------------------------------------------------
#  7. Ejecución principal
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Medir tiempo de ejecución total
    t0 = perf_counter()
    asigna, carga_z, makespan = brkga()
    t1 = perf_counter()

    # Imprimir resultados en consola
    print("\n=========== RESULTADOS ===========")
    print(f"Makespan   : {makespan:,.2f} s")
    print(f"Tiempo CPU: {t1 - t0:,.2f} s\n")
    print("Cargas por zona:")
    for z, c in sorted(carga_z.items()):
        print(f"  {z:<4} → {c:,.2f} s ({c/makespan:5.2%})")

    # Exportar a Excel y PNG
    exportar_excel(asigna, carga_z, makespan)
    guardar_png(carga_z)
