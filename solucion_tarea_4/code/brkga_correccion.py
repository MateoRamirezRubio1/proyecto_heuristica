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
def decode(keys: np.ndarray):
    """
    1) Ordena pedidos por prioridad ascendente (clave aleatoria).
    2) Para cada pedido, elige la salida LIBRE que minimiza el makespan
       provisional (best-fit min-max).
    3) Retira la salida elegida de las libres (1 salida = 1 pedido).
    4) Devuelve (asignación, carga_por_zona, makespan).
    """
    # Paso 1: derivar orden de pedidos según keys
    orden = [p for _, p in sorted(zip(keys, pedidos))]

    # Inicializar cargas zonales a cero
    carga = {z: 0.0 for z in zonas}

    # Todas las salidas empiezan libres
    libres = set(salidas)

    asigna = {}

    # Paso 2: asignar cada pedido
    for p in orden:
        mejor_k, mejor_mk = None, float("inf")

        # Probar cada salida libre
        for k in libres:
            z = zona_de_salida[k]
            nueva = carga[z] + tiempo[(p, k)]
            mk_prov = max(nueva, max(carga.values()))
            # Mantener la salida que produzca el makespan provisional más bajo
            if mk_prov < mejor_mk:
                mejor_mk, mejor_k = mk_prov, k

        # Si no hay salida libre algo anda mal (instancia inviable)
        if mejor_k is None:
            raise ValueError("No quedan salidas libres para asignar el pedido.")

        # Confirmar asignación y actualizar estructuras
        asigna[p] = mejor_k
        libres.remove(mejor_k)  # R4
        carga[zona_de_salida[mejor_k]] += tiempo[(p, mejor_k)]

    # Paso 3: calcular makespan final
    makespan = max(carga.values())
    return asigna, carga, makespan


# ----------------------------------------------------------------------
#  4. Búsqueda local de intercambio ("swap") en asignaciones
# ----------------------------------------------------------------------
def mejora_swap(asigna: dict, carga: dict, mk: float, iter_max: int = 300):
    """
    Intenta iter_max intercambios aleatorios entre dos pedidos:
      - Calcula nuevo makespan si se intercambian sus salidas.
      - Acepta el swap si reduce el makespan (first-improve).
    Retorna la asignación, cargas y makespan actualizados.
    """
    pedidos_lista = list(pedidos)
    for _ in range(iter_max):
        p1, p2 = random.sample(pedidos_lista, 2)
        k1, k2 = asigna[p1], asigna[p2]
        if k1 == k2:
            continue  # mismo k, nada que intercambiar

        # Zonas originales
        z1, z2 = zona_de_salida[k1], zona_de_salida[k2]

        # Copia de cargas para prueba
        carga_tmp = carga.copy()
        # Ajuste incremental de tiempos
        carga_tmp[z1] += tiempo[(p2, k1)] - tiempo[(p1, k1)]
        carga_tmp[z2] += tiempo[(p1, k2)] - tiempo[(p2, k2)]
        nuevo_mk = max(carga_tmp.values())

        # Si mejora, aceptar intercambio
        if nuevo_mk < mk:
            asigna[p1], asigna[p2] = k2, k1
            carga, mk = carga_tmp, nuevo_mk

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
    – Inicializa población aleatoria de vectores de claves.
    – Cada generación: selecciona élite, cruza (biased crossover),
      añade mutantes, evalúa y refina élite con swaps.
    – Para si no hay mejora tras sin_mejora_max generaciones.
    """
    random.seed(seed)
    np.random.seed(seed)

    n_ped = len(pedidos)
    n_elite = int(tam_pob * frac_elite)
    n_mut = int(tam_pob * frac_mut)

    # 5.1 Población inicial
    poblacion = [np.random.rand(n_ped) for _ in range(tam_pob)]
    datos = [decode(ind) for ind in poblacion]  # lista de tuplas (asig, carga, mk)
    mejor_sol = min(datos, key=lambda d: d[2])
    sin_mejora = 0

    # 5.2 Ciclo evolutivo
    for gen in range(generaciones):
        # Seleccionar élite
        pares = sorted(zip([d[2] for d in datos], poblacion), key=lambda x: x[0])
        elite_ind = [ind.copy() for (_, ind) in pares[:n_elite]]
        no_elite = [ind for (_, ind) in pares[n_elite:]]

        # 5.2.1 Nuevo pool
        nueva = elite_ind.copy()

        # 5.2.2 Cruce sesgado (biased crossover)
        while len(nueva) < tam_pob - n_mut:
            pe = random.choice(elite_ind)
            pn = random.choice(no_elite)
            mask = np.random.rand(n_ped) < rho
            hijo = np.where(mask, pe, pn)
            nueva.append(hijo)

        # 5.2.3 Mutantes aleatorios
        nueva.extend(np.random.rand(n_mut, n_ped))

        # 5.2.4 Evaluar nueva población
        poblacion = nueva
        datos = [decode(ind) for ind in poblacion]

        # 5.2.5 Memética: refinamiento de swaps en élite
        for i in range(n_elite):
            asig, car, mk = datos[i]
            asig, car, mk = mejora_swap(asig, car, mk, iter_max=150)
            datos[i] = (asig, car, mk)

        # 5.2.6 Actualizar mejor global y chequeo de parada
        actual_mejor = min(datos, key=lambda d: d[2])
        if actual_mejor[2] < mejor_sol[2] - 1e-9:
            mejor_sol = actual_mejor
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= sin_mejora_max:
                print(f"Convergencia en generación {gen}")
                break

    # Desempaquetar solución final
    asig, car, mk = mejor_sol
    assert es_factible(asig), "Error: solución no factible."
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
