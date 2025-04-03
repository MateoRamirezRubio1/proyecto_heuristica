import importlib
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# lectura de datos y los dos algoritmos
from solucion_tarea_2.code import lectura_datos
from solucion_tarea_1.code import greedy_determinista, greedy_aleatorizado
from solucion_tarea_2.code import recocido_simulado


def set_datafile_in_lectura(datafile_path):
    lectura_datos.nombre_archivo_data = datafile_path
    importlib.reload(lectura_datos)
    # Forzar reload de los algoritmos, ellos importan lectura_datos
    importlib.reload(greedy_aleatorizado)
    importlib.reload(recocido_simulado)


def run_greedy():
    """
    Ejecuta 'build_greedy_minmax()' e identifica W_max y el tiempo.
    """
    t0 = time.time()
    X, W, max_time = greedy_aleatorizado.build_greedy_minmax_random_order()
    t1 = time.time()
    return max_time, (t1 - t0)


def run_recocido(
    T0=2000.0,
    Tmin=5.0,
    alpha=0.90,
    iter_por_temp=300,
    modo_vecindad="mix",
    objetivo="max",
):
    """
    Ejecuta 'recocido_simulado.recocido_simulado(...)', retorna (w_max, tiempo).
    """
    t0 = time.time()
    mejor_sol, mejor_wmax = recocido_simulado.recocido_simulado(
        T0=T0,
        Tmin=Tmin,
        alpha=alpha,
        iter_por_temp=iter_por_temp,
        modo_vecindad=modo_vecindad,
        objetivo=objetivo,
    )
    t1 = time.time()
    return mejor_wmax, (t1 - t0)


def plot_line_zoomed(df, out_plot="LinePlot_Wmax_Zoomed.png"):
    """
    Genera un lineplot con barras de error (±std) para cada algoritmo,
    a lo largo de las instancias.
    """
    # 1) Calcular media y std por (Instancia, Algoritmo)
    df_stats = (
        df.groupby(["Instancia", "Algoritmo"])["W_max"]
        .agg(["mean", "std"])
        .reset_index()
    )

    #   - x=Instancia
    #   - y=mean
    #   - hue=Algoritmo
    #   - error bars = std
    plt.figure(figsize=(10, 5))

    for alg in df_stats["Algoritmo"].unique():
        df_alg = df_stats[df_stats["Algoritmo"] == alg]
        plt.errorbar(
            x=df_alg["Instancia"],  # eje X
            y=df_alg["mean"],  # media en eje Y
            yerr=df_alg["std"],  # barras de error = std
            label=alg,
            fmt="o--",  # línea discontinua
            capsize=4,  # terminaciones de la barra de error
        )

    plt.title("Comparación de W_max (Media ± std) con eje Y recortado")
    plt.xlabel("Instancia")
    plt.ylabel("W_max (media ± std)")

    # 3) Recortar el eje Y para hacer zoom
    plt.ylim(550, 825)

    # Rotamos las etiquetas del eje X si se ven largas
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, bbox_inches="tight")
    plt.close()
    print(f"Se generó el gráfico en: {out_plot}")


def main():
    # Lista de todos los archivos de data
    datafiles = [
        "Data_40_Salidas_composición_zonas_homogéneas.xlsx",
        "Data_40_Salidas_composición_zonas_heterogéneas.xlsx",
        "Data_60_Salidas_composición_zonas_homogéneas.xlsx",
        "Data_60_Salidas_composición_zonas_heterogéneas.xlsx",
        "Data_80_Salidas_composición_zonas_homogéneas.xlsx",
        "Data_80_Salidas_composición_zonas_heterogéneas.xlsx",
    ]

    num_ejecuciones = 50  # cuántas ejecuciones por instancia y algoritmo

    # Acumular filas en una lista
    filas_global = []

    for datafile in datafiles:
        print(f"\n=== Cambiando a instancia: {datafile} ===")

        # 1) Ajustar lectura_datos.py para eel datafile actual
        set_datafile_in_lectura(datafile)

        # 2) Correr varias veces el Greedy
        print("Ejecutando GreedyDeterminista varias ejecuciones...")
        for ejecucion_id in range(1, num_ejecuciones + 1):
            wmax, dur = run_greedy()
            fila = {
                "Instancia": datafile,
                "Algoritmo": "GreedyDet",
                "Ejecución": ejecucion_id,
                "W_max": wmax,
                "Tiempo_s": dur,
            }
            filas_global.append(fila)

        # 3) Correr varias veces Recocido
        print("Ejecutando RecocidoSim varias ejecuciones...")
        for ejecucion_id in range(1, num_ejecuciones + 1):
            wmax, dur = run_recocido()
            fila = {
                "Instancia": datafile,
                "Algoritmo": "RecocidoSim",
                "Ejecución": ejecucion_id,
                "W_max": wmax,
                "Tiempo_s": dur,
            }
            filas_global.append(fila)

    # DataFrame global
    cols = ["Instancia", "Algoritmo", "Ejecución", "W_max", "Tiempo_s"]
    df_global = pd.DataFrame(filas_global, columns=cols)

    # Guardar DF final
    out_excel = (
        "./solucion_tarea_2/resultados_comparacion_algoritmos/Comparacion_Global.xlsx"
    )
    df_global.to_excel(out_excel, index=False)
    print(f"\nSe han guardado los resultados en {out_excel}")

    # Generar boxplot
    plot_line_zoomed(
        df_global,
        "./solucion_tarea_2/resultados_comparacion_algoritmos/Boxplot_Wmax_Instancia_Algoritmo.png",
    )


if __name__ == "__main__":
    main()
