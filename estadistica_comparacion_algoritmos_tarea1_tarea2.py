import pandas as pd
import pingouin as pg


def main():
    # 1) Cargar datos del Excel
    in_excel = (
        "./solucion_tarea_2/resultados_comparacion_algoritmos/Comparacion_Global.xlsx"
    )
    df = pd.read_excel(in_excel)

    # Asegurarnos de que existen las columnas necesarias
    required_cols = {"Instancia", "Algoritmo", "Ejecución", "W_max"}
    print("Columnas del dataframe:")
    print(df.columns)
    if not required_cols.issubset(df.columns):
        print(f"ERROR: Faltan columnas en {in_excel}. Se requieren {required_cols}")
        return

    # 2) Identificar las instancias
    instancias = df["Instancia"].unique()

    # 3) Preparar una lista/dict para guardar los resultados estadísticos
    resultados_list = []

    # Recorremos cada instancia y vemos cuántos algoritmos hay
    for inst in instancias:
        df_inst = df[df["Instancia"] == inst]
        algoritmos = df_inst["Algoritmo"].unique()

        if len(algoritmos) < 2:
            print(
                f"\n[Instancia={inst}] Solo hay {len(algoritmos)} algoritmo(s). Nada que comparar."
            )
            continue

        # Caso A: 2 algoritmos, aplicamos Wilcoxon
        if len(algoritmos) == 2:
            algA, algB = algoritmos[0], algoritmos[1]
            dataA = (
                df_inst[df_inst["Algoritmo"] == algA]
                .sort_values("Ejecución")["W_max"]
                .values
            )
            dataB = (
                df_inst[df_inst["Algoritmo"] == algB]
                .sort_values("Ejecución")["W_max"]
                .values
            )

            # Aseguramos que tengan la misma longitud
            nA, nB = len(dataA), len(dataB)
            if nA == 0 or nB == 0:
                print(
                    f"Instancia={inst}: Uno de los algoritmos no tiene datos. Se omite comparación."
                )
                continue
            if nA != nB:
                min_len = min(nA, nB)
                dataA = dataA[:min_len]
                dataB = dataB[:min_len]
                print(
                    f"Atención: Se truncaron corridas en Instancia={inst} "
                    f"para que ambos algoritmos tengan la misma cantidad de datos."
                )

            wilc = pg.wilcoxon(dataA, dataB)
            if wilc.empty:
                # Si la prueba no pudo ser calculada (datos idénticos o vacíos)
                resultados_list.append(
                    {
                        "Instancia": inst,
                        "AlgoritmosComparados": f"{algA} vs {algB}",
                        "Test": "Wilcoxon",
                        "W_Stat": None,
                        "p_Value": None,
                        "Conclusion": "No se pudo calcular (datos idénticos o insuficientes)",
                    }
                )
                print(
                    f"\n[Instancia={inst}] Comparando {algA} vs {algB} con Wilcoxon: "
                    "No se pudo calcular (posibles valores idénticos)."
                )
                continue

            # wilc DataFrame con 'W-val','p-val','CLES'
            w_val = wilc["W-val"].iloc[0]
            p_val = wilc["p-val"].iloc[0]

            conclusion = "NO hay diferencia significativa (p>=0.05)"
            if p_val < 0.05:
                conclusion = "HAY diferencia significativa (p<0.05)"

            resultados_list.append(
                {
                    "Instancia": inst,
                    "AlgoritmosComparados": f"{algA} vs {algB}",
                    "Test": "Wilcoxon",
                    "W_Stat": w_val,
                    "p_Value": p_val,
                    "Conclusion": conclusion,
                }
            )
            print(f"\n[Instancia={inst}] Comparando {algA} vs {algB} con Wilcoxon:")
            print(f"   W-val={w_val:.3f}, p-val={p_val:.4g} => {conclusion}")

        # Caso B: 3 o más algoritmos, seaplicaria otra técnica como Friedman
        else:
            print("Entra en Caso B")

    # 4) Exportar resultados a Excel
    df_res_estad = pd.DataFrame(resultados_list)
    out_stats = "./solucion_tarea_2/resultados_comparacion_algoritmos/Estadistica_Comparacion_Global.xlsx"
    df_res_estad.to_excel(out_stats, index=False)
    print(f"\n\nResultados estadísticos guardados en {out_stats}\n")


if __name__ == "__main__":
    main()
