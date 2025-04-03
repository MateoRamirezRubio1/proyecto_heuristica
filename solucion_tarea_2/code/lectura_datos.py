import pandas as pd

nombre_archivo_data = "Data_80_Salidas_composición_zonas_heterogéneas"

excel_modelo = pd.ExcelFile(f"../Data_PTL/{nombre_archivo_data}.xlsx")

# Lectura de conjuntos
Conjunto_pedidos = pd.read_excel(excel_modelo, "Pedidos", index_col=0)  # Pedidos
Conjunto_zonas = pd.read_excel(excel_modelo, "Zonas", index_col=0)  # Zonas
Conjunto_salidas = pd.read_excel(excel_modelo, "Salidas", index_col=0)  # Salidas
Conjunto_skus = pd.read_excel(excel_modelo, "SKU", index_col=0)  # SKUs
# Conjunto_trabajadores = pd.read_excel(excel_modelo, 'Trabajadores', index_col=0) #Trabajadores

# Lectura de parametros de salidas
N_Salidas = pd.read_excel(
    excel_modelo, "Salidas_en_cada_zona", index_col=0
)  # Cantidad de salidas por cada zona
Salidas_por_zona = pd.read_excel(
    excel_modelo, "Salidas_pertenece_zona", index_col=0
)  # Parametro binario, salidas que están incluidas en cada zona
Tiempo_salidas = pd.read_excel(
    excel_modelo, "Tiempo_salida", index_col=0
)  # Tiempo para desplazarse desde el lector al punto medio de cada salida

# Lectura de parametros de SKUs
SKUS_por_pedido = pd.read_excel(
    excel_modelo, "SKU_pertenece_pedido", index_col=0
)  # Parámetro binario, SKUS que están incluidas en un pedido
Tiempo_SKU = pd.read_excel(
    excel_modelo, "Tiempo_SKU", index_col=0
)  # Tiempo total de lectura, conteo, separación, depósito de cada ref por pedidotr

# Lectura de parametros adicionales
Parametros = pd.read_excel(excel_modelo, "Parametros", index_col=0)  # Parametros

pedidos = list(Conjunto_pedidos.index)
zonas = list(Conjunto_zonas.index)
salidas = list(Conjunto_salidas.index)
skus = list(Conjunto_skus.index)


V = Parametros["v"]
ZN = Parametros["zn"]

v = float(V.iloc[0])
zn = float(ZN.iloc[0])


# Salidas que están incluidas en cada zona
s = {(j, k): Salidas_por_zona.at[j, k] for k in salidas for j in zonas}

# SKUS que están incluidas en un pedido
rp = {(i, m): SKUS_por_pedido.at[i, m] for m in skus for i in pedidos}

# Tiempo de surtir un SKU en una salida
d = {(j, k): Tiempo_salidas.at[j, k] for k in salidas for j in zonas}


tra = {(i, m): Tiempo_SKU.at[i, m] for m in skus for i in pedidos}


n_sal = N_Salidas.Num_Salidas


tiempo = {}

for i in pedidos:
    for k in salidas:
        # Encontrar la zona j tal que s[j, k] = 1
        zona_k = None
        for j in zonas:
            if s[(j, k)] == 1:
                zona_k = j
                break

        # Si por alguna razón k no pertenece a ninguna zona, lo ignoramos
        if zona_k is None:
            continue

        # Calcular la parte "distancia" = 2*(d[j,k]/v)
        dist_ik = 2.0 * (d[(zona_k, k)] / v)

        # Ahora sumamos por todos los SKUs presentes en i
        total_time = 0.0
        for m in skus:
            if rp[(i, m)] == 1:
                # El tiempo por ese SKU es t_{r_{i,m}} + dist_ik
                total_time += tra[(i, m)] + dist_ik

        # Guardamos en el diccionario
        tiempo[(i, k)] = total_time
