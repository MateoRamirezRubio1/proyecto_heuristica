# Tarea 1 - Heurística 2025

Esta sección contiene la **Tarea 1** del proyecto de la asignatura Heurística. Se incluyen los métodos constructivos (determinista y aleatorizado), la data de prueba, los informes en PDF y las soluciones en Excel.

## Tabla de Contenidos

1. [Código (`code/`)](#codigo)
   - [greedy_determinista.py](code/greedy_determinista.py)  
   - [greedy_aleatorizado.py](code/greedy_aleatorizado.py)  
   - [lectura_datos.py](code/lectura_datos.py)

2. [Data PTL (`Data_PTL/`)](#data-ptl)
   - [Data_40_Salidas\_composición\_zonas...](Data_PTL/Data_40_Salidas_composición_zonas_homogéneas.xlsx)  
   - [Data_60_Salidas\_composición\_zonas...](Data_PTL/Data_60_Salidas_composición_zonas_homogéneas.xlsx)  
   - [Data_80_Salidas\_composición\_zonas...](Data_PTL/Data_80_Salidas_composición_zonas_homogéneas.xlsx)  

3. [Gráficos de Solución (`grafico_solucion/`)](#graficos-de-solucion)
   - [balance_greedy_Data_40.png](grafico_solucion/balance_greedy_Data_40.png)  
   - [balance_greedy_Data_60.png](grafico_solucion/balance_greedy_Data_60.png)  
   - [balance_greedy_Data_80.png](grafico_solucion/balance_greedy_Data_80.png)  
   - [balance_greedy_random_Data_40.png](grafico_solucion/balance_greedy_random_Data_40.png)  
   - [balance_greedy_random_Data_60.png](grafico_solucion/balance_greedy_random_Data_60.png)  
   - [balance_greedy_random_Data_80.png](grafico_solucion/balance_greedy_random_Data_80.png)

4. [Informes (`informes/`)](#informes)
   - [informe_resultados.pdf](informes/informe_resultados.pdf)  
   - [metodo_1_greedy_determinista.pdf](informes/metodo_1_greedy_determinista.pdf)  
   - [metodo_2_greedy_aleatorizado.pdf](informes/metodo_2_greedy_aleatorizado.pdf)

5. [Soluciones en Excel (`soluciones_plantilla_excel/`)](#soluciones-en-excel)
   - [Solucion_Greedy_Data_40.xlsx](soluciones_plantilla_excel/Solucion_Greedy_Data_40.xlsx)  
   - [Solucion_Greedy_Data_60.xlsx](soluciones_plantilla_excel/Solucion_Greedy_Data_60.xlsx)  
   - [Solucion_Greedy_Data_80.xlsx](soluciones_plantilla_excel/Solucion_Greedy_Data_80.xlsx)  
   - [Solucion_Greedy_Random_Data_40.xlsx](soluciones_plantilla_excel/Solucion_Greedy_Random_Data_40.xlsx)  
   - [Solucion_Greedy_Random_Data_60.xlsx](soluciones_plantilla_excel/Solucion_Greedy_Random_Data_60.xlsx)  
   - [Solucion_Greedy_Random_Data_80.xlsx](soluciones_plantilla_excel/Solucion_Greedy_Random_Data_80.xlsx)

---

<a name="codigo"></a>
## 1. Código

En la carpeta [`code/`](code/) se encuentran:
- **`greedy_determinista.py`**: Implementación del método constructivo determinista que ordena pedidos por tiempo promedio y asigna minimizando \(\max(W_j)\).  
- **`greedy_aleatorizado.py`**: Variante que baraja el orden de pedidos de forma aleatoria antes de asignarlos.  
- **`lectura_datos.py`**: Script que carga la data (Excel) y construye los diccionarios `tiempo[(i,k)]`, `s[(j,k)]`, etc.

<a name="data-ptl"></a>
## 2. Data PTL

En la carpeta [`Data_PTL/`](Data_PTL/) se incluyen los archivos Excel con la data.  
- **Data_40**, **Data_60**, **Data_80**: Varia la cantidad de pedidos/zonas/salidas.

<a name="graficos-de-solucion"></a>
## 3. Gráficos de Solución

En [`grafico_solucion/`](grafico_solucion/) se encuentran las imágenes `.png` generadas por los scripts al finalizar la asignación.  
- `balance_greedy_Data_XX.png` (método determinista).  
- `balance_greedy_random_Data_XX.png` (método aleatorio).  

XX puede ser 40, 60 u 80 según la instancia.

<a name="informes"></a>
## 4. Informes

La carpeta [`informes/`](informes/) contiene los documentos PDF detallando cada método y el reporte de resultados:
- **`metodo_1_greedy_determinista.pdf`**  
- **`metodo_2_greedy_aleatorizado.pdf`**  
- **`informe_resultados.pdf`** (resumen comparativo de ejecución en las tres instancias).

<a name="soluciones-en-excel"></a>
## 5. Soluciones en Excel

En [`soluciones_plantilla_excel/`](soluciones_plantilla_excel/) se ubican los archivos `.xlsx` que contienen la solución según la plantilla de Excel
- `Solucion_Greedy_Data_XX.xlsx`  
- `Solucion_Greedy_Random_Data_XX.xlsx`

---

## Cómo Ejecutar el Código

1. **Instalar requerimientos**:  
   Estando en la base del proyecto, `/`
   ```bash
   pip install -r requirements.txt
   cd solucion_tarea_1
   ```

2. **Preparar la data**:  
   En el archivo: `/code/lectura_datos.py` debes cambiar el nombre de la variable `nombre_archivo_data` a alguno de los nombres de los archivos Excel de la carpeta `/Data_PTL`.

3. **Ejecutar el método**:  
   Dirigirte a la carpeta `/code` y ejecutar el método escogido:
   ```bash
   cd code
   ```

   Para el Greedy Determinista:
   ```bash
   python greedy_determinista.py
   ```

   Para el Greedy Aleatorizado:
   ```bash
   python greedy_aleatorizado.py
   ```

4. **Revisar resultados**:  
   - En la carpeta `/soluciones_plantilla_excel/` se ubicarán los archivos con las plantillas Excel de solución.
   - En la carpeta `/grafico_solucion/` se pueden ver las imágenes de la gráficas del balance final por zona.
