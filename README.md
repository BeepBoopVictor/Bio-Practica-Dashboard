# Inmuno 3D - Panel interactivo del sistema inmunitario (PNAS 2023)

Dashboard en Streamlit para explorar, analizar y visualizar de forma interactiva el número total de células inmunes, su masa y distribución por tejidos/sistemas en el cuerpo humano, basado en Milo et al., PNAS (2023). Incluye visor 3D de órganos, comparativas por sistemas y tipos celulares, análisis avanzados (PCA, correlaciones, eficiencia) y un simulador de hipótesis. El ejercicio/propósito docente está descrito en el documento de la actividad de Bioinformática incluido en el repositorio.

**Autores**: Víctor Gil y María Alonso

## Contenidos del repositorio

- `main.py` - aplicación principal de Streamlit con navegación por páginas, carga de datos y visualizaciones (barras, dispersión log–log, pies, tablas) y un mosaico de análisis avanzados. Gestiona también el visor 3D con mallas y el diccionario de sistemas corporales.

- `meshloader.py` - utilidades para cargar y normalizar mallas OBJ/STL con trimesh (devuelve V, F), facilitando el renderizado 3D consistente entre órganos.

- `organs.py` - componentes y helpers para escenas 3D con Plotly (Mesh3d), estados/propiedades de órganos y coloreado por valores.


### Datos requeridos

La app espera el fichero Excel `data/pnas.2308511120.sd02.xlsx` (datos suplementarios del artículo) y un directorio `models/` con mallas 3D (`.stl`/`.obj`) para órganos (p. ej., `spleen.stl`, `liver.stl`, etc.). Las rutas se definen en la cabecera de `main.py`.

### Instalación

Requisitos recomendados: Python 3.9+ y un entorno virtual.


```
pip install streamlit pandas plotly openpyxl trimesh meshio numpy 
scikit-learn scipy
```


Estas librerías están indicadas en los comentarios de cabecera del proyecto.


---

## Preparación de datos y modelos 3D

1. Coloca el Excel suplementario en `data/pnas.2308511120.sd02.xlsx`.

2. Crea models/ y añade mallas de órganos (`.stl`/`.obj`). Los nombres de archivo se normalizan automáticamente (reemplazo de _/- y title case).

3. El diccionario `tejido:sistema` en `main.py` controla las agregaciones por sistema. Incluye normalización robusta de claves y alias (p. ej., `bone marrow:red marrow`).

4. Las utilidades de `meshloader.py` permiten normalizar mallas a una diagonal objetivo, centrarlas y unificarlas de tamaño/escala.

---

## Ejecución

Desde la raíz del proyecto:

```
streamlit run main.py
```

La app también detecta ejecución directa e imprime un recordatorio si no se lanza con Streamlit.

## Uso rápido

### Barra lateral (controles principales)

- **Cohorte:** Hombre, Mujer, Niño (10 años).

- **Métrica:** Número de células o Masa (g), con opción de escala log para barras.

- **Color 3D:** posibilidad de colorear según una simulación de adiposidad.

- **Posición 3D:** rotación Z, traslaciones (X/Y/Z) y escala de la malla.

- **Filtro por sistema/órgano** con detección automática de mallas disponibles.

- **Limpiar caché** para recargar datos/modelos.

### Secciones del proyecto (navegación superior)

La app ofrece una navegación estable entre páginas:

1. **📘 Visión general**

    Introducción al panel, cómo leer las visualizaciones y limitaciones/supuestos del modelo (cuerpo de referencia 73 kg, extrapolaciones, heterogeneidad de fuentes, instantánea estática).

2. **🧬 Distribución**

    - Barras por tejido (con opción log), resaltando el órgano seleccionado.

    - Dispersión log–log (Masa vs Células) con línea de tendencia y color por sistema.
Explica la variación de órdenes de magnitud entre tejidos y patrones de escalado.

3. **💪 Tipos celulares**

    - Barras por tipo de célula (neutrófilos, linfocitos, etc.), comparando su contribución total en células o masa.

4. **🧍 Cohortes**
    
    - Comparación Hombre vs Mujer vs Niño (top-10 tejidos) en la métrica elegida, con barras agrupadas y opción log.
  
      <img width="1388" height="832" alt="cohortePorTejido" src="https://github.com/user-attachments/assets/9f6c1e27-536b-4332-937f-1507fd3c5ce1" />


5. **🧾 Conclusiones**
    
    - Resumen automático: tejido principal, sistema dominante (según el diccionario) y contribución del órgano seleccionado, más recordatorio de limitaciones.

6. **🔬 Laboratorio de Ciencia de Datos (subsecciones)**

    - **📈 Correlaciones**
        
        Pearson entre cohortes y Spearman entre variables derivadas (p. ej., células/gramo, logs).

      <img width="1427" height="610" alt="matriz_cohorte" src="https://github.com/user-attachments/assets/59ad9b6d-7bcf-4e01-b46b-03e401381e98" />


    - **🧠 PCA**
    
        Componentes principales (2D/3D) con scikit-learn, estandarización previa y varianza explicada.

      <img width="725" height="446" alt="PCAtejido3d" src="https://github.com/user-attachments/assets/3feb9b2c-5dbd-426d-8f43-b2a80664c9a6" />

    - **⚖️ Eficiencia inmune**

        Células por gramo por tejido, top/bottom, histogramas y lectura biológica de densidad vs tamaño.

    - **👥 Variabilidad por cohorte**

        Ratios (Mujer/Hombre, Niño/Hombre) o diferencias absolutas por tejido, ordenadas por magnitud.

    - **🧪 Simulador multivariable**

        What-if con sliders para modificar múltiples tejidos (adiposo, bazo, ganglios, médula roja, hígado, músculo esquelético) y ver impacto en totales, cuotas por sistema (tartas) y Δ por tejido.

    - **🧪 Hipótesis**

        - Tres hipótesis con análisis guiado:

            **H1 - Redistribución bajo adiposidad:** sensibilidad de cuotas por sistema en puntos porcentuales, comparación base vs +X% y waterfall de Δ células.

            **H2 - Eficiencia y optimalidad:** relación log–log entre masa y células/gramo, cálculo de ρ de Spearman, tablas top/bottom de eficiencia.

            **H3 - Escalado con tamaño corporal:** alometría entre masa corporal y masa inmune, estimando exponente a partir de las cohortes con ajuste log–log.

### Visor 3D de órganos

- Carga y normalización de mallas 3D con `trimesh` (centro y escala a diagonal objetivo). Renderizado con Plotly Mesh3d y controles de color/transformación por órgano.

- Si no hay malla real, `organs.py` ofrece un fallback con cajas paramétricas y estado del órgano (color, opacidad, visibilidad, traslaciones, rotación, escala).

La siguiente imágen muestra el funcionamiento de este visor:ç

<img width="1412" height="605" alt="organo" src="https://github.com/user-attachments/assets/22cd6fe3-ce0d-4aac-aa66-b2b7c5f8bb95" />


## Estructura de datos

- **Fuente:** `data/pnas.2308511120.sd02.xlsx`. `main.py` estandariza nombres de columnas, crea un campo `tissue_norm`, define cohortes y agrega por tejido o tipo celular (suma de totales de células y masa).

- **Diccionario tejido - sistema:** configurable, con normalización robusta y alias frecuentes para evitar desajustes de clave.





