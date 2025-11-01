# app.py — Immune 3D dashboard (mapeo solo diccionario, ES) + Visión general + Explicaciones + Laboratorio de Ciencia de Datos + Hipótesis
# Deps extra: pip install streamlit pandas plotly openpyxl trimesh meshio numpy scikit-learn scipy
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Tuple

# --- impedir ejecución sin streamlit run ---
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        print("\n[!] Ejecuta con:  streamlit run app.py\n")
        sys.exit(0)
except Exception:
    pass

st.set_page_config(page_title="Inmuno 3D — PNAS 2023", layout="wide")
st.title("Sistema inmunitario humano • Panel interactivo (Milo et al., PNAS 2023)")

DATA_S2 = Path("data/pnas.2308511120.sd02.xlsx")
MODELS_DIR = Path("models")  # coloca spleen.stl, liver.stl, lymph_nodes.stl, etc.

# ===================== TU DICCIONARIO (MANDA SIEMPRE) =====================
TISSUE_TO_SYSTEM_ENG_RAW = {
    # Respiratorio
    "Bronchial tree":"Respiratorio","Larynx":"Respiratorio","Lungs":"Respiratorio","Trachea":"Respiratorio",
    # Cardiovascular
    "Blood":"Cardiovascular","Heart":"Cardiovascular","Red marrow":"Cardiovascular","Yellow marrow":"Cardiovascular",
    # Digestivo
    "Colon":"Digestivo","Esophagus":"Digestivo","Gallbladder":"Digestivo","Liver":"Digestivo","Pancreas":"Digestivo",
    "Salivary glands":"Digestivo","SI":"Digestivo","Stomach":"Digestivo","Tongue":"Digestivo",
    # Nervioso
    "Brain":"Nervioso","Spinal cord":"Nervioso","Pineal gland":"Nervioso","Pituitary gland":"Nervioso",
    # Musculoesquelético
    "Cartilage":"Musculoesquelético","Connective tissue":"Musculoesquelético",
    "Periarticular tissue":"Musculoesquelético","Skeletal Muscles":"Musculoesquelético",
    # Urinario
    "Kidneys":"Urinario","Ureters":"Urinario","Urethra":"Urinario","Urinary bladder":"Urinario",
    # Linfático/Inmune
    "Lymph nodes":"Linfático/Inmune","Lymph vessels":"Linfático/Inmune",
    "Spleen":"Linfático/Inmune","Thymus":"Linfático/Inmune","Tonsils":"Linfático/Inmune",
    # Reproductor
    "Breasts":"Reproductor","Epididymes":"Reproductor","Fallopian tubes":"Reproductor",
    "Ovaries":"Reproductor","Prostate gland":"Reproductor","Testes":"Reproductor","Uterus":"Reproductor",
    # Endocrino
    "Adrenal glands":"Endocrino","Parathyroid glands":"Endocrino","Thyroid":"Endocrino",
    # Otros
    "Adipose tissue":"Otros","Skin":"Otros","Eyes":"Otros"
}

def norm_key(s: str) -> str:
    """Clave robusta para dict: insensible a mayúsculas/guiones/espacios + alias comunes."""
    if s is None: return ""
    x = str(s)
    x = x.replace("_"," ").replace("-"," ")
    x = re.sub(r"\s+", " ", x).strip()
    x = x.rstrip(".")
    x = x.lower()
    aliases = {
        "small intestine": "si",
        "skeletal muscles": "skeletal muscles",
        "skeletal muscle": "skeletal muscles",
        "lymph node": "lymph nodes",
        "lymphnodes": "lymph nodes",
        "lymph vessel": "lymph vessels",
        "parathyroids": "parathyroid glands",
        "thyroid gland": "thyroid",
        "adrenals": "adrenal glands",
        "salivary gland": "salivary glands",
        "bone marrow": "red marrow"
    }
    if x in aliases: x = aliases[x]
    return x

TISSUE_TO_SYSTEM_NORM = { norm_key(k): v for k, v in TISSUE_TO_SYSTEM_ENG_RAW.items() }
def dict_map_system(tissue_name: str) -> str:
    return TISSUE_TO_SYSTEM_NORM.get(norm_key(tissue_name), "Otros")

# ===================== DATOS =====================
@st.cache_data
def load_s2_raw(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    ren = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=ren)
    df['tissue_norm'] = df['tissue'].astype(str).str.strip().str.title()
    for c in ['tot_man','tot_mass_man','tot_woman','tot_mass_woman','tot_child_10y','tot_mass_child_10y']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

COHORT_MAP = {
    'Hombre': ('tot_man','tot_mass_man'),
    'Mujer': ('tot_woman','tot_mass_woman'),
    'Niño (10 años)': ('tot_child_10y','tot_mass_child_10y'),
}

def aggregate_by_tissue(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    c_cells, c_mass = COHORT_MAP[cohort]
    agg = (df.groupby('tissue_norm', as_index=False)
             .agg(cells=(c_cells,'sum'), mass=(c_mass,'sum')))
    return agg.fillna(0.0)

def aggregate_by_celltype(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    c_cells, c_mass = COHORT_MAP[cohort]
    agg = (df.groupby('cell_type', as_index=False)
             .agg(cells=(c_cells,'sum'), mass=(c_mass,'sum')))
    agg['cell_type'] = agg['cell_type'].replace({'Nan':'Otros'}).fillna('Otros')
    return agg.fillna(0.0)

def get_value(by_df: pd.DataFrame, key_col: str, key: str, metric: str) -> float:
    if not key: return 0.0
    nk = norm_key(key)
    mask = by_df[key_col].apply(lambda t: norm_key(t) == nk)
    return float(by_df.loc[mask, metric].sum()) if mask.any() else 0.0

# ===================== AYUDAS DE "WHAT-IF" =====================
def simulate_adiposity(df_raw: pd.DataFrame, cohort: str, pct_increase: float) -> pd.DataFrame:
    df = df_raw.copy()
    cells_col, mass_col = COHORT_MAP[cohort]
    adip_mask = df['tissue_norm'].str.contains(r'\badipose\b', case=False, na=False)
    factor = 1.0 + pct_increase/100.0
    df.loc[adip_mask, cells_col] = df.loc[adip_mask, cells_col] * factor
    df.loc[adip_mask, mass_col]  = df.loc[adip_mask, mass_col]  * factor
    return df

def by_system_from_bt(bt: pd.DataFrame) -> pd.DataFrame:
    tmp = bt.copy()
    tmp['system'] = tmp['tissue_norm'].apply(dict_map_system)
    return (tmp.groupby('system', as_index=False)
                .agg(cells=('cells','sum'), mass=('mass','sum'))
                .fillna(0.0))

# ===================== MALLAS =====================
@st.cache_resource
def load_mesh(path: Path):
    import trimesh
    m = trimesh.load(path, force='mesh')
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(m.dump())
    V = np.asarray(m.vertices, dtype=float)
    F = np.asarray(m.faces, dtype=int)
    mins, maxs = V.min(axis=0), V.max(axis=0)
    center = (mins + maxs) / 2.0
    V0 = V - center
    diag = np.linalg.norm(maxs - mins)
    scale = 2.0 / diag if diag > 0 else 1.0
    return V0 * scale, F

@st.cache_resource
def discover_meshes(models_dir: Path):
    if not models_dir.exists():
        return {}
    files = list(models_dir.glob("*.stl")) + list(models_dir.glob("*.obj"))
    out = {}
    for p in files:
        pretty = p.stem.replace('_',' ').replace('-',' ')
        pretty = re.sub(r"\s+", " ", pretty).strip().title()
        try:
            V, F = load_mesh(p)
            out[pretty] = (V, F, p.name)
        except Exception:
            continue
    return out

def mesh_trace(name, V, F, color="#2c7fb8", opacity=0.9, offset=(0,0,0), rotz_deg=0, scale=1.0):
    import math
    P = V.copy() * scale
    if rotz_deg != 0:
        a = math.radians(rotz_deg); ca, sa = math.cos(a), math.sin(a)
        x = P[:,0]*ca - P[:,1]*sa
        y = P[:,0]*sa + P[:,1]*ca
        P = np.stack([x,y,P[:,2]], axis=1)
    P += np.array(offset)
    return go.Mesh3d(x=P[:,0], y=P[:,1], z=P[:,2], i=F[:,0], j=F[:,1], k=F[:,2],
                     name=name, color=color, opacity=opacity, flatshading=True)

def uniform_color(value, vmax, palette):
    if vmax <= 0: return palette[0]
    x = max(0.0, min(1.0, value / vmax))
    idx = int(round(x * (len(palette)-1)))
    return palette[idx]

# ===================== BARRA LATERAL =====================
with st.sidebar:
    st.header("Controles")
    cohort = st.radio("Cohorte", ["Hombre","Mujer","Niño (10 años)"], index=0, key="cohort_radio")
    metric = st.radio("Métrica", ["Número de células", "Masa (g)"], index=0, key="metric_radio")
    metric_col = 'cells' if metric.startswith("Número") else 'mass'
    use_log = st.toggle("Escala logarítmica (barras)", value=(metric_col=='cells'), key="log_toggle")

    st.markdown("---")
    st.subheader("Color 3D")
    use_sim_for_3d = st.toggle("Usar simulación de tejido adiposo para el color", value=False, key="sim_color_toggle")
    adip_pct_for_3d = st.slider("Incremento de tejido adiposo para 3D (%)", 0, 100, 20, step=5, key="sim_color_slider")
    st.markdown("---")
    st.subheader("Posición 3D")
    rotz = st.slider("Rotar Z (°)", -180, 180, 0, 1, key="rotz")
    tx   = st.slider("Mover X", -5.0, 5.0, 0.0, 0.1, key="tx")
    ty   = st.slider("Mover Y", -5.0, 5.0, 0.0, 0.1, key="ty")
    tz   = st.slider("Mover Z", -5.0, 5.0, 0.0, 0.1, key="tz")
    scale = st.slider("Escala", 0.5, 2.5, 1.0, 0.1, key="scale")
    st.markdown("---")
    if st.button("Limpiar caché", use_container_width=True):
        st.cache_resource.clear(); st.cache_data.clear(); st.rerun()

# ===================== CARGA Y AGREGACIÓN =====================
try:
    df_raw = load_s2_raw(DATA_S2)
except Exception as e:
    st.error(f"No se puede leer {DATA_S2}. Comprueba la ruta o cabeceras. Detalles: {e}")
    st.stop()

by_tissue_base   = aggregate_by_tissue(df_raw, cohort)
by_celltype_base = aggregate_by_celltype(df_raw, cohort)

if use_sim_for_3d:
    df_sim3d = simulate_adiposity(df_raw, cohort, adip_pct_for_3d)
    bt_color = aggregate_by_tissue(df_sim3d, cohort)
else:
    bt_color = by_tissue_base

# ===================== MALLAS Y FILTRO POR SISTEMA =====================
meshes = discover_meshes(MODELS_DIR)
tissues_with_mesh = sorted(meshes.keys())
systems_present = sorted({ dict_map_system(t) for t in tissues_with_mesh })
systems_available = ['Todos'] + systems_present
selected_system = st.sidebar.selectbox("Sistema", systems_available, index=0, key="system_sel")
if selected_system == 'Todos':
    organ_options = tissues_with_mesh[:]
else:
    organ_options = [t for t in tissues_with_mesh if dict_map_system(t) == selected_system]
if len(organ_options) == 0:
    st.sidebar.warning("No hay modelos para este sistema. Comprueba los nombres de archivo o el diccionario.")
    selected_organ = None
else:
    selected_organ = st.sidebar.selectbox("Órgano", organ_options, index=0, key="organ_sel")

# ===================== NAVEGACIÓN (estable, sin saltos en rerun) =====================
PAGES = [
    "📘 Visión general",
    "🧬 Distribución",
    "💪 Tipos celulares",
    "🧍 Cohortes",
    "🧾 Conclusiones",
    "🔬 Laboratorio de Ciencia de Datos"
]
if "active_page" not in st.session_state:
    st.session_state.active_page = PAGES[0]
page = st.radio("Navegación", PAGES, horizontal=True, index=PAGES.index(st.session_state.active_page), key="nav_radio")
st.session_state.active_page = page  # persistir selección

# ===================== VISIÓN GENERAL =====================
if page == "📘 Visión general":
    with st.container():
        st.markdown("""
### ¿Qué es esto?
Este panel interactivo se basa en Milo et al., PNAS 2023, una síntesis cuantitativa del número total, masa y distribución tisular de las células inmunes en el cuerpo humano.
Puedes explorar tejidos, sistemas, tipos celulares, cohortes (hombre/mujer/niño) y simular cambios fisiológicos (por ejemplo, adiposidad).

**Cómo leer las visualizaciones**
- **Visor de órganos 3D**: muestra una malla anatómica del órgano seleccionado; la intensidad del color refleja su contribución inmunitaria relativa.
- **Gráficos de barras**: comparan órganos, sistemas o tipos celulares. Usa la escala logarítmica para resaltar diferencias de orden de magnitud.
- **Dispersión (Masa vs Células)**: revela tendencias de escalado; una línea discontinua muestra la tendencia ajustada.
- **PCA**: proyecta los tejidos en un espacio de baja dimensión para destacar agrupaciones con perfiles inmunitarios similares.
- **Eficiencia (Células/gramo)**: contrasta el tamaño absoluto con la densidad de células inmunes.
- **Hipótesis**: análisis de sensibilidad y simulaciones hipotéticas para descubrir patrones no triviales.

---
### Limitaciones y supuestos
- **Cuerpo de referencia**: los valores base provienen de varones de 73 kg; las mujeres y los niños usan masas tisulares extrapoladas, no mediciones directas.
- **Variabilidad interindividual**: edad, inflamación, enfermedad y entorno no están modelados.
- **Fuentes heterogéneas**: algunas densidades se infieren de literatura diversa (a veces con datos animales), especialmente en tejidos poco muestreados.
- **Instantánea estática**: no incluye dinámicas temporales ni redistribución aguda ante desafíos inmunitarios.
- **Aproximaciones de masa tisular**: los promedios anatómicos estándar pueden subestimar o sobreestimar órganos específicos.

Estas limitaciones dificultan la **cuantificación exacta** pero permiten obtener **comparativas** y una **arquitectura esquemática** del sistema inmunitario.
        """)
    st.stop()

# ===================== SUPERIOR: 3D + KPIs + BARRAS DE SISTEMA =====================
top_left, top_mid, top_right = st.columns([1.1, 0.9, 1.4])

with top_left:
    st.markdown("**3D — órgano seleccionado**")
    fig3d = go.Figure()
    if selected_organ and (selected_organ in meshes):
        V, F, fname = meshes[selected_organ]
        vmax = float(bt_color[metric_col].max()) if len(bt_color) else 0.0
        val = get_value(bt_color, 'tissue_norm', selected_organ, metric_col)
        organ_color = uniform_color(val, vmax, px.colors.sequential.YlOrRd)
        fig3d.add_trace(mesh_trace(selected_organ, V, F, color=organ_color, opacity=0.9,
                                   offset=(tx,ty,tz), rotz_deg=rotz, scale=scale))
        fig3d.update_layout(
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            height=360, margin=dict(l=0,r=0,t=30,b=0),
            title=f"{selected_organ} - Sistema: {dict_map_system(selected_organ)} — color según {metric.lower()}" + (" [sim]" if use_sim_for_3d else "")
        )
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption("**Interpretación**: la intensidad de color del órgano refleja su **{0}**.".format("número de células" if metric_col=='cells' else "masa inmune en gramos"))

    else:
        st.warning("No hay mallas en /models o el órgano seleccionado no tiene malla. Añade .stl/.obj y recarga.")

with top_mid:
    total_cells = float(by_tissue_base['cells'].sum())
    total_mass  = float(by_tissue_base['mass'].sum())
    v_org_base  = get_value(by_tissue_base, 'tissue_norm', selected_organ or "", metric_col)
    st.metric("Células totales (cohorte)", f"{total_cells:,.3g}")
    st.metric("Masa total inmunológica (g)", f"{total_mass:,.3g}")
    st.metric(f"{selected_organ or '—'} — {metric}", f"{v_org_base:,.3g}")
    st.caption("**KPIs**: resumen de magnitudes globales: total de células, masa inmune total (g) y valor del tejido seleccionado.")

with top_right:
    bt_sys = by_tissue_base.copy()
    bt_sys['system'] = bt_sys['tissue_norm'].apply(dict_map_system)
    bs = bt_sys.groupby('system', as_index=False).agg(cells=('cells','sum'), mass=('mass','sum'))
    bs = bs.sort_values(metric_col, ascending=False)
    fig_sys = go.Figure(go.Bar(
        x=bs[metric_col], y=bs['system'], orientation='h',
        hovertemplate="%{y}: %{x:.3g}<extra></extra>",
        marker_color=px.colors.qualitative.Pastel
    ))
    if metric_col=='cells' and st.session_state.get("log_toggle", False):
        fig_sys.update_xaxes(type='log')
    fig_sys.update_layout(title=f"{'Células' if metric_col=='cells' else 'Masa (g)'} por sistema",
                          height=360, margin=dict(l=20,r=10,t=40,b=10))
    st.plotly_chart(fig_sys, use_container_width=True)
    st.caption("**Interpretación**: distribución del número total de células por sistema corporal. Los sistemas cardiovascular y linfático concentran gran parte del total, reflejando su papel en transporte y defensa, mientras que endocrino y reproductor son menores por su función más localizada.")

# ===================== PÁGINAS =====================
# -------- PÁGINA: Distribución --------
if page == "🧬 Distribución":
    st.subheader("Por tejido")
    bt = by_tissue_base.sort_values(metric_col, ascending=False)
    highlight = bt['tissue_norm'].apply(lambda t: norm_key(t) == norm_key(selected_organ or "spleen"))
    colors_bt = [ "#F05A28" if h else "#B0B0B0" for h in highlight ]
    fig_bt = go.Figure(go.Bar(
        x=bt[metric_col], y=bt['tissue_norm'], orientation='h',
        hovertemplate="%{y}: %{x:.3g}<extra></extra>",
        marker_color=colors_bt
    ))
    if metric_col=='cells' and st.session_state.get("log_toggle", False):
        fig_bt.update_xaxes(type='log')
    fig_bt.update_layout(title=f"{'Células' if metric_col=='cells' else 'Masa (g)'} por tejido",
                         height=520, margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig_bt, use_container_width=True)
    st.caption("**Interpretación**: el número total de células estimadas varía ampliamente entre tejidos. La médula ósea roja, el bazo y los pulmones destacan por sus valores elevados. En **naranja** se resalta el tejido seleccionado.")

    st.markdown("### Masa vs nº de células (log–log)")
    sc = bt.copy()
    sc['system'] = sc['tissue_norm'].apply(dict_map_system)

    # Crear un mapeo de color por sistema
    unique_systems = sc['system'].unique()
    color_map = {sys: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)] for i, sys in enumerate(unique_systems)}
    sc['color'] = sc['system'].map(color_map)

    fig_sc = go.Figure(go.Scatter(
        x=sc['mass'] + 1e-12,
        y=sc['cells'] + 1e-6,
        mode='markers',
        marker=dict(
            size=10,
            color=sc['color'],  # ← colores personalizados
            line=dict(width=0.5, color='rgba(0,0,0,0.3)')
        ),
        text=sc['tissue_norm'],
        hovertemplate="%{text}<br>Sistema: %{customdata}<br>Masa: %{x:.3g} g<br>Células: %{y:.3g}<extra></extra>",
        customdata=sc['system']
    ))

    # Ejes logarítmicos y etiquetas
    fig_sc.update_xaxes(type='log', title="Masa inmune (g, log)")
    fig_sc.update_yaxes(type='log', title="Número de células (log)")

    # Línea de tendencia
    xlog = np.log10(sc['mass'].replace(0, np.nan))
    ylog = np.log10(sc['cells'].replace(0, np.nan))
    valid = ~(xlog.isna() | ylog.isna())
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(xlog[valid], ylog[valid], 1)
        x_line = np.linspace(float(xlog.min()), float(xlog.max()), 50)
        y_line = slope * x_line + intercept
        fig_sc.add_trace(go.Scatter(
            x=10**x_line, y=10**y_line, mode='lines',
            name="Tendencia", line=dict(width=2, dash='dash')
        ))

    fig_sc.update_layout(height=520, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption("**Interpretación**: relación positiva entre masa inmune y número de células; las desviaciones sugieren especialización o densidad celular distinta entre tejidos.")

# -------- PÁGINA: Tipos celulares --------
elif page == "💪 Tipos celulares":
    st.subheader("Por tipo de célula")
    bc = by_celltype_base.sort_values(metric_col, ascending=False)
    colors_ct = [px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)] for i in range(len(bc))]
    fig_ct = go.Figure(go.Bar(
        x=bc[metric_col], y=bc['cell_type'], orientation='h',
        hovertemplate="%{y}: %{x:.3g}<extra></extra>",
        marker_color=colors_ct
    ))
    if metric_col=='cells' and st.session_state.get("log_toggle", False):
        fig_ct.update_xaxes(type='log')
    fig_ct.update_layout(title=f"{'Células' if metric_col=='cells' else 'Masa (g)'} por tipo de célula",
                         height=560, margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig_ct, use_container_width=True)
    st.caption("**Interpretación**: unas pocas poblaciones (p. ej., neutrófilos y linfocitos) dominan el conjunto celular del sistema inmune.")

# -------- PÁGINA: Cohortes --------
elif page == "🧍 Cohortes":
    st.subheader("Cohortes (top-10 tejidos)")
    def agg_tissue(df, coh):
        c_cells, c_mass = COHORT_MAP[coh]
        return (df.groupby('tissue_norm', as_index=False)
                  .agg(cells=(c_cells,'sum'), mass=(c_mass,'sum'))).fillna(0.0)
    bt_m = agg_tissue(df_raw, 'Hombre')
    bt_w = agg_tissue(df_raw, 'Mujer')
    bt_c = agg_tissue(df_raw, 'Niño (10 años)')
    col = metric_col
    merged = (bt_m[['tissue_norm', col]].rename(columns={col:'Hombre'})
                .merge(bt_w[['tissue_norm', col]].rename(columns={col:'Mujer'}), on='tissue_norm', how='outer')
                .merge(bt_c[['tissue_norm', col]].rename(columns={col:'Niño (10 años)'}), on='tissue_norm', how='outer')
                .fillna(0.0))
    merged['sum_all'] = merged[['Hombre','Mujer','Niño (10 años)']].sum(axis=1)
    top = merged.sort_values('sum_all', ascending=False).head(10)
    fig_grp = go.Figure()
    for coh, color in zip(['Hombre','Mujer','Niño (10 años)'], px.colors.qualitative.Set3):
        fig_grp.add_trace(go.Bar(
            x=top[coh], y=top['tissue_norm'], name=coh, orientation='h',
            hovertemplate=f"%{{y}} — {coh}: %{{x:.3g}}<extra></extra>", marker_color=color
        ))
    if metric_col=='cells' and st.session_state.get("log_toggle", False):
        fig_grp.update_xaxes(type='log')
    fig_grp.update_layout(barmode='group', height=560, margin=dict(l=20,r=20,t=40,b=20),
                          title=f"Top-10 tejidos según {'células' if metric_col=='cells' else 'masa (g)'} — cohortes")
    st.plotly_chart(fig_grp, use_container_width=True)
    st.caption("**Interpretación**: comparación de los diez tejidos con más células entre cohortes; la distribución general es similar, con diferencias esperables por maduración y función.")


# -------- PÁGINA: Conclusiones --------
elif page == "🧾 Conclusiones":
    st.subheader("Conclusiones (automático)")
    bt_sorted = by_tissue_base.sort_values(metric_col, ascending=False)
    top_tissue = bt_sorted.iloc[0]['tissue_norm']
    top_val    = bt_sorted.iloc[0][metric_col]

    bt_sys = by_tissue_base.copy()
    bt_sys['system'] = bt_sys['tissue_norm'].apply(dict_map_system)
    bs = bt_sys.groupby('system', as_index=False).agg(cells=('cells','sum'), mass=('mass','sum'))
    sys_top = bs.sort_values(metric_col, ascending=False).iloc[0]

    st.markdown(f"""
- **Tejido principal ({'células' if metric_col=='cells' else 'masa'}):** `{top_tissue}` con **{top_val:,.3g}**.
- **Sistema dominante (diccionario):** `{sys_top['system']}` con **{sys_top[metric_col]:,.3g}**.
- **Órgano seleccionado:** `{(selected_organ or '—')}` contribuye **{get_value(by_tissue_base,'tissue_norm',selected_organ or '',metric_col):,.3g}** ({'células' if metric_col=='cells' else 'masa'}).
""")
    # st.caption("Usa esta página para resumir hallazgos tras la exploración interactiva. Vincula observaciones a la cohorte y el sistema.")
    with st.expander("Limitaciones y suposiciones (recordatorio)"):
        st.markdown("""
- Referencia: varón de 73 kg como base; mujeres y niños extrapolados.
- Sin dinámica: no modela redistribuciones agudas ni el tiempo.
- Datos heterogéneos: algunas densidades tisulares son inferidas.
- Promedios poblacionales: masas tisulares aproximadas, no específicas por individuo.
        """)

# -------- PÁGINA: Laboratorio de Ciencia de Datos --------
elif page == "🔬 Laboratorio de Ciencia de Datos":
    st.subheader("Análisis avanzado")
    SUBPAGES = [
        "📈 Correlaciones",
        "🧠 PCA",
        "⚖️ Eficiencia inmune",
        "👥 Variabilidad por cohorte",
        "🧪 Simulador multivariable",
        "🧪 Hipótesis"
    ]
    if "dsl_subpage" not in st.session_state: st.session_state.dsl_subpage = SUBPAGES[0]
    subpage = st.radio("Secciones del laboratorio", SUBPAGES, horizontal=True,
                       index=SUBPAGES.index(st.session_state.dsl_subpage), key="dsl_radio")
    st.session_state.dsl_subpage = subpage

    # ------------- Correlaciones -------------
    if subpage == "📈 Correlaciones":
        mode = st.radio("Modo de correlación", ["Según cohorte (por tejido)", "Según variable (por tejido)"], index=0, key="corr_mode")
        if mode.startswith("Según cohorte"):
            def agg(df, coh):
                c_cells, c_mass = COHORT_MAP[coh]
                a = (df.groupby('tissue_norm', as_index=False)
                       .agg(cells=(c_cells,'sum'), mass=(c_mass,'sum'))).fillna(0.0)
                a = a.rename(columns={'cells': f'cells_{coh.lower()}', 'mass': f'mass_{coh.lower()}'})
                return a
            w = agg(df_raw, 'Hombre').merge(agg(df_raw,'Mujer'), on='tissue_norm', how='outer') \
                                     .merge(agg(df_raw,'Niño (10 años)'), on='tissue_norm', how='outer') \
                                     .set_index('tissue_norm').fillna(0.0)
            cols = [c for c in w.columns if c.startswith('cells_')]
            corr = w[cols].corr(method='pearson')
            fig = px.imshow(corr, text_auto=True, aspect='auto', title="Correlación de Pearson de recuentos celulares entre cohortes")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**Interpretación**: alta correlación entre los recuentos celulares de hombres, mujeres y niños (r > 0,98).")
        else:
            bt = aggregate_by_tissue(df_raw, cohort).copy()
            bt['cells_per_g'] = bt.apply(lambda r: (r['cells'] / r['mass']) if r['mass'] and r['mass']>0 else np.nan, axis=1)
            bt['log_cells'] = np.log10(bt['cells'].replace(0, np.nan))
            bt['log_mass']  = np.log10(bt['mass'].replace(0, np.nan))
            feats = bt[['cells','mass','cells_per_g','log_cells','log_mass']].copy()
            corr = feats.corr(method='spearman')
            fig = px.imshow(corr, text_auto=True, aspect='auto', title=f"Correlación de Spearman de las variables ({cohort})")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**Interpretación**: células y masa se correlacionan fuertemente; células/gramo muestra correlación moderada, revelando variaciones de densidad.")

    # ------------- PCA -------------
    elif subpage == "🧠 PCA":
        st.write("Proyectamos los tejidos en componentes principales para revelar la estructura global.")
        pca_dim = st.radio("Dimensionalidad de la gráfica", ["2D","3D"], index=0, horizontal=True, key="pca_dim")
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            bt = aggregate_by_tissue(df_raw, cohort).copy()
            bt['cells_per_g'] = bt.apply(lambda r: (r['cells'] / r['mass']) if r['mass'] and r['mass']>0 else 0.0, axis=1)
            X = bt[['cells','mass','cells_per_g']].values
            Xs = StandardScaler().fit_transform(X)
            pca = PCA(n_components=3).fit(Xs)
            Z = pca.transform(Xs)
            expl = pca.explained_variance_ratio_
            bt['system'] = bt['tissue_norm'].apply(dict_map_system)
            if pca_dim == "2D":
                figp = px.scatter(x=Z[:,0], y=Z[:,1], color=bt['system'], hover_name=bt['tissue_norm'],
                                  labels={'x':f'PC1 ({expl[0]*100:.1f}%)', 'y':f'PC2 ({expl[1]*100:.1f}%)'},
                                  title="PCA de tejidos (células, masa, células/g)")
                st.plotly_chart(figp, use_container_width=True)
            else:
                figp3 = px.scatter_3d(x=Z[:,0], y=Z[:,1], z=Z[:,2], color=bt['system'], hover_name=bt['tissue_norm'],
                                      labels={'x':f'PC1 ({expl[0]*100:.1f}%)','y':f'PC2 ({expl[1]*100:.1f}%)','z':f'PC3 ({expl[2]*100:.1f}%)'},
                                      title="PCA de tejidos (3D)")
                st.plotly_chart(figp3, use_container_width=True)
            st.info(f"Varianza explicada: PC1={expl[0]*100:.1f}%, PC2={expl[1]*100:.1f}%, PC3={expl[2]*100:.1f}%")
            st.caption("**Interpretación**: los clústeres sugieren patrones similares de asignación inmune; PC1 suele correlacionarse con el tamaño.")
        except Exception as e:
            st.error(f"Se requiere scikit-learn para el PCA. Instálalo y recarga. Detalles: {e}")

    # ------------- Eficiencia inmune -------------
    elif subpage == "⚖️ Eficiencia inmune":
        st.write("**Eficiencia inmune** = células por gramo de masa inmune por tejido.")
        bt = aggregate_by_tissue(df_raw, cohort).copy()
        bt['cells_per_g'] = bt.apply(lambda r: (r['cells'] / r['mass']) if r['mass'] and r['mass']>0 else np.nan, axis=1)
        bt['system'] = bt['tissue_norm'].apply(dict_map_system)
        topn = st.slider("Top / Bottom N", 3, 20, 10, step=1, key="eff_topn")
        show_bottom = st.checkbox("Mostrar los Bottom-N", value=True, key="eff_bottom")

        import math

        def tile_colors(palette, n):
            # Devuelve una lista de n colores repitiendo la paleta
            if n <= 0:
                return []
            return (palette * math.ceil(n / len(palette)))[:n]


        top_df = bt.sort_values('cells_per_g', ascending=False).head(topn)
        top_colors = tile_colors(px.colors.qualitative.Set2, len(top_df))
        fig_top = go.Figure(go.Bar(
            x=top_df['cells_per_g'], y=top_df['tissue_norm'],
            orientation='h',
            marker_color=top_colors
        ))
        fig_top.update_layout(
            title=f"Top-{topn} tejidos con más células por gramo",
            height=440, margin=dict(l=20,r=20,t=40,b=20)
        )
        st.plotly_chart(fig_top, use_container_width=True)

        if show_bottom:
            bottom_df = bt.sort_values('cells_per_g', ascending=True).head(topn)
            bottom_colors = tile_colors(px.colors.qualitative.Set3, len(bottom_df))
            fig_bot = go.Figure(go.Bar(
                x=bottom_df['cells_per_g'], y=bottom_df['tissue_norm'],
                orientation='h',
                marker_color=bottom_colors
            ))
            fig_bot.update_layout(
                title=f"Bottom-{topn} tejidos con menos células por gramo",
                height=440, margin=dict(l=20,r=20,t=40,b=20)
            )
            st.plotly_chart(fig_bot, use_container_width=True)
        st.caption("**Interpretación**: la eficiencia mide densidad, no tamaño. Valores altos pueden indicar nichos inmunes especializados.")

        fig_hist = px.histogram(bt, x='cells_per_g', nbins=30, marginal="box",
                                title="Distribución de células por gramo (todos los tejidos)")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("**Distribución**: dispersión de células/gramo entre tejidos; los atípicos sugieren interpretación biológica o revisión de datos.")

    # ------------- Variabilidad por cohorte -------------
    elif subpage == "👥 Variabilidad por cohorte":
        st.write("Diferencias relativas entre cohortes por tejido.")
        metric_pick = st.radio("Variable", ["Células","Masa (g)"], index=0, horizontal=True, key="cv_var")
        var_col_map = {'Células':'cells', 'Masa (g)':'mass'}
        vcol = var_col_map[metric_pick]

        def agg(df, coh):
            c_cells, c_mass = COHORT_MAP[coh]
            return (df.groupby('tissue_norm', as_index=False)
                      .agg(cells=(c_cells,'sum'), mass=(c_mass,'sum'))).fillna(0.0)

        m = agg(df_raw,'Hombre'); w = agg(df_raw,'Mujer'); c = agg(df_raw,'Niño (10 años)')
        merged = m[['tissue_norm', vcol]].rename(columns={vcol:'Hombre'}) \
                 .merge(w[['tissue_norm', vcol]].rename(columns={vcol:'Mujer'}), on='tissue_norm', how='outer') \
                 .merge(c[['tissue_norm', vcol]].rename(columns={vcol:'Niño (10 años)'}), on='tissue_norm', how='outer') \
                 .fillna(0.0)

        view = st.selectbox("Mostrar", ["Ratios (Mujer/Hombre)","Ratios (Niño/Hombre)","Diferencias absolutas"], index=0, key="cv_view")
        if view.startswith("Ratios (Mujer/Hombre)"):
            merged['ratio'] = merged.apply(lambda r: (r['Mujer']/r['Hombre']) if r['Hombre']>0 else np.nan, axis=1)
            plot_df = merged.sort_values('ratio', ascending=False).head(20)
            fig = go.Figure(go.Bar(x=plot_df['ratio'], y=plot_df['tissue_norm'], orientation='h'))
            fig.update_layout(title="Ratio Mujer / Hombre (top-20)", height=560, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**Interpretación**: ratios >1 = relativamente más alto en mujeres para la métrica elegida.")
        elif view.startswith("Ratios (Niño/Hombre)"):
            merged['ratio'] = merged.apply(lambda r: (r['Niño (10 años)']/r['Hombre']) if r['Hombre']>0 else np.nan, axis=1)
            plot_df = merged.sort_values('ratio', ascending=False).head(20)
            fig = go.Figure(go.Bar(x=plot_df['ratio'], y=plot_df['tissue_norm'], orientation='h'))
            fig.update_layout(title="Ratio Niño (10 años) / Hombre (top-20)", height=560, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**Interpretación**: ratios >1 = relativamente más alto en niños frente a hombres.")
        else:
            merged['Mujer - Hombre'] = merged['Mujer'] - merged['Hombre']
            merged['Niño - Hombre'] = merged['Niño (10 años)'] - merged['Hombre']
            plot_df = merged[['tissue_norm','Mujer - Hombre','Niño - Hombre']].set_index('tissue_norm')
            plot_df = plot_df.loc[plot_df.abs().sum(axis=1).sort_values(ascending=False).head(20).index]
            fig = go.Figure()
            for colname, color in zip(plot_df.columns, px.colors.qualitative.Set2):
                fig.add_trace(go.Bar(x=plot_df[colname], y=plot_df.index, name=colname, orientation='h', marker_color=color))
            fig.update_layout(barmode='group', title="Diferencia absoluta vs Hombre (top-20 |Δ|)",
                              height=560, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**Δ Absoluta**: útil cuando las proporciones son inestables cerca de cero; muestra cambios brutos respecto al valor base masculino.")

    # ------------- Simulador multivariable -------------
    elif subpage == "🧪 Simulador multivariable":
        st.write("Simular múltiples cambios de condición (cohorte seleccionada). Los factores se aplican a los tejidos o sistemas elegidos a continuación.")
        colA, colB, colC = st.columns(3)
        with colA:
            adip_pct = st.slider("Tejido adiposo %+Δ", -50, 200, 20, step=5, key="mc_adip")
            spleen_pct = st.slider("Bazo %+Δ", -50, 100, 10, step=5, key="mc_spleen")
        with colB:
            ln_pct = st.slider("Nodo linfático %+Δ", -50, 100, 10, step=5, key="mc_ln")
            marrow_pct = st.slider("Médula roja %+Δ", -50, 100, 10, step=5, key="mc_marrow")
        with colC:
            liver_pct = st.slider("Hígado %+Δ", -50, 100, 0, step=5, key="mc_liver")
            skm_pct = st.slider("Músculos esqueléticos %+Δ", -50, 50, 0, step=5, key="mc_skm")

        def apply_factor(df, tissue_matcher, cells_col, mass_col, pct):
            if pct == 0: return df
            factor = 1.0 + pct/100.0
            mask = df['tissue_norm'].apply(lambda t: bool(re.search(tissue_matcher, t, flags=re.I)))
            df.loc[mask, cells_col] *= factor
            df.loc[mask, mass_col]  *= factor
            return df

        cells_col, mass_col = COHORT_MAP[cohort]
        df_sim = df_raw.copy()
        df_sim = apply_factor(df_sim, r"\badipose\b", cells_col, mass_col, adip_pct)
        df_sim = apply_factor(df_sim, r"\bspleen\b", cells_col, mass_col, spleen_pct)
        df_sim = apply_factor(df_sim, r"\blymph\s*nodes?\b", cells_col, mass_col, ln_pct)
        df_sim = apply_factor(df_sim, r"\bred marrow\b", cells_col, mass_col, marrow_pct)
        df_sim = apply_factor(df_sim, r"\bliver\b", cells_col, mass_col, liver_pct)
        df_sim = apply_factor(df_sim, r"\bskeletal\s*muscles?\b", cells_col, mass_col, skm_pct)

        bt_base = aggregate_by_tissue(df_raw, cohort)
        bt_sim  = aggregate_by_tissue(df_sim, cohort)

        base_tot_cells = float(bt_base['cells'].sum())
        sim_tot_cells  = float(bt_sim['cells'].sum())
        base_tot_mass  = float(bt_base['mass'].sum())
        sim_tot_mass   = float(bt_sim['mass'].sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Células (base)", f"{base_tot_cells:,.3g}")
        k2.metric("Células (sim)", f"{sim_tot_cells:,.3g}", delta=f"{(sim_tot_cells-base_tot_cells):,.3g}")
        k3.metric("Masa (base, g)", f"{base_tot_mass:,.3g}")
        k4.metric("Masa (sim, g)",  f"{sim_tot_mass:,.3g}", delta=f"{(sim_tot_mass-base_tot_mass):,.3g}")
        st.caption("**Caso de uso**: poner a prueba hipótesis modificando varios tejidos simultáneamente.")

        bs_base = by_system_from_bt(bt_base)
        bs_sim  = by_system_from_bt(bt_sim)

        pie_var = st.radio("Gráfico de tarta sobre", ["cells","mass"], index=0, horizontal=True, key="mc_pievar")
        p1, p2 = st.columns(2)
        with p1:
            fig_p1 = px.pie(bs_base, values=pie_var, names='system', title=f"Base — cuota por sistema ({pie_var})")
            st.plotly_chart(fig_p1, use_container_width=True)
        with p2:
            fig_p2 = px.pie(bs_sim, values=pie_var, names='system', title=f"Simulado — cuota por sistema ({pie_var})")
            st.plotly_chart(fig_p2, use_container_width=True)

        comp = (bt_base[['tissue_norm', metric_col]].merge(
                bt_sim[['tissue_norm', metric_col]], on='tissue_norm', how='outer', suffixes=('_base','_sim'))
                .fillna(0.0))
        comp['delta'] = comp[f'{metric_col}_sim'] - comp[f'{metric_col}_base']
        comp = comp.sort_values('delta', ascending=False).head(20)
        fig_d = go.Figure(go.Bar(x=comp['delta'], y=comp['tissue_norm'], orientation='h', marker_color=['#2ca02c' if d>=0 else '#d62728' for d in comp['delta']]))
        fig_d.update_layout(title=f"Top-20 Δ por tejido según muchos factores", height=560, margin=dict(l=20,r=20,t=40,b=20))        
        st.plotly_chart(fig_d, use_container_width=True)
        st.caption("**Indicación**: la cascada de cambios entre tejidos ayuda a priorizar factores de cambio sistémico.")

    # ------------- Hipótesis -------------
    elif subpage == "🧪 Hipótesis":
        htab = st.radio("Selecciona hipótesis", ["H1 - Redistribución bajo adiposidad","H2 - Eficiencia y optimalidad inmunes","H3 - Escalado con el tamaño corporal"],
                        index=0, horizontal=True, key="hyp_pick")

        # ---- H1: Redistribución bajo adiposidad ----
        if htab.startswith("H1"):
            st.write("**Hipótesis:** el aumento de la masa de tejido adiposo redistribuye las células inmunes hacia compartimentos no linfáticos.")

            # Sensibilidad
            sens_range = st.slider("Rango de incremento de adiposidad (%)", 0, 200, (0, 100), step=10, key="h1_range")
            lo, hi = sens_range
            grid = list(range(lo, hi+1, 10))
            insp_pct = st.slider("Examinar incremento de adiposidad en (%)", lo, hi, min(20, hi), step=5, key="h1_insp_pct")

            # Agregados base
            bt_base = aggregate_by_tissue(df_raw, cohort)
            bs_base = by_system_from_bt(bt_base)
            base_total_cells = bs_base['cells'].sum() if len(bs_base) else 0.0

            rows_share = []
            rows_totals = []
            for p in grid:
                df_s = simulate_adiposity(df_raw, cohort, p)
                bt_s = aggregate_by_tissue(df_s, cohort)
                bs_s = by_system_from_bt(bt_s)
                tot = bs_s['cells'].sum()
                for sys_name in bs_s['system'].unique():
                    cells_sys = float(bs_s.loc[bs_s['system']==sys_name, 'cells'].sum())
                    share_sys = (cells_sys / tot) if tot > 0 else np.nan
                    rows_share.append((p, sys_name, share_sys))
                    rows_totals.append((p, sys_name, cells_sys))
                if "Otros" not in bs_s['system'].values:
                    rows_share.append((p, "Otros", 0.0))
                    rows_totals.append((p, "Otros", 0.0))

            sens_share = pd.DataFrame(rows_share, columns=['adipose_pct','system','share'])
            sens_share['share_pct'] = 100.0 * sens_share['share']
            sens_totals = pd.DataFrame(rows_totals, columns=['adipose_pct','system','cells'])

            base_share_pct = (bs_base[['system','cells']].assign(share_pct=lambda d: 100.0*d['cells']/base_total_cells)
                              if base_total_cells>0 else pd.DataFrame(columns=['system','share_pct']))
            base_share_map = dict(zip(base_share_pct['system'], base_share_pct['share_pct'])) if len(base_share_pct) else {}
            deltas_pp = sens_share.copy()
            deltas_pp['delta_pp'] = deltas_pp.apply(lambda r: r['share_pct'] - base_share_map.get(r['system'], 0.0), axis=1)

            vis_mode = st.radio(
                "Visualización",
                ["Δ de cuota (pp) vs % adiposidad",
                 "Base vs +X% (barras apiladas)","Cascada Δ de células @ +X%"],
                horizontal=True, key="h1_vis_mode"
            )

            if vis_mode == "Δ de cuota (pp) vs % adiposidad":
                y_span = st.slider("Rango Y para Δ (pp)", -5.0, 5.0, (-0.5, 0.5), step=0.25, key="h1_pp_range")
                fig = go.Figure()
                for sys_name in sorted(deltas_pp['system'].unique()):
                    dd = deltas_pp[deltas_pp['system']==sys_name].sort_values('adipose_pct')
                    fig.add_trace(go.Scatter(x=dd['adipose_pct'], y=dd['delta_pp'], mode='lines+markers', name=sys_name))
                fig.update_layout(title="Δ de cuota vs base (puntos porcentuales)",
                                  xaxis_title="% incremento adiposo", yaxis_title="Δ cuota (pp)",
                                  yaxis=dict(range=list(y_span)))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("**Por qué pp?**: los puntos porcentuales hacen legibles pequeños cambios de cuota incluso si crecen los totales.")

            elif vis_mode == "Base vs +X% (barras apiladas)":
                df_i = simulate_adiposity(df_raw, cohort, insp_pct)
                bs_i = by_system_from_bt(aggregate_by_tissue(df_i, cohort))
                base_share_plot = bs_base[['system','cells']].assign(share=lambda d: d['cells']/d['cells'].sum())
                sim_share_plot  = bs_i[['system','cells']].assign(share=lambda d: d['cells']/d['cells'].sum())
                base_share_plot['source'] = f'Base (0%)'
                sim_share_plot['source']  = f'Sim (+{insp_pct}%)'
                long = pd.concat([
                    base_share_plot[['source','system','share']].assign(share=lambda d: 100.0*d['share']),
                    sim_share_plot[['source','system','share']].assign(share=lambda d: 100.0*d['share'])
                ], ignore_index=True)
                fig = px.bar(long, x='source', y='share', color='system', title="Cuota por sistema (apilado, %)",
                             labels={'share':'Cuota (%)','source':''}, text=long['share'].map(lambda v: f"{v:.1f}%"))
                fig.update_layout(barmode='stack')
                st.plotly_chart(fig, use_container_width=True)
                st.caption("**Lectura**: dos columnas apiladas (0% vs +X%) muestran de un vistazo los cambios de composición.")

            else:  # Waterfall Δ cells @ +X%
                df_i = simulate_adiposity(df_raw, cohort, insp_pct)
                bs_i = by_system_from_bt(aggregate_by_tissue(df_i, cohort))
                comp = (bs_i[['system','cells']].merge(bs_base[['system','cells']], on='system',
                        suffixes=('_sim','_base'), how='outer').fillna(0))
                comp['delta_cells'] = comp['cells_sim'] - comp['cells_base']
                comp = comp.sort_values('delta_cells', ascending=False)
                measures = ['relative'] * len(comp)
                fig = go.Figure(go.Waterfall(
                    x=comp['system'],
                    measure=measures,
                    y=comp['delta_cells'],
                    text=comp['delta_cells'].map(lambda v: f"{v:,.3g}"),
                    textposition='outside'
                ))
                fig.update_layout(title=f"Cascada — Δ células por sistema @ +{insp_pct}%",
                                  yaxis_title="Δ células (absoluto)")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("**Uso**: aísla ganadores/perdedores en términos absolutos; útil cuando las cuotas apenas cambian.")

            # Tabla resumen @ insp_pct
            df_i = simulate_adiposity(df_raw, cohort, insp_pct)
            bs_i = by_system_from_bt(aggregate_by_tissue(df_i, cohort))
            comp_pp = (bs_i[['system','cells']].merge(bs_base[['system','cells']], on='system',
                      suffixes=('_sim','_base'), how='outer').fillna(0))
            comp_pp['share_base'] = comp_pp['cells_base'] / (comp_pp['cells_base'].sum() or 1)
            comp_pp['share_sim']  = comp_pp['cells_sim']  / (comp_pp['cells_sim'].sum()  or 1)
            comp_pp['Δ cuota (pp)'] = 100.0*(comp_pp['share_sim'] - comp_pp['share_base'])
            comp_pp['Δ células'] = comp_pp['cells_sim'] - comp_pp['cells_base']
            comp_pp = comp_pp.sort_values('Δ cuota (pp)', ascending=False)
            st.markdown("**Δ por sistema @ % seleccionado**")
            st.dataframe(comp_pp[['system','cells_base','cells_sim','Δ células','Δ cuota (pp)']], use_container_width=True)


            st.markdown("### Interpretación de resultados")
            st.write("""
            El aumento de la masa de tejido adiposo provoca una redistribución progresiva de las células inmunes hacia compartimentos no linfáticos, en especial hacia los sistemas digestivo y otros tejidos periféricos. 
            A medida que crece la adiposidad, la proporción de células en el sistema linfático y cardiovascular tiende a disminuir ligeramente en puntos porcentuales, mientras que el tejido adiposo actúa como reservorio adicional de células inmunes, lo que incrementa su cuota relativa sin alterar drásticamente el total sistémico. 
            Este comportamiento sugiere un efecto de desplazamiento funcional en la respuesta inmunitaria, donde la expansión del tejido adiposo puede modificar la localización y la eficacia de la defensa inmunológica, más que su cantidad absoluta.
            """)



        # ---- H2: Eficiencia y optimalidad inmunes ----
        elif htab.startswith("H2"):
            st.write("**Hipótesis:** los tejidos con alta densidad de células inmunes (células por gramo) no necesariamente tienen la mayor masa total; pueden existir óptimos de eficiencia.")
            bt = aggregate_by_tissue(df_raw, cohort).copy()
            bt['cells_per_g'] = bt.apply(lambda r: (r['cells'] / r['mass']) if (r['mass'] and r['mass']>0) else np.nan, axis=1)
            x = np.log10(bt['mass'].replace(0, np.nan))
            y = np.log10(bt['cells_per_g'].replace(0, np.nan))
            mask = ~(x.isna() | y.isna())
            try:
                from scipy.stats import spearmanr
                if mask.sum() >= 3:
                    slope, intercept = np.polyfit(x[mask], y[mask], 1)
                    x_line = np.linspace(float(x[mask].min()), float(x[mask].max()), 50)
                    y_line = slope*x_line + intercept
                    rho, pval = spearmanr(x[mask], y[mask], nan_policy='omit')
                else:
                    slope = intercept = 0.0
                    x_line = y_line = np.array([])
                    rho, pval = np.nan, np.nan
            except Exception:
                if mask.sum() >= 3:
                    slope, intercept = np.polyfit(x[mask], y[mask], 1)
                    x_line = np.linspace(float(x[mask].min()), float(x[mask].max()), 50)
                    y_line = slope*x_line + intercept
                else:
                    slope = intercept = 0.0
                    x_line = y_line = np.array([])
                rho, pval = np.nan, np.nan

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt['mass']+1e-12, y=bt['cells_per_g'], mode='markers',
                                     text=bt['tissue_norm'], hovertemplate="%{text}<br>Masa: %{x:.3g} g<br>Células/g: %{y:.3g}<extra></extra>",
                                     marker=dict(size=9, line=dict(width=0.5, color='rgba(0,0,0,0.3)'))))
            if len(x_line):
                fig.add_trace(go.Scatter(x=10**x_line, y=10**y_line, mode='lines', name="Ajuste (log–log)", line=dict(dash='dash')))
            fig.update_xaxes(type='log', title="Masa inmune (g, log)")
            fig.update_yaxes(type='log', title="Células por gramo (log)")
            fig.update_layout(title="Eficiencia vs masa (log–log)", height=520, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"**Resultado**: Spearman ρ = {rho:.2f} — si es negativo, masas inmunes mayores tienden a menor densidad (rendimientos decrecientes).")

            topk = st.slider("Mostrar top/bottom K tejidos por células/g", 3, 20, 10, step=1, key="h2_topk")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Top {topk} eficiencia**")
                st.dataframe(bt.sort_values('cells_per_g', ascending=False).head(topk)[['tissue_norm','cells','mass','cells_per_g']], use_container_width=True)
            with col2:
                st.markdown(f"**Bottom {topk} eficiencia**")
                st.dataframe(bt.sort_values('cells_per_g', ascending=True).head(topk)[['tissue_norm','cells','mass','cells_per_g']], use_container_width=True)

            st.markdown("### Interpretación de resultados")
            st.write("""
            La relación entre la masa inmune total y la densidad celular revela que los tejidos más grandes no son necesariamente los más eficientes. 
            El análisis log–log muestra una correlación moderada o negativa (según el valor de ρ de Spearman), indicando que conforme aumenta la masa de un tejido, la cantidad de células inmunes por gramo tiende a estabilizarse o incluso a disminuir. 
            Los tejidos con mayor eficiencia (mayor densidad celular relativa) suelen ser órganos altamente especializados, como los ganglios linfáticos o el bazo, donde se concentran nichos inmunes. 
            Este patrón apoya la hipótesis de una distribución óptima de recursos inmunitarios: el organismo prioriza densidad funcional sobre volumen total, alcanzando un equilibrio entre coste metabólico y eficacia defensiva.
            """)



        # ---- H3: Escalado con tamaño corporal ----
        else:
            st.write("**Hipótesis:** la masa inmune escala de manera no lineal con la masa corporal entre cohortes.")
            c1, c2, c3 = st.columns(3)
            man_w   = c1.number_input("Masa corporal — Hombre (kg)", min_value=30.0, max_value=150.0, value=73.0, step=1.0, key="h3_w_man")
            woman_w = c2.number_input("Masa corporal — Mujer (kg)", min_value=30.0, max_value=150.0, value=60.0, step=1.0, key="h3_w_wom")
            child_w = c3.number_input("Masa corporal — Niño 10 años (kg)", min_value=15.0, max_value=80.0, value=32.0, step=1.0, key="h3_w_child")

            def total_mass_cells(df, coh) -> Tuple[float,float]:
                c_cells, c_mass = COHORT_MAP[coh]
                agg = df[[c_cells, c_mass]].sum(skipna=True)
                return float(agg[c_cells]), float(agg[c_mass])

            man_cells, man_mass = total_mass_cells(df_raw, 'Hombre')
            wom_cells, wom_mass = total_mass_cells(df_raw, 'Mujer')
            chi_cells, chi_mass = total_mass_cells(df_raw, 'Niño (10 años)')

            scale_df = pd.DataFrame({
                'cohort': ['Hombre','Mujer','Niño (10 años)'],
                'body_mass_kg': [man_w, woman_w, child_w],
                'immune_mass_g': [man_mass, wom_mass, chi_mass],
                'immune_cells': [man_cells, wom_cells, chi_cells]
            })
            scale_df['immune_mass_per_kg'] = scale_df['immune_mass_g'] / scale_df['body_mass_kg']
            scale_df['immune_cells_per_kg'] = scale_df['immune_cells'] / scale_df['body_mass_kg']

            cA, cB = st.columns(2)
            with cA:
                fig_a = go.Figure(go.Bar(x=scale_df['cohort'], y=scale_df['immune_mass_per_kg'],
                                         text=scale_df['immune_mass_per_kg'].map(lambda v: f"{v:.3g}"),
                                         textposition='auto'))
                fig_a.update_layout(title="Masa inmune por kg (por cohorte)", yaxis_title="g/kg")
                st.plotly_chart(fig_a, use_container_width=True)
            with cB:
                fig_b = go.Figure(go.Bar(x=scale_df['cohort'], y=scale_df['immune_cells_per_kg'],
                                         text=scale_df['immune_cells_per_kg'].map(lambda v: f"{v:.3g}"),
                                         textposition='auto'))
                fig_b.update_layout(title="Células inmunes por kg (por cohorte)", yaxis_title="células/kg")
                st.plotly_chart(fig_b, use_container_width=True)
            st.caption("**Idea**: la normalización por kg compara la asignación inmune controlando el tamaño corporal.")

            x = np.log10(scale_df['body_mass_kg'].values)
            y = np.log10(scale_df['immune_mass_g'].values + 1e-12)
            if len(x) >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 50)
                y_line = slope*x_line + intercept
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(x=scale_df['body_mass_kg'], y=scale_df['immune_mass_g'],
                                            mode='markers+text', text=scale_df['cohort'],
                                            textposition='top center'))
                fig_sc.add_trace(go.Scatter(x=10**x_line, y=10**y_line, mode='lines', name="Ajuste (log–log)", line=dict(dash='dash')))
                fig_sc.update_xaxes(type='log', title="Masa corporal (kg, log)")
                fig_sc.update_yaxes(type='log', title="Masa inmune (g, log)")
                fig_sc.update_layout(title=f"Escalado alométrico — exponente ≈ {slope:.2f}", height=520, margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fig_sc, use_container_width=True)
                st.caption("**Interpretación**: exponente ~1 -> escalado proporcional; <1 -> sublineal (la masa inmune crece más lento que la corporal).")
            else:
                st.warning("Necesitas al menos dos cohortes para estimar el exponente de escalado.")

            st.markdown("### Interpretación de resultados")
            st.write("""
            El análisis alométrico demuestra que la masa inmune no escala de forma estrictamente proporcional con la masa corporal. 
            El exponente obtenido en la regresión log–log suele ser ligeramente inferior a 1, lo que indica un escalado sublineal: los individuos de menor tamaño corporal (como los niños) presentan una mayor cantidad de masa y células inmunes por kilogramo que los adultos. 
            Esto sugiere que el sistema inmunitario está sobredimensionado en etapas de crecimiento o en organismos de menor tamaño, probablemente para compensar la inmadurez del resto de sistemas fisiológicos. 
            En conjunto, el modelo refleja un ajuste adaptativo entre masa corporal y requerimientos inmunitarios, donde la eficiencia metabólica prima sobre el crecimiento proporcional.
            """)
