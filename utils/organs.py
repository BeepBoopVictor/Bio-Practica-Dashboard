# utils/organs.py (añade o reemplaza lo necesario)
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math
import numpy as np
import plotly.graph_objects as go

@dataclass
class OrganState:
    name: str
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    color: str = "lightgray"
    opacity: float = 0.55
    visible: bool = True
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    rz_deg: float = 0.0
    scale: float = 1.0
    # malla real (opcional)
    V: Optional[np.ndarray] = None   # (N,3)
    F: Optional[np.ndarray] = None   # (M,3)
    # tamaño de caja fallback para cuando no hay V/F:
    box_size: Tuple[float, float, float] = (1.0, 0.6, 0.6)

def default_organs() -> Dict[str, OrganState]:
    return {
        "Spleen":      OrganState("Spleen",      center=(-2.0,  0.0, 0.0), color="royalblue",    box_size=(1.2,0.8,0.6)),
        "Liver":       OrganState("Liver",       center=( 0.0, -2.0, 0.0), color="seagreen",     box_size=(2.0,1.0,0.6)),
        "Marrow":      OrganState("Marrow",      center=( 2.0,  0.0, 0.0), color="darkorange",   box_size=(1.5,0.6,0.6)),
        "Lymph nodes": OrganState("Lymph nodes", center=( 0.0,  2.0, 0.0), color="mediumpurple", box_size=(1.4,0.8,0.6)),
        "Adipose":     OrganState("Adipose",     center=( 0.0,  0.0, 0.0), color="salmon",       box_size=(1.6,1.2,0.6)),
    }

# ---- Transformaciones geométricas ----

def _rotz(points: np.ndarray, angle_deg: float, about: Tuple[float,float,float]) -> np.ndarray:
    if angle_deg == 0:
        return points
    ang = math.radians(angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    shift = points - np.array(about)
    x = shift[:,0]*ca - shift[:,1]*sa
    y = shift[:,0]*sa + shift[:,1]*ca
    z = shift[:,2]
    return np.stack([x,y,z], axis=1) + np.array(about)

def _apply_transform(V: np.ndarray, organ: OrganState) -> np.ndarray:
    # escala relativa
    Vt = V * organ.scale
    # rotación alrededor del centro “anatómico” del órgano
    Vt = _rotz(Vt, organ.rz_deg, about=(0.0,0.0,0.0))
    # traslada al centro anatómico global + sliders
    Vt = Vt + np.array(organ.center) + np.array([organ.tx, organ.ty, organ.tz])
    return Vt

def _box_vertices(center, size):
    cx, cy, cz = center
    sx, sy, sz = size
    corners = np.array([
        [-sx, -sy, -sz], [ sx, -sy, -sz], [ sx,  sy, -sz], [-sx,  sy, -sz],
        [-sx, -sy,  sz], [ sx, -sy,  sz], [ sx,  sy,  sz], [-sx,  sy,  sz],
    ], dtype=float)
    return corners + np.array(center)

_TRI_I = np.array([0,0,0,1,1,2,4,5,6,2,3,7])
_TRI_J = np.array([1,2,3,5,6,3,5,6,7,6,7,4])
_TRI_K = np.array([2,3,0,6,3,0,6,7,4,7,4,0])

def organ_trace(organ: OrganState) -> go.Mesh3d:
    """Crea el Mesh3d a partir de una malla real (si existe) o una caja fallback."""
    if organ.V is not None and organ.F is not None:
        Vt = _apply_transform(organ.V, organ)
        x,y,z = Vt[:,0], Vt[:,1], Vt[:,2]
        i,j,k = organ.F[:,0], organ.F[:,1], organ.F[:,2]
        return go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            name=organ.name, color=organ.color, opacity=organ.opacity,
            visible=True if organ.visible else "legendonly", flatshading=True
        )
    else:
        # caja de fallback
        pts = _box_vertices(organ.center, tuple(s*organ.scale for s in organ.box_size))
        pts = _rotz(pts, organ.rz_deg, about=organ.center)
        pts = pts + np.array([organ.tx, organ.ty, organ.tz])
        x,y,z = pts[:,0], pts[:,1], pts[:,2]
        return go.Mesh3d(
            x=x, y=y, z=z, i=_TRI_I, j=_TRI_J, k=_TRI_K,
            name=organ.name, color=organ.color, opacity=organ.opacity,
            visible=True if organ.visible else "legendonly", flatshading=True
        )

def build_scene(organs: Dict[str, OrganState], title="Órganos (3D)") -> go.Figure:
    fig = go.Figure()
    for organ in organs.values():
        fig.add_trace(organ_trace(organ))
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        legend=dict(itemsizing="constant"),
        height=540, margin=dict(l=0,r=0,t=30,b=0), title=title
    )
    return fig

def color_by_value(organs: Dict[str, OrganState], values: Dict[str, float], palette):
    import numpy as np
    keys = list(organs.keys())
    arr = np.array([max(0.0, float(values.get(k,0.0))) for k in keys], dtype=float)
    if arr.max() > 0: arr = arr / arr.max()
    n = len(palette) - 1
    for k, v in zip(keys, arr):
        organs[k].color = palette[int(round(v*n))]