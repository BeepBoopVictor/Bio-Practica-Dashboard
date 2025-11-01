# utils/meshloader.py
from pathlib import Path
import numpy as np
import trimesh

def load_mesh(path: Path):
    """
    Carga una malla OBJ/STL y devuelve (V, F) como arrays numpy:
      V: shape (N,3) vértices
      F: shape (M,3) caras (índices enteros a V)
    """
    mesh = trimesh.load(path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())  # colapsa escena a una sola malla
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    return V, F

def normalize_vertices(V: np.ndarray, target_diag: float = 2.0):
    """
    Centra y escala los vértices para que la diagonal del bounding box ≈ target_diag.
    Esto hace que distintos órganos tengan tamaños “comparables” en la escena.
    """
    mins = V.min(axis=0); maxs = V.max(axis=0)
    center = (mins + maxs) / 2.0
    V0 = V - center
    diag = np.linalg.norm(maxs - mins)
    if diag > 0:
        scale = target_diag / diag
        V0 *= scale
    else:
        scale = 1.0
    return V0, scale, center

def load_and_normalize(path: Path, target_diag: float = 2.0):
    V, F = load_mesh(path)
    Vn, scale, center = normalize_vertices(V, target_diag=target_diag)
    return Vn, F