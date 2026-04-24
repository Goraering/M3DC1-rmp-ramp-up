from __future__ import annotations

import inspect

import numpy as np
from matplotlib.figure import Figure

from .read_field import read_field
from .read_lcfs import read_lcfs

_READ_FIELD_KW = set(inspect.signature(read_field).parameters.keys())
_READ_LCFS_KW = set(inspect.signature(read_lcfs).parameters.keys())


def get_lcfs(psi=None, x=None, z=None, *, psival=None, axis=None, filename="C1.h5", points=200, slice=0, **kwargs):
    """
    Return an approximate LCFS contour path as shape (2, N).
    """
    field_kwargs = {k: v for k, v in kwargs.items() if k in _READ_FIELD_KW}
    lcfs_kwargs = {k: v for k, v in kwargs.items() if k in _READ_LCFS_KW}
    if psi is None or x is None or z is None:
        pmeta = read_field(
            "psi",
            filename=filename,
            timeslices=slice,
            points=points,
            equilibrium=True,
            return_meta=True,
            **field_kwargs,
        )
        psi2d = np.asarray(pmeta.data)[0, :, :] if np.asarray(pmeta.data).ndim == 3 else np.asarray(pmeta.data)
        xv = np.asarray(pmeta.r, dtype=float).reshape(-1)
        zv = np.asarray(pmeta.z, dtype=float).reshape(-1)
    else:
        arr = np.asarray(psi)
        psi2d = arr[0, :, :] if arr.ndim == 3 else arr
        xv = np.asarray(x, dtype=float).reshape(-1)
        zv = np.asarray(z, dtype=float).reshape(-1)

    if psival is None:
        lc = read_lcfs(filename=filename, slice=slice, return_meta=True, **lcfs_kwargs)
        psival = float(lc.psilim)

    fig = Figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(xv, zv, psi2d.T, levels=[float(psival)])
    
    try:
        all_x, all_y = [], []
        
        if hasattr(cs, 'collections'):
            # Legacy behavior (Matplotlib < 3.8)
            if not cs.collections or not cs.collections[0].get_paths():
                return np.zeros((2, 0), dtype=float)
            
            # 모든 path(곡선)를 순회
            for path in cs.collections[0].get_paths():
                vertices = path.vertices
                all_x.extend(vertices[:, 0])
                all_x.append(np.nan) # 곡선 구분을 위해 nan 삽입
                all_y.extend(vertices[:, 1])
                all_y.append(np.nan)
                
        else:
            # New behavior (Matplotlib >= 3.8)
            paths = cs.get_paths()
            if not paths or not paths[0].to_polygons():
                return np.zeros((2, 0), dtype=float)
            
            # 분리된 모든 다각형(곡선)을 순회
            for poly in paths[0].to_polygons():
                all_x.extend(poly[:, 0])
                all_x.append(np.nan) # 곡선 구분을 위해 nan 삽입
                all_y.extend(poly[:, 1])
                all_y.append(np.nan)
        
        # 추출된 데이터가 없는 경우 안전망
        if not all_x:
            return np.zeros((2, 0), dtype=float)
            
        # 마지막에 불필요하게 들어간 nan을 제거하고 (2, N) 형태로 반환
        return np.array([all_x[:-1], all_y[:-1]], dtype=float)

    finally:
        fig.clear()
