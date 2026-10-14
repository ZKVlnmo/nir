"""SRTM HGT -> высота земли под дроном + GSD. Простая версия с numpy."""
from math import floor
from pathlib import Path

import numpy as np

NODATA = -32768


def _read_hgt(path):
    """Читает HGT в numpy-массив. Возвращает (grid, south, west, n)."""
    path = Path(path)
    name = path.name.upper()

    south = int(name[1:3]) * (1 if name[0] == 'N' else -1)
    west = int(name[4:7]) * (1 if name[3] == 'E' else -1)

    data = np.fromfile(path, dtype='>i2')
    n = int(np.sqrt(data.size))
    return data.reshape(n, n), south, west, n


def _interpolate(grid, south, west, n, lat, lon):
    """Билинейная интерполяция высоты."""
    row = (south + 1 - lat) * (n - 1)
    col = (lon - west) * (n - 1)
    r0 = min(int(floor(row)), n - 2)
    c0 = min(int(floor(col)), n - 2)
    dy, dx = row - r0, col - c0

    v00 = int(grid[r0, c0])
    v01 = int(grid[r0, c0 + 1])
    v10 = int(grid[r0 + 1, c0])
    v11 = int(grid[r0 + 1, c0 + 1])

    if NODATA in (v00, v01, v10, v11):
        raise ValueError('NODATA в ячейке')

    return (v00 * (1 - dy) * (1 - dx)
            + v01 * (1 - dy) * dx
            + v10 * dy * (1 - dx)
            + v11 * dy * dx)


def elevation(dem_dir, lat, lon):
    """Высота земли в точке (lat, lon) над EGM96."""
    south, west = floor(lat), floor(lon)

    ns = 'N' if south >= 0 else 'S'
    ew = 'E' if west >= 0 else 'W'
    name = f'{ns}{abs(south):02d}{ew}{abs(west):03d}.hgt'

    path = Path(dem_dir) / name
    if not path.is_file():
        path = Path(dem_dir) / name.lower()
    if not path.is_file():
        raise FileNotFoundError(f'Не найден {name}')

    grid, s, w, n = _read_hgt(path)
    return _interpolate(grid, s, w, n, lat, lon)


def gsd(dem_dir, lat, lon, camera_height, datum,
        fx, fy, N=None):
    """Высота земли + скорректированный GSD.

    datum='egm96' o-> высотiа камеры уже в EGM96
    datum='ellipsoid' -> H = h - N
    """

    if datum == 'egm96':
        H_cam = float(camera_height)
    elif datum == 'ellipsoid':
        if N is None:
            raise ValueError('Нужна N для эллипсоидальной высоты')
        H_cam = camera_height - N
    else:
        raise ValueError('datum: egm96 или ellipsoid')


    H_surf = elevation(dem_dir, lat, lon)


    hrel = H_cam - H_surf
    if hrel <= 0:
        raise ValueError('Камера ниже земли')

    # 4. GSD
    return {
        'surface_height_egm96_m': H_surf,
        'camera_height_egm96_m': H_cam,
        'relative_height_m': hrel,
        'gsd_x_m_per_pixel': hrel / fx,
        'gsd_y_m_per_pixel': hrel / fy,
    }


if __name__ == '__main__':
    import json
    result = gsd(
        'dem', 59.9386, 30.3141,
        camera_height=200,
        datum='egm96',
        fx=1000, fy=1000,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))