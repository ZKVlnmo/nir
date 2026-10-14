"""Проекция пикселя на локальную горизонтальную плоскость, Python >= 3.10.

WGS84, высоты эллипсоидальные в метрах. Оси камеры: вправо, вниз, вперёд.
Требует: opencv-python, numpy.
"""
from math import sin, cos, atan2, sqrt, hypot, radians, degrees
import json

import cv2
import numpy as np


A = 6378137.0
E2 = 6.6943799901413165e-3


def _mv(m, v):
    """Матрица 3x3 умножить на вектор."""
    result = []
    for row in m:
        s = 0
        for a, b in zip(row, v):
            s += a * b
        result.append(s)
    return tuple(result)


def _mm(a, b):
    """Матрица 3x3 умножить на матрицу 3x3."""
    result = []
    for i in range(3):
        row = []
        for j in range(3):
            s = 0
            for k in range(3):
                s += a[i][k] * b[k][j]
            row.append(s)
        result.append(tuple(row))
    return tuple(result)


def camera_to_enu(yaw_deg=0.0, pitch_deg=-90.0, roll_deg=0.0):
    """Углы камеры -> матрица поворота в ENU.

    yaw: азимут от севера по часовой (0 = север, 90 = восток)
    pitch: наклон оптической оси (-90 = вниз, 0 = горизонт)
    roll: поворот вокруг оптической оси
    """
    y, p, r = map(radians, (yaw_deg, pitch_deg, roll_deg))
    rz = ((cos(y), -sin(y), 0), (sin(y), cos(y), 0), (0, 0, 1))
    ry = ((cos(p), 0, sin(p)), (0, 1, 0), (-sin(p), 0, cos(p)))
    mount = ((0, 0, 1), (1, 0, 0), (0, 1, 0))
    roll = ((cos(r), -sin(r), 0), (sin(r), cos(r), 0), (0, 0, 1))
    ned_to_enu = ((0, 1, 0), (1, 0, 0), (0, 0, -1))
    return _mm(ned_to_enu, _mm(rz, _mm(ry, _mm(mount, roll))))


def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    """Широта, долгота, высота -> X, Y, Z (ECEF)."""
    lat, lon = radians(lat_deg), radians(lon_deg)
    n = A / sqrt(1 - E2*sin(lat)**2)
    return ((n + h_m)*cos(lat)*cos(lon),
            (n + h_m)*cos(lat)*sin(lon),
            (n*(1 - E2) + h_m)*sin(lat))


def ecef_to_geodetic(x, y, z):
    """X, Y, Z (ECEF) -> широта, долгота, высота (WGS84)."""
    p = hypot(x, y)
    if p < 1e-9:
        return (90.0 if z > 0 else -90.0), 0.0, abs(z) - A*sqrt(1 - E2)
    lat = atan2(z, p*(1 - E2))
    for _ in range(30):
        n = A / sqrt(1 - E2*sin(lat)**2)
        nxt = atan2(z + E2*n*sin(lat), p)
        if abs(nxt - lat) < 1e-14:
            lat = nxt
            break
        lat = nxt
    h = p*cos(lat) + z*sin(lat) - A*sqrt(1 - E2*sin(lat)**2)
    return degrees(lat), degrees(atan2(y, x)), h


class Intrinsics:
    """Внутренние параметры камеры. Дисторсия устраняется через OpenCV."""

    def __init__(self, fx, fy, cx, cy, width, height,
                 distortion=(0., 0., 0., 0., 0.)):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.distortion = tuple(distortion)

        self.K = np.array([[fx, 0.0, cx],
                           [0.0, fy, cy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)

        self.dist = np.array(distortion, dtype=np.float64)

    def ray(self, u, v):
        """Пиксель -> луч (x, y, 1) с устранением дисторсии через OpenCV.
        """

        uv = np.array([[u, v]], dtype=np.float64).reshape(-1, 1, 2)
        norm = cv2.undistortPoints(uv, self.K, self.dist)
        x, y = norm[0][0]
        return (float(x), float(y), 1.0)


class GeoPoint:
    """Результат проекции: географическая точка."""

    def __init__(self, latitude_deg, longitude_deg, ellipsoid_height_m,
                 east_m, north_m, slant_range_m):
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.ellipsoid_height_m = ellipsoid_height_m
        self.east_m = east_m
        self.north_m = north_m
        self.slant_range_m = slant_range_m


class PixelProjector:
    """Пересечение луча с локальной горизонтальной плоскостью."""

    def __init__(self, intrinsics, latitude_deg, longitude_deg,
                 camera_height_m, ground_height_m,
                 yaw_deg=0., pitch_deg=-90., roll_deg=0.):
        self.k = intrinsics
        self.r = camera_to_enu(yaw_deg, pitch_deg, roll_deg)
        self.origin = geodetic_to_ecef(latitude_deg, longitude_deg,
                                       camera_height_m)
        self.dh = camera_height_m - ground_height_m

        lat, lon = radians(latitude_deg), radians(longitude_deg)
        self.enu_to_ecef = ((-sin(lon), -sin(lat)*cos(lon), cos(lat)*cos(lon)),
                            (cos(lon), -sin(lat)*sin(lon), cos(lat)*sin(lon)),
                            (0., cos(lat), sin(lat)))

    def project(self, u, v):
        """Пиксель -> GeoPoint."""
        # 1. Пиксель -> луч в системе камеры
        ray_cam = self.k.ray(u, v)
        # 2. Луч в ENU
        ray = _mv(self.r, ray_cam)
        # 3. Луч должен идти вниз
        if ray[2] >= 0:
            raise ValueError('Луч не пересекает поверхность')
        # 4. Пересечение с плоскостью up = -dh
        t = -self.dh / ray[2]
        east, north, up = (t * x for x in ray)
        # 5. ENU -> ECEF
        delta = _mv(self.enu_to_ecef, (east, north, up))
        xyz = tuple(a + b for a, b in zip(self.origin, delta))
        # 6. ECEF -> WGS84
        lat, lon, height = ecef_to_geodetic(*xyz)
        return GeoPoint(lat, lon, height, east, north,
                        sqrt(east*east + north*north + up*up))


def main():
    with open('example.json', encoding='utf-8') as f:
        config = json.load(f)

    projector = PixelProjector(Intrinsics(**config['intrinsics']),
                               **config['pose'])

    result = []
    for u, v in config['pixels']:
        point = projector.project(u, v)
        result.append({
            'latitude_deg': point.latitude_deg,
            'longitude_deg': point.longitude_deg,
            'ellipsoid_height_m': point.ellipsoid_height_m,
            'east_m': point.east_m,
            'north_m': point.north_m,
            'slant_range_m': point.slant_range_m,
        })

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()