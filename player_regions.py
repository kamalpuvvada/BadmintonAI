"""Explicit image-space inclusion regions; these are not physical calibrations."""
import numpy as np


def validate_polygon(points):
    vertices = np.asarray(points, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1] != 2 or not 3 <= len(vertices) <= 12
            or not np.isfinite(vertices).all() or (vertices < 0).any() or (vertices > 1).any()
            or len(set(map(tuple, vertices))) != len(vertices)):
        raise ValueError("Court polygon needs 3..12 distinct finite normalized x,y vertices")
    edges = np.roll(vertices, -1, axis=0) - vertices
    offsets = vertices[None, :, :] - vertices[:, None, :]
    cross = edges[:, None, 0] * offsets[:, :, 1] - edges[:, None, 1] * offsets[:, :, 0]
    area = abs(np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) - vertices[:, 1] * np.roll(vertices[:, 0], -1))) / 2
    if area < 1e-8 or not ((cross >= -1e-10).all() or (cross <= 1e-10).all()):
        raise ValueError("Court polygon must be convex with vertices in boundary order")
    return tuple(map(tuple, vertices.tolist()))


def parse_polygon(value):
    if value is None:
        return None
    try:
        return validate_polygon([list(map(float, pair.split(','))) for pair in value.split(';')])
    except (ValueError, TypeError) as exc:
        raise ValueError("Court polygon must be convex normalized x,y pairs separated by semicolons") from exc


def polygon_mask(points, polygon):
    vertices = np.asarray(validate_polygon(polygon))
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    edges = np.roll(vertices, -1, axis=0) - vertices
    offsets = points[None, :, :] - vertices[:, None, :]
    cross = edges[:, None, 0] * offsets[:, :, 1] - edges[:, None, 1] * offsets[:, :, 0]
    return (cross >= -1e-10).all(axis=0) | (cross <= 1e-10).all(axis=0)
