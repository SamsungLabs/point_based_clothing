import math

import numpy as np
import torch


def correct_bbox(ltrb, h, w):
    l, t, r, b = ltrb

    l = max(0, l)
    t = max(0, t)
    r = min(w, r)
    b = min(h, b)

    return (l, t, r, b)


def get_bbox_from_verts(verts):
    verts_projected = verts.copy()
    verts_projected[:, :2] /= verts_projected[:, 2:]

    x = verts_projected[:, 0]
    y = verts_projected[:, 1]

    x = x[x == x]
    y = y[y == y]

    # get bbox in format (left, top, right, bottom)
    l = int(np.min(x))
    t = int(np.min(y))
    r = int(np.max(x))
    b = int(np.max(y))

    return (l, t, r, b)


def scale_bbox(ltrb, scale):
    left, upper, right, lower = ltrb
    width, height = right - left, lower - upper

    x_center, y_center = (right + left) // 2, (lower + upper) // 2
    new_width, new_height = int(scale * width), int(scale * height)

    new_left = x_center - new_width // 2
    new_right = new_left + new_width

    new_upper = y_center - new_height // 2
    new_lower = new_upper + new_height

    return new_left, new_upper, new_right, new_lower


def get_square_bbox(ltrb):
    left, upper, right, lower = ltrb
    width, height = right - left, lower - upper

    if width > height:
        y_center = (upper + lower) // 2
        upper = y_center - width // 2
        lower = upper + width
    else:
        x_center = (left + right) // 2
        left = x_center - height // 2
        right = left + height

    return left, upper, right, lower
