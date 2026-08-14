#!/usr/bin/env python3
"""Compare rendered S4 node positions with the Howard p.64 figure."""

import argparse
import html
import json
import math
import re
import sys


PRINTED = {
    "4321": (821, 273), "4312": (1231, 344), "3421": (529, 363),
    "3412": (952, 438), "4231": (762, 480), "4132": (1551, 619),
    "3241": (187, 670), "4213": (1071, 734), "2431": (449, 757),
    "4123": (1452, 809), "3142": (1020, 848), "2341": (151, 861),
    "2413": (755, 1004), "1432": (1628, 1007), "3214": (227, 1068),
    "1342": (1367, 1135), "3124": (640, 1170), "1423": (1520, 1181),
    "2314": (199, 1246), "2143": (804, 1365), "1324": (1001, 1449),
    "1243": (1188, 1463), "2134": (528, 1498), "1234": (929, 1606),
}
STRETCH = 1.1419
ANGLE_TOLERANCE = 1.5
MEAN_RESID_TOLERANCE = 30.0
MAX_RESID_TOLERANCE = 60.0


def load_positions(path):
    dom = open(path).read()
    match = re.search(r'<pre id="renderedPositions"[^>]*>(.*?)</pre>', dom, re.S)
    if not match:
        raise ValueError("the page did not emit rendered positions")
    return json.loads(html.unescape(match.group(1)))


def mutate(positions, name):
    points = {label: list(point) for label, point in positions.items()}
    if name == "none":
        return points
    if name == "swap-labels":
        points["1342"], points["1423"] = points["1423"], points["1342"]
    elif name == "shift-node":
        points["1342"][0] += 250
    elif name == "mirror":
        center = sum(point[0] for point in points.values()) / len(points)
        for point in points.values():
            point[0] = 2 * center - point[0]
    elif name == "rotate":
        center_x = sum(point[0] for point in points.values()) / len(points)
        center_y = sum(point[1] for point in points.values()) / len(points)
        angle = math.radians(3)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for point in points.values():
            x, y = point[0] - center_x, point[1] - center_y
            point[0] = center_x + x * cos_a - y * sin_a
            point[1] = center_y + x * sin_a + y * cos_a
    else:
        raise ValueError(f"unknown mutation: {name}")
    return points


def verify(ours):
    common = sorted(set(ours) & set(PRINTED))
    if len(common) != 24:
        return [f"matched {len(common)} labels, expected 24"], None

    ax = [ours[label][0] for label in common]
    ay = [ours[label][1] for label in common]
    bx = [PRINTED[label][0] / STRETCH for label in common]
    by = [PRINTED[label][1] for label in common]
    amx, amy = sum(ax) / 24, sum(ay) / 24
    bmx, bmy = sum(bx) / 24, sum(by) / 24
    ax = [value - amx for value in ax]
    ay = [value - amy for value in ay]
    bx = [value - bmx for value in bx]
    by = [value - bmy for value in by]

    num = sum(ax[i] * by[i] - ay[i] * bx[i] for i in range(24))
    den = sum(ax[i] * bx[i] + ay[i] * by[i] for i in range(24))
    theta = math.degrees(math.atan2(num, den))
    cos_t, sin_t = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    scale = (den * cos_t + num * sin_t) / sum(
        ax[i] * ax[i] + ay[i] * ay[i] for i in range(24)
    )
    residuals = []
    for i in range(24):
        px = scale * (ax[i] * cos_t - ay[i] * sin_t)
        py = scale * (ax[i] * sin_t + ay[i] * cos_t)
        residuals.append(math.hypot(bx[i] - px, by[i] - py))
    mean_resid = sum(residuals) / 24
    max_resid = max(residuals)
    span = 2 * max(math.hypot(bx[i], by[i]) for i in range(24))

    failures = []
    if abs(theta) > ANGLE_TOLERANCE:
        failures.append(
            f"rendered layout needs {theta:+.2f} deg of rotation to match the printed figure "
            f"(tolerance {ANGLE_TOLERANCE})"
        )
    if mean_resid > MEAN_RESID_TOLERANCE:
        failures.append(
            f"mean residual {mean_resid:.1f} px after best alignment, tolerance "
            f"{MEAN_RESID_TOLERANCE} px on a {span:.0f} px span"
        )
    if max_resid > MAX_RESID_TOLERANCE:
        failures.append(
            f"maximum node residual {max_resid:.1f} px after best alignment, tolerance "
            f"{MAX_RESID_TOLERANCE} px"
        )
    return failures, (theta, mean_resid, max_resid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dom")
    parser.add_argument(
        "--mutate",
        choices=["none", "swap-labels", "shift-node", "mirror", "rotate"],
        default="none",
    )
    args = parser.parse_args()
    try:
        positions = mutate(load_positions(args.dom), args.mutate)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}")
        return 1

    failures, metrics = verify(positions)
    if failures:
        print(f"FAIL: orientation gate ({args.mutate})")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    theta, mean_resid, max_resid = metrics
    print(
        "PASS: rendered layout matches the printed p.64 figure, "
        f"{theta:+.2f} deg rotation, {mean_resid:.1f} px mean residual, "
        f"{max_resid:.1f} px maximum residual."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
