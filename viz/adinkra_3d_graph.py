#!/usr/bin/env python3
"""
Generate 3D "ghost graph" visualizations of adinkra chromotopologies.

Produces four figures:
  1. adinkra_3d_n4.png   - N=4, [4,1,4] code (4 bosons, 4 fermions, 4 colors)
  2. adinkra_3d_n8.png   - N=8, [8,4,4] Hamming code (8 bosons, 8 fermions, 8 colors)
  3. adinkra_3d_n16_e8xe8.png - N=16, E8xE8 topology (128 bosons, 128 fermions)
  4. adinkra_3d_n16_d16.png   - N=16, D16 topology (128 bosons, 128 fermions)

An adinkra is a bipartite graph:
  - Bosons (white/open circles) on the bottom plane
  - Fermions (black/filled circles) on the top plane
  - Edges colored by supersymmetry generator index (1 color per Q_I)
  - Dashed edges carry negative signs

The N=16 figures use extreme transparency (alpha ~ 0.02-0.05) to create a
ghostly density-plot effect where overlapping edges produce brighter regions.

All output is saved alongside this script in viz/.
"""

import json
import os
import sys
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
N16_CODES_PATH = os.path.join(os.path.dirname(OUTPUT_DIR), "adinkra_codes_n16.json")
DPI = 300

# ---------------------------------------------------------------------------
# Adinkra construction utilities
# ---------------------------------------------------------------------------

def generate_code(generators, n):
    """
    Build the full linear code from a set of generator codewords over GF(2).
    Returns a set of all codewords (as ints).
    """
    code = {0}
    for g in generators:
        new = set()
        for c in code:
            new.add(c ^ g)
        code = code | new
    return code


def build_adinkra_graph(n, generators):
    """
    Build the adinkra quotient graph from the N-cube quotiented by a doubly
    even binary code.

    Returns:
        bosons:  list of boson representative indices
        fermions: list of fermion representative indices
        edges:   list of (boson_idx, fermion_idx, color, sign)
                 where boson_idx/fermion_idx are indices into the bosons/fermions lists
    """
    code = generate_code(generators, n)

    # Partition N-cube vertices into equivalence classes under XOR with code
    all_vertices = range(1 << n)
    visited = set()
    classes = []
    vertex_to_class = {}

    for v in all_vertices:
        if v in visited:
            continue
        orbit = set()
        for c in code:
            orbit.add(v ^ c)
        classes.append(min(orbit))  # canonical representative = smallest element
        class_idx = len(classes) - 1
        for u in orbit:
            visited.add(u)
            vertex_to_class[u] = class_idx

    # Bipartition: parity of the Hamming weight of the representative
    # In the N-cube, bosons have even weight, fermions have odd weight
    # (or vice versa; convention: even = boson)
    reps = classes
    boson_indices = []
    fermion_indices = []
    is_boson = {}

    for idx, rep in enumerate(reps):
        hw = bin(rep).count('1')
        if hw % 2 == 0:
            boson_indices.append(idx)
            is_boson[idx] = True
        else:
            fermion_indices.append(idx)
            is_boson[idx] = False

    # Map class indices to position indices within boson/fermion lists
    boson_pos = {idx: i for i, idx in enumerate(boson_indices)}
    fermion_pos = {idx: i for i, idx in enumerate(fermion_indices)}

    # Build edges: for each color I (0..N-1), each boson class connects to
    # exactly one fermion class via flipping bit I
    edges = []
    seen_edges = set()

    for color in range(n):
        mask = 1 << color
        for cls_idx, rep in enumerate(reps):
            if not is_boson.get(cls_idx, False):
                continue
            neighbor_vertex = rep ^ mask
            neighbor_cls = vertex_to_class[neighbor_vertex]

            edge_key = (cls_idx, neighbor_cls, color)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            # Sign: determined by the number of 1-bits in positions < color
            # in the representative. This gives the standard "L-matrix" sign.
            lower_bits = rep & ((1 << color) - 1)
            sign = (-1) ** bin(lower_bits).count('1')

            b_pos = boson_pos[cls_idx]
            f_pos = fermion_pos[neighbor_cls]
            edges.append((b_pos, f_pos, color, sign))

    return len(boson_indices), len(fermion_indices), edges


def circular_layout_3d(n_nodes, z, radius=1.0, phase=0.0):
    """Place nodes evenly on a circle at height z."""
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False) + phase
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z_arr = np.full(n_nodes, z)
    return x, y, z_arr


def spherical_layout_3d(n_nodes, z_center, radius=1.0, seed=42):
    """
    Place nodes on a spherical shell using the Fibonacci spiral for even spacing,
    then shift vertically by z_center.
    """
    rng = np.random.RandomState(seed)
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_nodes, dtype=float)

    # Fibonacci sphere
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_nodes)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi) + z_center

    return x, y, z


def get_edge_colors(n_colors, cmap_name='tab20'):
    """Get N distinct colors from a colormap."""
    if n_colors <= 10:
        cmap = matplotlib.colormaps['tab10']
        return [cmap(i / 10) for i in range(n_colors)]
    elif n_colors <= 20:
        cmap = matplotlib.colormaps['tab20']
        return [cmap(i / 20) for i in range(n_colors)]
    else:
        cmap = matplotlib.colormaps['rainbow']
        return [cmap(i / n_colors) for i in range(n_colors)]


def spectral_rainbow_16():
    """16 perceptually distinct colors for N=16 edge coloring."""
    cmap = matplotlib.colormaps['gist_rainbow']
    return [cmap(i / 16) for i in range(16)]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_adinkra_3d(
    n_bosons, n_fermions, edges, n_colors,
    title, filename,
    boson_radius=1.0, fermion_radius=1.0,
    boson_z=0.0, fermion_z=1.0,
    edge_alpha=0.4, edge_lw=1.0,
    boson_size=60, fermion_size=60,
    node_alpha=0.9,
    use_spherical=False,
    elev=25, azim=45,
    figsize=(10, 8),
    extra_angles=None,
    boson_phase=0.0, fermion_phase=0.0,
):
    """
    Render a single 3D adinkra ghost graph and save to PNG.

    Parameters
    ----------
    extra_angles : list of (elev, azim) tuples for additional viewpoints
                   (saved as filename_elev_azim.png)
    """
    edge_colors_palette = spectral_rainbow_16() if n_colors == 16 else get_edge_colors(n_colors)

    if use_spherical:
        bx, by, bz = spherical_layout_3d(n_bosons, boson_z, boson_radius, seed=42)
        fx, fy, fz = spherical_layout_3d(n_fermions, fermion_z, fermion_radius, seed=137)
    else:
        bx, by, bz = circular_layout_3d(n_bosons, boson_z, boson_radius, boson_phase)
        fx, fy, fz = circular_layout_3d(n_fermions, fermion_z, fermion_radius, fermion_phase)

    def _render_view(ev, az, out_path):
        fig = plt.figure(figsize=figsize, facecolor='#0a0a0a')
        ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a0a')

        # Draw edges
        for (bi, fi, color, sign) in edges:
            c = list(edge_colors_palette[color])
            c[3] = edge_alpha  # set alpha
            ls = '--' if sign < 0 else '-'
            ax.plot(
                [bx[bi], fx[fi]],
                [by[bi], fy[fi]],
                [bz[bi], fz[fi]],
                color=c, linestyle=ls, linewidth=edge_lw,
                zorder=1,
            )

        # Draw boson nodes (open circles -- hollow, white border)
        # Glow layer behind bosons for visibility on dark background
        ax.scatter(
            bx, by, bz,
            s=boson_size * 2.5, c='#4488ff', edgecolors='none',
            alpha=node_alpha * 0.15, linewidths=0, zorder=4,
            marker='o', depthshade=False,
        )
        ax.scatter(
            bx, by, bz,
            s=boson_size, c='#0a0a0a', edgecolors='#88bbff',
            alpha=node_alpha, linewidths=1.5, zorder=5,
            marker='o', depthshade=False,
        )

        # Draw fermion nodes (filled circles -- solid white)
        # Glow layer behind fermions
        ax.scatter(
            fx, fy, fz,
            s=fermion_size * 2.5, c='#ff8844', edgecolors='none',
            alpha=node_alpha * 0.15, linewidths=0, zorder=4,
            marker='o', depthshade=False,
        )
        ax.scatter(
            fx, fy, fz,
            s=fermion_size, c='#ffcc88', edgecolors='#ff8844',
            alpha=node_alpha, linewidths=1.0, zorder=5,
            marker='o', depthshade=False,
        )

        ax.view_init(elev=ev, azim=az)

        # Style the axes
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Make pane backgrounds dark
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#1a1a1a')
        ax.yaxis.pane.set_edgecolor('#1a1a1a')
        ax.zaxis.pane.set_edgecolor('#1a1a1a')

        # Grid
        ax.grid(False)

        # Title
        ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(out_path, dpi=DPI, facecolor='#0a0a0a', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    out = os.path.join(OUTPUT_DIR, filename)
    _render_view(elev, azim, out)

    if extra_angles:
        base, ext = os.path.splitext(filename)
        for ev, az in extra_angles:
            angle_filename = f"{base}_e{ev}_a{az}{ext}"
            _render_view(ev, az, os.path.join(OUTPUT_DIR, angle_filename))


# ---------------------------------------------------------------------------
# N=4 adinkra: [4,1,4] code
# ---------------------------------------------------------------------------

def make_n4():
    """N=4, k=1 adinkra from the [4,1,4] repetition code."""
    print("\n=== N=4 adinkra ([4,1,4] code) ===")
    n = 4
    generators = [0b1111]  # single generator: all four bits set
    nb, nf, edges = build_adinkra_graph(n, generators)
    print(f"  Bosons: {nb}, Fermions: {nf}, Edges: {len(edges)}")

    render_adinkra_3d(
        nb, nf, edges, n_colors=4,
        title="N=4 Adinkra  [4,1,4]",
        filename="adinkra_3d_n4.png",
        boson_radius=1.0, fermion_radius=1.0,
        boson_z=0.0, fermion_z=2.0,
        edge_alpha=0.55, edge_lw=2.0,
        boson_size=200, fermion_size=200,
        node_alpha=1.0,
        elev=20, azim=35,
        figsize=(10, 8),
        boson_phase=0.0, fermion_phase=np.pi / 4,
    )


# ---------------------------------------------------------------------------
# N=8 adinkra: [8,4,4] Hamming code
# ---------------------------------------------------------------------------

def make_n8():
    """N=8, k=4 adinkra from the [8,4,4] extended Hamming code."""
    print("\n=== N=8 adinkra ([8,4,4] Hamming code) ===")
    n = 8
    # Extended Hamming [8,4,4] generators
    generators = [
        0b11100001,
        0b11010010,
        0b10110100,
        0b01111000,
    ]
    nb, nf, edges = build_adinkra_graph(n, generators)
    print(f"  Bosons: {nb}, Fermions: {nf}, Edges: {len(edges)}")

    render_adinkra_3d(
        nb, nf, edges, n_colors=8,
        title="N=8 Adinkra  [8,4,4] Hamming",
        filename="adinkra_3d_n8.png",
        boson_radius=1.5, fermion_radius=1.5,
        boson_z=0.0, fermion_z=2.5,
        edge_alpha=0.35, edge_lw=1.2,
        boson_size=120, fermion_size=120,
        node_alpha=1.0,
        elev=25, azim=40,
        figsize=(10, 8),
        boson_phase=0.0, fermion_phase=np.pi / 8,
    )


# ---------------------------------------------------------------------------
# N=16 adinkras from the code catalog
# ---------------------------------------------------------------------------

def load_n16_generators(code_index):
    """Load generator codewords from the N=16 catalog."""
    with open(N16_CODES_PATH) as f:
        data = json.load(f)
    code_entry = data['codes'][code_index]
    return code_entry['generators_raw']


def make_n16(code_index, label, filename, extra_angles=None):
    """Build and render an N=16 adinkra."""
    print(f"\n=== N=16 adinkra ({label}, code index {code_index}) ===")
    n = 16
    generators = load_n16_generators(code_index)
    print(f"  Generators: {[hex(g) for g in generators]}")

    nb, nf, edges = build_adinkra_graph(n, generators)
    print(f"  Bosons: {nb}, Fermions: {nf}, Edges: {len(edges)}")

    if extra_angles is None:
        extra_angles = [(10, 0), (45, 90), (80, 180)]

    render_adinkra_3d(
        nb, nf, edges, n_colors=16,
        title=f"N=16 Adinkra  {label}",
        filename=filename,
        boson_radius=3.0, fermion_radius=3.0,
        boson_z=0.0, fermion_z=5.0,
        edge_alpha=0.08, edge_lw=0.4,
        boson_size=12, fermion_size=12,
        node_alpha=0.85,
        use_spherical=True,
        elev=20, azim=45,
        figsize=(14, 11),
        extra_angles=extra_angles,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Output directory: {OUTPUT_DIR}")

    make_n4()
    make_n8()

    if os.path.exists(N16_CODES_PATH):
        make_n16(
            code_index=75,
            label="E8 x E8 topology",
            filename="adinkra_3d_n16_e8xe8.png",
        )
        make_n16(
            code_index=76,
            label="D16 topology",
            filename="adinkra_3d_n16_d16.png",
        )
    else:
        print(f"\nWARNING: N=16 code catalog not found at {N16_CODES_PATH}")
        print("  Skipping N=16 visualizations.")

    print("\nDone.")


if __name__ == "__main__":
    main()
