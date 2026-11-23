#!/usr/bin/env python3
"""Table-of-contents graphic (3.25 x 1.75 in, the ACS size) in the blue-pink water-lily palette and sans-serif type.

The 3-D surface is J(phi, omega) at psi = 40 degrees, read from amide_correction_slice.tsv (this folder) and drawn
here with a small software renderer (no Blender): the frozen 151 x 151 grid is interpolated, each pixel is coloured by
the J colour scale below. By default (--look smooth) the colour runs smoothly through the palette and there are no
boundary or contour lines on the surface; --look banded gives the earlier 7-step version with thin white band edges.
No fitting, QM or MD is involved.

usage: python plot_TOC.py [--out DIR] [--look smooth|banded]
Writes NMe_TOC_energy_surface_3D.{pdf,svg,png,tif}; PNG/TIFF are 600 dpi, with RGB TIFF.
"""
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from manual_structures import image_for, STRUCTURE_SCALE
# ACS TOC lettering is sans-serif. This independent script preserves its
# own font settings rather than importing the manuscript's serif wrapper.
TOC_FONT = "Liberation Sans"
TOC_MIN_PT = 6.5

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

NAVY, BLUE2, MID, LIGHT = "#184581", "#1B6DC0", "#4396CD", "#9CC4E5"
PINK, SAND, CREAM = "#F9D4CA", "#FAE0B7", "#FBECD7"
INK, SECOND, EDGE, GRID = "#182b36", "#4a636e", "#6f7f89", "#d3dae0"
EDGES = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 10]            # kcal/mol; contour spacing 2.5
BANDS = [NAVY, BLUE2, MID, LIGHT, CREAM, SAND, PINK]
CENTRES = [-8.75, -6.25, -3.75, -1.25, 1.25, 3.75, 7.5]   # mid-points of the 7 bands: the smooth scale passes through each palette colour here
PAL = np.array([[int(c[k:k + 2], 16) for k in (1, 3, 5)] for c in BANDS], np.float32)


def smooth_colour(J):
    """Palette colour (RGB 0-255) for J (kcal/mol): linear blend between the band-centre colours, clamped at both ends."""
    J = np.asarray(J, np.float32)
    return np.stack([np.interp(J, CENTRES, PAL[:, k]) for k in range(3)], axis=-1).astype(np.float32)
W, H = 1320.0, 710.0                                     # drawing units; 1320 u = 3.25 in
PT = 72 * 3.25 / W                                       # points per unit (0.177)

# oblique projection of the (phi, omega, J) box, in drawing units
A = np.array([584.1, 498.9])                             # phi = -120, omega = 0, J = -10
PHI_V, OM_V, J_U = np.array([354.0, 119.0]), np.array([139.3, -153.4]), 9.84


def proj(phi, om, J):
    return (A[0] + (np.asarray(phi) + 120) / 180 * PHI_V[0] + np.asarray(om) / 180 * OM_V[0],
            A[1] + (np.asarray(phi) + 120) / 180 * PHI_V[1] + np.asarray(om) / 180 * OM_V[1] - (np.asarray(J) + 10) * J_U)


def read_grid(path):
    with open(path, encoding="utf-8") as f:
        names = f.readline().strip().split("\t")
    cols = [names.index(k) for k in ("omega_deg", "phi_deg", "J_kcal_mol_at_psi40")]
    om, phi, J = np.loadtxt(path, delimiter="\t", skiprows=1, usecols=cols).T
    po, pp = np.unique(om), np.unique(phi)
    grid = np.empty((po.size, pp.size))
    grid[np.searchsorted(po, om), np.searchsorted(pp, phi)] = J
    assert grid.shape == (151, 151) and len(om) == grid.size
    return po, pp, grid


def render_surface(grid, ss=6, up=2, look="smooth"):
    """Painter's-algorithm rendering with per-pixel interpolation. Returns an RGBA array, 3 px per drawing unit."""
    om = np.linspace(0, 180, grid.shape[0] * up - (up - 1))
    phi = np.linspace(-120, 60, grid.shape[1] * up - (up - 1))
    from scipy.interpolate import RectBivariateSpline        # cubic interpolation on the exact grid
    spl = RectBivariateSpline(np.linspace(0, 180, grid.shape[0]), np.linspace(-120, 60, grid.shape[1]), grid, kx=3, ky=3)
    Jg = spl(om, phi)
    X0, Y0, wu, hu = 575.0, 222.0, 515.0, 410.0
    px, py = proj(phi[None, :], om[:, None], Jg)
    px, py = (px - X0) * ss, (py - Y0) * ss
    Wp, Hp = int(wu * ss), int(hu * ss)
    Jb = np.full((Hp, Wp), np.nan, np.float32)
    Ib = np.full((Hp, Wp), -1, np.int16)
    Sb = np.ones((Hp, Wp), np.float32)
    # light shading: +-6 % on the band colours, from the surface normal (light from upper left, in front)
    kz = .02
    gx = spl(om, phi, dy=1) * 180 * kz            # dJ/dphi per unit box width
    gy = spl(om, phi, dx=1) * 180 * kz            # dJ/domega
    nrm = np.sqrt(gx ** 2 + gy ** 2 + 1)
    l = np.array([-.5, -.6, .65]); l /= np.linalg.norm(l)
    shade = .88 + .18 * np.clip((-gx * l[0] - gy * l[1] + l[2]) / nrm, 0, 1)
    def tri(i, a, b, c):
        (x0, y0, j0, s0), (x1, y1, j1, s1), (x2, y2, j2, s2) = a, b, c
        xa, xb = int(max(0, np.floor(min(x0, x1, x2)))), int(min(Wp, np.ceil(max(x0, x1, x2)) + 1))
        ya, yb = int(max(0, np.floor(min(y0, y1, y2)))), int(min(Hp, np.ceil(max(y0, y1, y2)) + 1))
        if xa >= xb or ya >= yb:
            return
        X, Y = np.meshgrid(np.arange(xa, xb) + .5, np.arange(ya, yb) + .5)
        d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(d) < 1e-9:
            return
        w0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / d
        w1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / d
        w2 = 1 - w0 - w1
        m = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        Jb[ya:yb, xa:xb][m] = (w0 * j0 + w1 * j1 + w2 * j2)[m]
        Ib[ya:yb, xa:xb][m] = i
        Sb[ya:yb, xa:xb][m] = (w0 * s0 + w1 * s1 + w2 * s2)[m]
    for i in range(om.size - 2, -1, -1):               # far (large omega) to near
        for j in range(phi.size - 1):
            p00 = (px[i, j], py[i, j], Jg[i, j], shade[i, j]); p01 = (px[i, j + 1], py[i, j + 1], Jg[i, j + 1], shade[i, j + 1])
            p10 = (px[i + 1, j], py[i + 1, j], Jg[i + 1, j], shade[i + 1, j]); p11 = (px[i + 1, j + 1], py[i + 1, j + 1], Jg[i + 1, j + 1], shade[i + 1, j + 1])
            tri(i, p00, p01, p11)
            tri(i, p00, p11, p10)
    surf = ~np.isnan(Jb)
    band = np.where(surf, np.digitize(np.nan_to_num(Jb), EDGES[1:-1]), -1).astype(np.int8)
    rgba = np.zeros((Hp, Wp, 4), np.float32)
    pal = np.array([[int(c[k:k + 2], 16) for k in (1, 3, 5)] for c in BANDS], np.float32)
    if look == "smooth":
        rgba[surf, :3] = np.clip(smooth_colour(Jb[surf]) * Sb[surf][:, None], 0, 255)
    else:
        rgba[surf, :3] = np.clip(pal[band[surf]] * Sb[surf][:, None], 0, 255)
    rgba[surf, 3] = 255
    # thin white contour lines at the band edges (as in the original TOC); no dark outline. The surface rim and the
    # depth jumps are only toned down a little (a darker shade of the band colour), so the sheet reads without a line.
    contour = np.zeros_like(surf); rim = np.zeros_like(surf)
    for sl_a, sl_b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        a_, b_ = band[sl_a], band[sl_b]; ia, ib = Ib[sl_a], Ib[sl_b]
        both = (a_ >= 0) & (b_ >= 0)
        chg = both & (a_ != b_)
        edge = (a_ != b_) & ~both
        jump = both & (np.abs(ia.astype(int) - ib.astype(int)) > 10)
        for m, tgt in ((chg, contour), (edge | jump, rim)):
            t = np.zeros_like(surf); t[sl_a] |= m; t[sl_b] |= m; tgt |= t
    def disk(r):
        y, x = np.ogrid[-r:r + 1, -r:r + 1]; return (x * x + y * y) <= r * r
    pxpt = ss / PT                                                                       # px per point
    cl = binary_dilation(contour, disk(max(1, int(round(0.28 * pxpt / 2)))))            # 0.28 pt white contour
    rm = binary_dilation(rim, disk(max(1, int(round(0.45 * pxpt / 2))))) & surf         # soft rim, inside the sheet
    if look == "banded":                                  # the smooth look has no contour lines and no rim tone at all
        rgba[rm, :3] *= .80
        rgba[cl & surf, :3] = rgba[cl & surf, :3] * .15 + 255 * .85
    # premultiplied 2x2 down-sampling for anti-aliasing
    f = 2
    pre = rgba.copy(); pre[..., :3] *= pre[..., 3:] / 255
    h2, w2 = Hp // f * f, Wp // f * f
    pre = pre[:h2, :w2].reshape(h2 // f, f, w2 // f, f, 4).mean(axis=(1, 3))
    out = np.zeros_like(pre); aa = pre[..., 3:]
    out[..., :3] = np.where(aa > 0, pre[..., :3] / np.maximum(aa, 1e-6) * 255, 0); out[..., 3] = aa[..., 0]
    return np.clip(out, 0, 255).astype(np.uint8), (X0, Y0, wu, hu)


BONDS = [   # x1, y1, x2, y2, stroke, width -- frozen drawing of the capped N-methylalanine (original TOC coordinates)
    (108.0, 216.0, 154.57, 242.19, '#f3d7bd', 12.0),
    (189.36, 242.08, 235.0, 216.0, '#ceeae3', 12.0),
    (235.0, 216.0, 298.0, 252.0, '#ceeae3', 12.0),
    (68.44, 238.6, 108.0, 216.0, '#1f2b35', 3.6),
    (108.0, 216.0, 157.18, 243.67, '#1f2b35', 3.6),
    (186.76, 243.57, 235.0, 216.0, '#1f2b35', 3.6),
    (235.0, 216.0, 298.0, 252.0, '#1f2b35', 3.6),
    (298.0, 252.0, 339.25, 228.05, '#1f2b35', 3.6),
    (380.92, 227.77, 400.47, 238.76, '#1f2b35', 3.6),
    (172.0, 269.0, 172.0, 305.0, '#1f2b35', 3.6),
    (104.2, 214.5, 104.2, 167.0, '#1f2b35', 3.1),
    (111.8, 214.5, 111.8, 167.0, '#1f2b35', 3.1),
    (294.2, 253.5, 294.2, 306.0, '#1f2b35', 3.1),
    (301.8, 253.5, 301.8, 306.0, '#1f2b35', 3.1),
]


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("redraw_TOC"))
    p.add_argument("--look", choices=("smooth", "banded"), default="smooth",
                   help="smooth: continuous palette colour, no boundary lines (default); banded: 7 steps with thin white band edges")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    om, phi, grid = read_grid(HERE / "amide_correction_slice.tsv")
    assert abs(grid.min() + 6.566710) < 1e-5 and abs(grid.max() - 9.612154) < 1e-5     # the frozen J range
    img, (X0, Y0, wu, hu) = render_surface(grid, look=a.look)
    # Composite on the white page before resampling: transparent RGB fringes must
    # not introduce a dark outline around the numerical surface.
    alpha = img[..., 3:4].astype(np.float32) / 255.0
    img = np.rint(img[..., :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)

    plt.rcParams.update({"font.family": TOC_FONT, "font.size": 7.0,
                         "mathtext.fontset":"custom", "mathtext.rm":TOC_FONT,
                         "mathtext.it":TOC_FONT+":italic", "mathtext.bf":TOC_FONT+":bold",
                         "mathtext.sf":TOC_FONT, "mathtext.fallback":"stixsans",
                         "pdf.fonttype": 42, "ps.fonttype":42,
                         "svg.fonttype": "path", "axes.unicode_minus": True})
    fig = plt.figure(figsize=(3.25, 1.75), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    lw = lambda pt: pt
    T = lambda x, y, s, size, **k: ax.text(x, y, s, fontsize=max(size, TOC_MIN_PT), color=k.pop("color", INK), **k)

    # Open coordinate arrows only; no enclosing floor box, grid or surface rim.
    C = lambda ph, o: proj(ph, o, -10)
    for p0, p1 in (((-120, 0), (60, 0)), ((60, 0), (60, 180))):
        ax.annotate('', xy=C(*p1), xytext=C(*p0), zorder=4,
                    arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=.65,
                                    mutation_scale=5, shrinkA=0, shrinkB=0))
    for ph in (-60, 0):                                  # ticks on the phi axis (front edge)
        x, y = C(ph, 0); ax.plot([x, x - 5], [y, y + 12], color=EDGE, lw=.6, zorder=4)
    for o in (60, 120):                                  # ticks on the omega axis (right edge)
        x, y = C(60, o); ax.plot([x, x + 12], [y, y + 5], color=EDGE, lw=.6, zorder=4)
    ax.imshow(img, extent=(X0, X0 + wu, Y0 + hu, Y0), zorder=2, interpolation="lanczos", aspect="auto")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    # --- J axis
    x0 = A[0]; yb, yt = proj(-120, 0, -10)[1], proj(-120, 0, 10)[1]
    ax.plot([x0, x0], [yb, yt], color=EDGE, lw=.8, zorder=4, solid_capstyle="butt")
    for v, lab in ((10, "+10"), (0, "0"), (-10, "−10")):
        y = proj(-120, 0, v)[1]
        ax.plot([x0 - 11, x0], [y, y], color=EDGE, lw=.8, zorder=4, solid_capstyle="butt")
        T(x0 - 17, y, lab, 6.0, ha="right", va="center", color=SECOND)
    T(x0, yt - 22, "J", 8.0, ha="center", va="center", style="italic", weight="bold", color=SECOND)
    T(747, 604, "φ", 8.0, ha="center", va="center", style="italic", weight="bold", color=SECOND)
    T(1040, 548, "ω", 8.0, ha="center", va="center", style="italic", weight="bold", color=SECOND)

    # Authors' Full/R1 snapshot; preserve the v3.0.7 TOC typography and energy surface.
    mw, mh = 36 * STRUCTURE_SCALE, 40 * STRUCTURE_SCALE
    molecule = fig.add_axes(((210-mw/2)/234, 1-(43+mh/2)/126, mw/234, mh/126), zorder=5)
    molecule.imshow(image_for('FULL', 'R1', cutout=True), interpolation='none')
    molecule.set_axis_off()
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    T(1282, 66, "tri-NMe in DMSO", 8.0, ha="right", va="center", weight="bold")
    T(1282, 106, "Representative Full frame", 6.5, ha="right", va="center", color=SECOND)

    # --- title block
    T(64, 80, "N-methylated cyclic peptides", 8.0, ha="left", va="center", color=SECOND, weight="bold")
    ax.add_patch(Rectangle((64, 38), 54, 7, fc=BLUE2, ec="none", zorder=2))
    T(62, 150, "Backbone–amide", 9.0, ha="left", va="center", weight="bold")
    T(62, 218, "coupling", 9.0, ha="left", va="center", weight="bold")

    # --- capped N-methylalanine (bonds from the original drawing)
    tx = lambda x: 21 + 1.11 * x; ty = lambda y: 133 + 1.11 * y + 2
    colour = {"#f3d7bd": PINK, "#ceeae3": LIGHT, "#1f2b35": INK}
    for x1, y1, x2, y2, c, w in BONDS:
        hl = c in ("#f3d7bd", "#ceeae3")
        ax.plot([tx(x1), tx(x2)], [ty(y1), ty(y2)], color=colour[c], lw=w * 1.11 * PT, zorder=2 if hl else 3,
                solid_capstyle="round")
    ax.add_patch(Polygon([(tx(235), ty(216)), (tx(228), ty(167)), (tx(242), ty(167))], fc=INK, ec="none", zorder=3))
    for x, y, s, c in ((45, 261, "Me", INK), (424, 261, "Me", INK), (172, 334, "Me", BLUE2), (235, 157, "Me", INK),
                       (172, 261, "N", BLUE2), (360, 225, "NH", INK), (108, 157, "O", INK), (298, 334, "O", INK)):
        T(tx(x), ty(y) - 9, s, 6.5, ha="center", va="center", color=c, zorder=5)
    for x, y, s in ((138, 211, "ω"), (206, 271, "φ"), (269, 211, "ψ")):
        T(tx(x), ty(y) - 9, s, 7.5, ha="center", va="center", style="italic", weight="bold", zorder=5)

    # --- visual summary of the correction, without a numerical result callout
    ax.plot([64, 495], [538, 538], color=GRID, lw=.8, solid_capstyle="round")
    T(64, 580, "Fragment QM", 7.0, ha="left", va="center", color=SECOND)
    T(62, 641, r"$V = A + J$", 12.2, ha="left", va="center")

    # --- colour scale of J (kcal/mol)
    cx = 926.0                                           # bar + unit end on the right edge of the title above (x = 1290)
    T(1290, 622, "Fitted J at \u03c8 = 40\u00b0", 7.0, ha="right", va="center", weight="bold")
    xs = cx + (np.array(EDGES) + 10) * 9.75
    if a.look == "smooth":                                # continuous bar, same colour function as the surface
        grad = smooth_colour(np.linspace(-10, 10, 400))[None, :, :] / 255
        ax.imshow(grad, extent=(cx, cx + 195, 668, 657), zorder=3, interpolation="bilinear", aspect="auto")
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
    else:
        for k, c in enumerate(BANDS):
            ax.add_patch(Rectangle((xs[k], 657), xs[k + 1] - xs[k], 11, fc=c, ec=EDGE, lw=.4, zorder=3))
    for v, lab in ((-10, "\u221210"), (0, "0"), (10, "+10")):
        T(cx + (v + 10) * 9.75, 688, lab, 6.0, ha="center", va="center", color=SECOND)
    T(1290, 662, "kcal/mol", 6.5, ha="right", va="center", color=SECOND)

    stem = "NMe_TOC_energy_surface_3D"
    title = "Backbone\u2013amide coupling with a three-dimensional Fourier correction surface"
    desc = ("J(phi, psi, omega) at psi = 40 degrees: the height is J in kcal/mol, the horizontal axes are phi from -120 to +60 "
            "degrees and omega from 0 to 180 degrees. It is a fitted local amide-torsion correction, not a free-energy surface "
            "of the cyclic peptide. The molecule is a separate representative Full tri-NMe frame in DMSO.")
    fig.savefig(a.out / f"{stem}.pdf", facecolor="white", metadata={"Title": title, "Subject": desc})
    fig.savefig(a.out / f"{stem}.svg", facecolor="white", metadata={"Title": title, "Description": desc})
    fig.savefig(a.out / f"{stem}.png", dpi=600, facecolor="white")
    with Image.open(a.out / f"{stem}.png") as raster:
        raster.convert("RGB").save(a.out / f"{stem}.tif", compression="tiff_lzw", dpi=(600,600))
    plt.close(fig)
    print("TOC written:", a.out)


if __name__ == "__main__":
    main()
