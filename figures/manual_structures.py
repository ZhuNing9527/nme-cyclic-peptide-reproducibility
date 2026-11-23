"""Compose the authors' ChimeraX snapshots without rendering or changing coordinates."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent / 'inputs'
PAGE_WIDTH = 491.76
COMMON_CROP = (688, 191, 1896, 1521)
SOURCE_SIZE = (2400, 1800)
SOURCE_OVERRIDES = {
    ('FULL', 'R2'): {'canvas': (2048, 1536), 'crop': (552, 163, 1583, 1298)},
}
PREVIOUS_IMAGE_WIDTH_PT = 128.0
STRUCTURE_SCALE = 0.90
IMAGE_WIDTH_PT = PREVIOUS_IMAGE_WIDTH_PT * STRUCTURE_SCALE
IMAGE_HEIGHT_PT = IMAGE_WIDTH_PT * (COMMON_CROP[3] - COMMON_CROP[1]) / (COMMON_CROP[2] - COMMON_CROP[0])
PREVIOUS_IMAGE_HEIGHT_PT = IMAGE_HEIGHT_PT / STRUCTURE_SCALE
INK = '#182B36'
MUTED = '#506775'
SAND = '#FAE0B7'
ARMS = ('M0', 'A_ONLY', 'FULL')
ARM_LABELS = ('M0', 'A only', 'Full')
ARM_COLORS = ('#F9D4CA', '#4396CD', '#184581')
CENTERS = (.20, .51, .82)
FILES = {
    ('M0', 'R1'): 'Figure_6/Figure6a_M0_R1_white.png',
    ('A_ONLY', 'R1'): 'Figure_6/Figure6a_A_ONLY_R1_white.png',
    ('FULL', 'R1'): 'Figure_6/Figure6a_FULL_R1_white.png',
    ('M0', 'R2'): 'Figure_S3/FigureS3a_M0_R2_white.png',
    ('A_ONLY', 'R2'): 'Figure_S3/FigureS3a_A_ONLY_R2_white.png',
    ('FULL', 'R2'): 'Figure_S3/FigureS3a_FULL_R2_white.png',
}


def image_for(arm, repeat, cutout=False):
    path = ROOT / FILES[arm, repeat]
    override = SOURCE_OVERRIDES.get((arm, repeat), {})
    canvas = override.get('canvas', SOURCE_SIZE)
    crop = override.get('crop', COMMON_CROP)
    with Image.open(path) as source:
        if source.size != canvas:
            raise ValueError(f'Unexpected source canvas: {path.name}: {source.size}')
        # The updated Full/R2 export uses its native canvas; crop only outer white margin.
        # No rotation, coordinate editing or additional transparency is applied here.
        rgb = np.asarray(source.convert('RGB').crop(crop))
        if cutout:
            # Remove only the blank canvas. The already chosen pastel atom colors stay opaque.
            alpha = np.where(np.all(rgb == 255, axis=-1), 0, 255).astype(np.uint8)
            return np.dstack((rgb, alpha))
        return rgb


def frame_data(repeat):
    with (ROOT / 'Figure_6/representative_frames.tsv').open(encoding='utf-8', newline='') as stream:
        return {r['arm']: r for r in csv.DictReader(stream, delimiter='\t')
                if r['repeat'] == repeat and r['preferred_for_mechanism'] == 'True'}


def draw_structure_row(repeat):
    import figure_font  # apply the manuscript's existing Times-style policy
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib import patheffects as pe

    height = 214.0 if repeat == 'R1' else 204.48
    fig = plt.figure(figsize=(PAGE_WIDTH / 72, height / 72), facecolor='white')

    def text(x, y, value, **kwargs):
        return fig.text(x / PAGE_WIDTH, 1 - y / height, value, **kwargs)

    text(30, 13, 'a   Representative structures' if repeat == 'R1' else 'a   Representative R2 structures',
         fontsize=9.4, fontweight='bold', color=INK, va='center')
    text(PAGE_WIDTH - 17, 13, repeat,
         fontsize=7, color=MUTED, ha='right', va='center')
    frames = frame_data(repeat)
    assert set(frames) == set(ARMS)
    for arm, label, color, center in zip(ARMS, ARM_LABELS, ARM_COLORS, CENTERS):
        cx = center * PAGE_WIDTH
        text(cx, 28 if repeat == 'R1' else 26, label, fontsize=9, fontweight='bold', color=INK, ha='center', va='center')
        fig.add_artist(Line2D([(cx - 21) / PAGE_WIDTH, (cx + 21) / PAGE_WIDTH],
                             [1 - (36 if repeat == 'R1' else 34) / height] * 2, transform=fig.transFigure,
                             color=color, lw=2.2, solid_capstyle='butt',
                             path_effects=[pe.withStroke(linewidth=3.1, foreground='#6f7f89')]))
        image_top = (42 if repeat == 'R1' else 40) + (PREVIOUS_IMAGE_HEIGHT_PT - IMAGE_HEIGHT_PT) / 2
        ax = fig.add_axes(((cx - IMAGE_WIDTH_PT / 2) / PAGE_WIDTH,
                           1 - (image_top + IMAGE_HEIGHT_PT) / height,
                           IMAGE_WIDTH_PT / PAGE_WIDTH, IMAGE_HEIGHT_PT / height))
        ax.imshow(image_for(arm, repeat), interpolation='none')
        ax.set_axis_off()
        text(cx, 193 if repeat == 'R1' else 187, 'Outside M2' if arm == 'M0' else 'Inside M2',
             fontsize=7, color=MUTED, ha='center', va='center')
        if repeat == 'R2':
            record = frames[arm]
            text(cx, 199,
                 r'$\Delta\omega_{\mathrm{max}}$ ' + f"{float(record['omega_max_deviation_deg']):.1f}°  ·  cRMS {float(record['backbone_crms_deg']):.1f}°",
                 fontsize=7, color=INK, ha='center', va='center')
    distances = ('2.08', '2.47') if repeat == 'R1' else ('2.06', '2.27')
    note = f'M0 H···O contacts:  NH₄···O₁ = {distances[0]} Å;  NH₃···O₅ = {distances[1]} Å'
    if repeat == 'R1':
        text(PAGE_WIDTH / 2, height - 8, note, fontsize=7, color=MUTED, ha='center', va='center')
    return fig


def export_page(document, out, stem, formats=('pdf', 'png', 'svg'), dpi=600):
    import fitz
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if 'pdf' in formats:
        document.save(out / f'{stem}.pdf', garbage=4, deflate=True)
    rendered = (fitz.open(out / f'{stem}.pdf') if 'pdf' in formats else
                fitz.open(stream=document.tobytes(), filetype='pdf'))
    with rendered:
        page = rendered[0]
        if 'png' in formats:
            page.get_pixmap(dpi=dpi, alpha=False).save(out / f'{stem}.png')
        if 'svg' in formats:
            (out / f'{stem}.svg').write_text(page.get_svg_image(text_as_path=True), encoding='utf-8')


def compose(row_pdf, lower_pdf, clip, row_height, gap=0):
    import fitz
    clip = fitz.Rect(clip)
    document = fitz.open()
    page = document.new_page(width=PAGE_WIDTH, height=row_height + gap + clip.height)
    with fitz.open(row_pdf) as row, fitz.open(lower_pdf) as lower:
        page.show_pdf_page(fitz.Rect(0, 0, PAGE_WIDTH, row_height), row, 0)
        page.show_pdf_page(fitz.Rect(0, row_height + gap, PAGE_WIDTH, page.rect.height), lower, 0, clip=clip)
    return document


def source_manifest():
    return {
        'source_archive': 'NMe_3D_Final.zip',
        'source_canvas_px': list(SOURCE_SIZE),
        'shared_crop_px': list(COMMON_CROP),
        'structure_width_pt_Figure6_and_S3': IMAGE_WIDTH_PT,
        'structure_height_pt_Figure6_and_S3': IMAGE_HEIGHT_PT,
        'scale_factor_vs_v3_1_1': STRUCTURE_SCALE,
        'previous_structure_width_pt': PREVIOUS_IMAGE_WIDTH_PT,
        'source_overrides': {f'{arm}_{repeat}': {k: list(v) for k, v in spec.items()}
                             for (arm, repeat), spec in SOURCE_OVERRIDES.items()},
        'physical_placement': 'Source PNGs placed at the existing point dimensions; Full/R2 uses its native updated canvas and a margin-only crop.',
        'updated_author_snapshot': 'Figure_S3/FigureS3a_FULL_R2_white.png',
        'additional_rotation_applied': False,
        'snapshots': [dict(arm=arm, repeat=repeat, file=relative,
                           sha256=hashlib.sha256((ROOT / relative).read_bytes()).hexdigest())
                      for (arm, repeat), relative in FILES.items()],
        'atoms': {'C': '#9CC4E5', 'N': '#184581', 'O': '#F9D4CA',
                  'N-methyl C': '#FAE0B7', 'H': '#FBECD7'},
        'dashed_contacts': '#FAE0B7',
        'TOC_snapshot': FILES['FULL', 'R1'],
        'CXS_received': False,
        'coordinate_or_population_changes': False,
    }
