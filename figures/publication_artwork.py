"""Assemble approved artwork without changing numerical data or molecular views.

Main figures and the TOC preserve embedded vector graphics and fonts. Supporting
figures preserve the exact high-resolution PNGs from the approved SI Word file.
Scientific recalculation remains in qm/, validation/, acquisition/, and md/.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import fitz
from PIL import Image


# Map ownership takes priority over equally spaced letters. The first letter
# shares the 12 pt margin with d/f; b/c sit over their own map's left edge.
FIGURE4_MAP_LABELS = {
    'a': (12.0, 14.0),
    'b': (189.6554412841797, 14.0),
    'c': (319.80792236328125, 14.0),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spans(page):
    return [span for block in page.get_text('dict')['blocks']
            for line in block.get('lines', []) for span in line['spans']
            if span['text'].strip()]


def anchor_figure4(source, destination, positions=None):
    """Place a/b/c by changing only their text matrices, retaining embedded fonts.

    Reusing the approved CID glyphs avoids substituting system fonts. No data
    marks, plotted text, clipping paths or coordinates of other panels change.
    """
    positions = FIGURE4_MAP_LABELS if positions is None else positions
    codes = {'0044': 'a', '0045': 'b', '0046': 'c'}
    pattern = re.compile(
        rb'(BT\s*/PublicationArialBold\s+9\s+Tf\s+1\s+0\s+0\s+1\s+)'
        rb'(-?[\d.]+)\s+(-?[\d.]+)'
        rb'(\s+Tm\s*\[\s*<(0044|0045|0046)>\s*\]\s*TJ\s+ET)')
    seen = []
    changed = False
    with fitz.open(source) as document:
        page = document[0]

        def replace(match):
            letter = codes[match.group(5).decode()]
            seen.append(letter)
            x, y = positions[letter]
            pdf_y = page.rect.height - y
            if abs(float(match.group(2)) - x) < 0.0001 and abs(float(match.group(3)) - pdf_y) < 0.0001:
                return match.group(0)
            return match.group(1) + f'{x:.8f} {pdf_y:.8f}'.encode() + match.group(4)

        for xref in page.get_contents():
            original = document.xref_stream(xref)
            revised = pattern.sub(replace, original)
            if revised != original:
                document.update_stream(xref, revised)
                changed = True
        if sorted(seen) != ['a', 'b', 'c']:
            raise ValueError('The approved Figure 4 top-label objects were not found exactly once')
        if changed:
            document.save(destination, garbage=4, deflate=True)
    if not changed:
        shutil.copyfile(source, destination)


def validate_figure4(page, assets):
    top = {span['text']: span for span in spans(page)
           if span['text'] in FIGURE4_MAP_LABELS and span['origin'][1] < 30}
    if set(top) != set(FIGURE4_MAP_LABELS):
        raise ValueError('Figure 4 must have one a/b/c label over the maps')
    for letter, (x, y) in FIGURE4_MAP_LABELS.items():
        span = top[letter]
        if (abs(span['origin'][0] - x) > 0.001 or abs(span['origin'][1] - y) > 0.001
                or abs(span['size'] - 9) > 0.001 or 'Arial' not in span['font']):
            raise ValueError(f'Figure 4 {letter} has the wrong position or font')
    with (assets / 'figure4g_cutoffs.tsv').open(newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    if len(rows) != 25:
        raise ValueError('Figure 4g requires all 25 existing Table S11 cells')
    rows.sort(key=lambda row: (int(row['amide_deg']), int(row['backbone_deg'])))
    expected = [f"{float(row['equal_repeat_mean_pp']):+.2f}" for row in rows]
    text = page.get_text(clip=fitz.Rect(294, 380, 445, 500)).replace('−', '-')
    observed = re.findall(r'[+-]\d+\.\d{2}', text)
    if observed != expected:
        raise ValueError('Figure 4g labels differ from the equal-repeat Table S11 means')
    primary = [row for row in rows if row['amide_deg'] == '30' and row['backbone_deg'] == '45']
    if len(primary) != 1 or abs(float(primary[0]['equal_repeat_mean_pp']) - 4.739) > 0.00001:
        raise ValueError('Figure 4g primary cell must be +4.739 pp, printed as +4.74')
    # Locate the actual top-map vertical axes rather than spacing the letters.
    axes = []
    for drawing in page.get_drawings():
        for item in drawing['items']:
            if item[0] == 'l':
                start, end = item[1], item[2]
                if abs(start.x - end.x) < 0.01 and min(start.y, end.y) < 26 and max(start.y, end.y) > 110:
                    axes.append(start.x)
    for letter in ['b', 'c']:
        if not any(abs(value - top[letter]['origin'][0]) < 0.05 for value in axes):
            raise ValueError(f'Figure 4 {letter} is not over its own map axis')


def validate_vector(stem, page, assets):
    if stem.startswith('Figure_') and not stem.startswith('Figure_S'):
        if abs(page.rect.width - 491.76) > 0.01:
            raise ValueError(f'{stem} has the wrong main-figure canvas width')
    if stem == 'Figure_4':
        validate_figure4(page, assets)
    elif stem == 'Figure_5':
        # The 15 circles and 15 diamonds in the revised b/c row have square
        # bounds. Other paths include clipped historical input objects.
        boxes = [drawing['rect'] for drawing in page.get_drawings() if drawing['type'] == 'fs']
        circles = sum(abs(box.width - 3) < 0.005 and abs(box.height - 3) < 0.005 for box in boxes)
        diamonds = sum(abs(box.width - 4.24264) < 0.005
                       and abs(box.height - 4.24264) < 0.005 for box in boxes)
        if circles != 15 or diamonds != 15:
            raise ValueError('Figure 5 b/c markers must retain their circular/diamond proportions')
    elif stem == 'TOC_Graphic':
        if min(span['size'] for span in spans(page)) < 5.49:
            raise ValueError('TOC lettering must remain at least 5.5 pt')
        lines = defaultdict(list)
        for block in page.get_text('rawdict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    for char in span['chars']:
                        lines[round(char['origin'][1], 1)].append(char)
        if not any('40°' in ''.join(char['c'] for char in sorted(chars, key=lambda char: char['origin'][0]))
                   for chars in lines.values()):
            raise ValueError('The degree sign in the TOC 40° heading is missing')


def build(assets, output, only=(), dpi=600, svg=False):
    assets, output = Path(assets), Path(output)
    manifest = json.loads((assets / 'manifest.json').read_text())
    for relative, expected in manifest['sha256'].items():
        if sha256(assets / relative) != expected:
            raise ValueError(f'Approved artwork input changed: {relative}')
    selected = [entry for entry in manifest['entries']
                if not only or any(entry['stem'] == name or entry['stem'].startswith(name + '_part')
                                   for name in only)]
    if not selected or any(not any(entry['stem'] == name or entry['stem'].startswith(name + '_part')
                                   for entry in selected) for name in only):
        raise ValueError(f'Unknown figure selection: {list(only)}')
    produced = []
    for entry in selected:
        stem = entry['stem']
        source = assets / entry['source']
        folder = output / entry['group']
        folder.mkdir(parents=True, exist_ok=True)
        pdf = folder / f'{stem}.pdf'
        png = folder / f'{stem}.png'
        if entry['kind'] == 'vector_pdf':
            if stem == 'Figure_4':
                anchor_figure4(source, pdf)
            else:
                shutil.copyfile(source, pdf)
            with fitz.open(pdf) as document:
                validate_vector(stem, document[0], assets)
                document[0].get_pixmap(dpi=dpi, alpha=False).save(png)
                if svg:
                    pdf.with_suffix('.svg').write_text(document[0].get_svg_image(text_as_path=True))
        else:
            shutil.copyfile(source, png)
            with fitz.open() as document:
                page = document.new_page(width=entry['width_pt'], height=entry['height_pt'])
                page.insert_image(page.rect, filename=str(source), keep_proportion=True)
                document.save(pdf, garbage=4, deflate=True)
                if svg:
                    pdf.with_suffix('.svg').write_text(page.get_svg_image(text_as_path=True))
        if 'tiff_source' in entry:
            tiff = pdf.with_suffix('.tif')
            shutil.copyfile(assets / entry['tiff_source'], tiff)
            # Preserve the submitted pixels at the approved resolution. Image
            # edge interpolation in PDF renderers can vary between sessions.
            if dpi == 600:
                with Image.open(tiff) as image:
                    image.convert('RGB').save(png, dpi=(600, 600))
            produced.append(tiff)
        produced.extend([pdf, png])
        print('Wrote', pdf.relative_to(output))
    for stem in ['Figure_S5', 'Figure_S6']:
        parts = [output / 'Supporting' / f'{stem}_part{number}.pdf' for number in [1, 2]]
        if all(any(entry['stem'] == part.stem for entry in selected) for part in parts):
            combined = output / 'Supporting' / f'{stem}.pdf'
            with fitz.open() as document:
                for part in parts:
                    with fitz.open(part) as source:
                        document.insert_pdf(source)
                document.save(combined, garbage=4, deflate=True)
            produced.append(combined)
    report = {'approved_submission_sha256': manifest['approved_submission_sha256'],
              'figure4_map_labels_pt': FIGURE4_MAP_LABELS,
              'vector_png_dpi': dpi,
              'supporting_png_resolution': 'Original approved SI Word images; no resampling',
              'outputs': {p.relative_to(output).as_posix(): sha256(p) for p in produced}}
    (output / 'artwork_build.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Approved artwork and layout checks complete:', output)
