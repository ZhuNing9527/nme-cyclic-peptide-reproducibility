#!/usr/bin/env python3
"""Reproduce the submitted v3.4.5 artwork from distributed frozen inputs."""
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import subprocess
import sys

import fitz
from matplotlib import font_manager

from build_figures import build
from distance_labels import refine

HERE = Path(__file__).resolve().parent


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('figure_outputs'))
    args = parser.parse_args()
    output = args.out.resolve()
    for family in ['Liberation Serif', 'Liberation Sans']:
        font_manager.findfont(family, fallback_to_default=False)

    build(HERE / 'inputs', output)
    # Keep the chosen molecular views and all data marks. Refine only the two
    # existing distance annotations in the representative M0 snapshots.
    with TemporaryDirectory(prefix='nme_figure_labels_') as temporary:
        temporary = Path(temporary)
        for group, stem in [('Main', 'Figure_5'), ('Supporting', 'Figure_S3')]:
            source = output / group / f'{stem}.pdf'
            refined = temporary / f'{stem}.pdf'
            refine(source, refined)
            for extension in ['.pdf', '.png', '.svg']:
                shutil.copyfile(refined.with_suffix(extension), source.with_suffix(extension))

        # Uniform scaling preserves the vector data and matches manuscript width.
        path = output / 'Main/Figure_2.pdf'
        scaled = temporary / 'Figure_2.pdf'
        with fitz.open(path) as source, fitz.open() as target:
            width = 6.72 * 72
            page = target.new_page(width=width, height=source[0].rect.height * width / source[0].rect.width)
            page.show_pdf_page(page.rect, source, 0)
            target.save(scaled, garbage=4, deflate=True)
            page.get_pixmap(dpi=600, alpha=False).save(path.with_suffix('.png'))
            path.with_suffix('.svg').write_text(page.get_svg_image(text_as_path=True))
        shutil.copyfile(scaled, path)

    subprocess.run([sys.executable, '-B', str(HERE / 'TOC/plot_TOC.py'),
                    '--out', str(output / 'TOC'), '--look', 'smooth'], check=True)
    # Assembly diagnostics are internal to the build, not manuscript attachments.
    for path in output.rglob('*_Verification.json'):
        path.unlink()
    print('Final main figures, supporting figures and TOC written to', output)


if __name__ == '__main__':
    main()
