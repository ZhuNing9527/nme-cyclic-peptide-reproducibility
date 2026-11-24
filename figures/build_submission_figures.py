#!/usr/bin/env python3
"""Export the approved HighTors submission artwork and validate its layout."""
from argparse import ArgumentParser
from pathlib import Path

from publication_artwork import build


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('figure_outputs'))
    parser.add_argument('--only', action='append', default=[], metavar='FIGURE',
                        help='Export one stem, e.g. Figure_4; repeat for several figures')
    parser.add_argument('--dpi', type=int, default=600,
                        help='PNG resolution for vector masters (default: 600)')
    parser.add_argument('--svg', action='store_true',
                        help='Also export SVG; supporting figures retain their raster source')
    args = parser.parse_args()
    if not 72 <= args.dpi <= 1200:
        parser.error('--dpi must be between 72 and 1200')
    build(Path(__file__).resolve().parent / 'publication_assets', args.out.resolve(),
          only=args.only, dpi=args.dpi, svg=args.svg)


if __name__ == '__main__':
    main()
