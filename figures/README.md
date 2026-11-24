# Approved submission artwork

The default entry point exports the final figures corresponding to the approved
submission archive with SHA256
`cd7344d263b16834303a140e92c74fea2bd2e21bbe8df545b014540616a3b2db`.
Its input manifest is `publication_assets/manifest.json`.

## Run

From the repository root, using Python 3.10 or later:

```sh
python -m pip install -r figures/requirements.txt
python -B figures/build_submission_figures.py --out figure_outputs
```

To export Figure 4 alone, including an outlined SVG:

```sh
python -B figures/build_submission_figures.py --only Figure_4 --svg --out figure4_output
```

Outputs are in `Main/`, `Supporting/`, and `TOC/`. The complete build writes 18
individual figure PDFs, two additional combined PDFs for continued Figures S5/S6,
PNG files, the approved TOC TIFF and `artwork_build.json`. Vector-master PNGs are
rendered at 600 dpi by default (`--dpi` changes this). The 600 dpi TOC PNG uses
the approved TIFF pixels exactly. Supporting PNGs are copied
at their native resolution without interpolation. `--only` can be repeated.
`--svg` is optional; supporting SVGs embed their original raster images.

## Current layout

All main figures retain the approved embedded Arial fonts. Their canvas width
is 491.76 pt. Panel letters/titles use the approved 9 pt hierarchy; the Scheme 1
step titles are also 9 pt. Mathematical symbols and unit superscripts are retained.
No externally installed font or redistributed font file is needed for export.

Figure 4 has panels a-c above the two-column d/e and f/g rows. The top letters
have the following page coordinates (points, measured from the upper left):

| Letter | x | Baseline y | Alignment |
|---|---:|---:|---|
| a | 12.00000 | 14.00000 | Shared left margin with d/f |
| b | 189.65544 | 14.00000 | Left edge of the A-only map |
| c | 319.80792 | 14.00000 | Left edge of the Full map |

`publication_artwork.py` anchors these three text matrices while preserving the
existing embedded CID glyphs and all plotted content. It does not distribute
labels at equal intervals from the outer margin. The 25 Figure 4g cells use the
equal-repeat Full-minus-A-only differences from Table S11; the primary 30-degree
amide/45-degree backbone cell is 4.739 pp, displayed as +4.74. The tab-separated
export retains the R1/R2 values at the displayed precision of that SI table.

Figure 5 preserves the revised single row of b/c/d panels and the undistorted
circle/diamond markers. The model headings retain 9 pt lettering. TOC lettering
is at least 5.5 pt, and the 40-degree slice label retains its degree sign.

The export validates source hashes, Figure 4 label positions and all 25 heatmap
values, Figure 5 b/c marker proportions, the common main-figure width, and TOC
minimum font size and degree sign. It fails if these checks do not pass.

## Sources and reproduction scope

`publication_assets/Main/` and `publication_assets/TOC/TOC_Graphic.pdf` contain
the approved vector compositions, including the authors' unchanged molecular
views. `publication_assets/Supporting/` contains the original high-resolution
PNGs from the approved SI Word document. They are exported to PDFs at the physical
sizes recorded in the manifest, without replacing the raster artwork with a
claimed vector reconstruction. The approved TOC TIFF is preserved byte for byte.

This entry point reproduces the approved final artwork from supplied masters;
it does not claim to replot every scientific panel from raw data. It changes
no energies, populations, coordinates or sampling. Numerical recalculation is
covered by the repository's qm/, validation/, acquisition/ and md/ workflows.

The earlier scripts, frozen vector panels, numerical inputs and manual ChimeraX
images remain in their original paths. The previous submission-assembly entry
point is preserved as `build_legacy_submission.py`; its output has historical
fonts/layouts and is not the final submitted artwork. The original source folder
numbers are retained for traceability. The current numerical figure/table mapping
is in `../data/Supporting_Data.zip`.
