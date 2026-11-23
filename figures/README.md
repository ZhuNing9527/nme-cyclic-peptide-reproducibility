# Final figure reproduction

These files reproduce the artwork for manuscript v3.4.5. The five-column
Scheme 1, centered compact Figure 4d and Figure 5d, and supplied manual molecular
views are retained. Figure 5/S3 distance annotations use 6.5 pt lettering.
The TOC uses 9 pt Backbone–amide coupling lettering and a smooth numerical
surface with an open coordinate frame and no dark surface rim.

Install Python dependencies from figures/requirements.txt. The fonts
Liberation Serif and Liberation Sans must be available to Matplotlib; on
Debian/Ubuntu these are supplied by the fonts-liberation package.

From the repository root:

```sh
python -m pip install -r figures/requirements.txt
python -B figures/build_submission_figures.py --out figure_outputs
```

Output folders are Main/, Supporting/ and TOC/. Main and supporting figures
have vector PDF, PNG and SVG exports. Continued figures S5/S6 additionally have
combined PDFs. TOC includes a 600 dpi RGB TIFF at 3.25 by 1.75 inches.

The composition combines the distributed frozen vector panels and numerical
tables with the authors' original ChimeraX PNGs. Molecular coordinates and
view angles are not regenerated or rotated. The original six molecular PNGs
are in inputs/Figure_6/ and inputs/Figure_S3/. Final label refinement affects
only the two pre-existing M0 distance labels in the composed figures.

The frozen inputs/Figure_1 through Figure_7 and Figure_S1 through Figure_S8
folder names retain their source identities. They are assembly inputs, not the
current manuscript figure numbering; the generated Main/ and Supporting/
folders use current numbering. Numerical data and the current figure/table
mapping are also distributed in ../data/Supporting_Data.zip.

TOC/amide_correction_slice.tsv is the unchanged 151 by 151 fitted J(phi,omega)
grid at psi = 40 degrees. This local energy correction is not a cyclic-peptide
free-energy surface. The script plots the grid numerically; no image-generation
service, new fit, QM calculation, MD trajectory or density estimate is used.
