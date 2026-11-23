#!/usr/bin/env python3
"""Enlarge only the two raster distance labels in Figure 5 and Figure S3."""
from pathlib import Path
from io import BytesIO
from hashlib import sha256
import argparse
import json
from matplotlib import font_manager

import fitz
import numpy as np
from PIL import Image
from docx import Document
from docx.oxml.ns import qn


def image_pixels(doc, info):
    return np.asarray(Image.open(BytesIO(doc.extract_image(info["xref"])["image"])).convert("RGB"))


def refine(source, target):
    doc = fitz.open(source)
    page = doc[0]
    original_paths = repr(page.get_drawings())
    original_text = page.get_text()
    original_images = page.get_image_info(xrefs=True)
    selected = min(original_images, key=lambda im: im["bbox"][0])
    rect = fitz.Rect(selected["bbox"])
    raw = Image.fromarray(image_pixels(doc, selected))
    flip = selected["transform"][3] < 0
    displayed = raw.transpose(Image.Transpose.FLIP_TOP_BOTTOM) if flip else raw
    before = np.asarray(displayed)
    after = before.copy()
    if source.stem == "Figure_5":
        regions = [(58.8, 60.1, 73.4, 65.5), (37.3, 71.3, 52.0, 76.8)]
        placements = [(56., 64.8, "2.08 Å"), (30.5, 76.2, "2.47 Å")]
    else:
        regions = [(64.5, 58.0, 81.0, 63.4), (37.0, 74.5, 55.0, 80.1)]
        placements = [(59.5, 63.0, "2.06 Å"), (30., 80., "2.27 Å")]
    allowed = np.zeros(before.shape[:2], dtype=bool)
    removed = []
    for x0, y0, x1, y1 in regions:
        xa, xb = int(x0 * raw.width / rect.width), int(np.ceil(x1 * raw.width / rect.width))
        ya, yb = int(y0 * raw.height / rect.height), int(np.ceil(y1 * raw.height / rect.height))
        allowed[ya:yb, xa:xb] = True
        rgb = after[ya:yb, xa:xb].astype(np.int16)
        mask = (rgb[:, :, 0] < 248) & (rgb[:, :, 2] - rgb[:, :, 0] > 1.5) & (rgb[:, :, 1] - rgb[:, :, 0] > .5)
        removed.append(int(mask.sum()))
        rgb[mask] = 255
        after[ya:yb, xa:xb] = rgb.astype(np.uint8)
    assert min(removed) > 50
    assert np.array_equal(before[~allowed], after[~allowed])
    altered = Image.fromarray(after)
    if flip:
        altered = altered.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    stream = BytesIO()
    altered.save(stream, format="PNG")
    page.replace_image(selected["xref"], stream=stream.getvalue())
    font = font_manager.findfont("Liberation Sans", fallback_to_default=False)
    page.insert_font(fontname="distance_annotation", fontfile=font)
    for x, y, text in placements:
        page.insert_text((rect.x0 + x, rect.y0 + y), text, fontname="distance_annotation", fontsize=6.5,
                         color=(24 / 255, 43 / 255, 54 / 255))
    target.write_bytes(doc.tobytes(garbage=4, deflate=True))
    result = fitz.open(target)
    assert repr(result[0].get_drawings()) == original_paths, "Original plot paths changed"
    assert result[0].get_text().startswith(original_text), "Original vector labels changed"
    result_images = result[0].get_image_info(xrefs=True)
    assert [im["bbox"] for im in original_images] == [im["bbox"] for im in result_images]
    for a, b in zip(original_images, result_images):
        if a["bbox"] != selected["bbox"]:
            assert np.array_equal(image_pixels(doc, a), image_pixels(result, b))
    pix = result[0].get_pixmap(dpi=600, alpha=False)
    Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(target.with_suffix(".png"), dpi=(600, 600), optimize=True)
    target.with_suffix(".svg").write_text(result[0].get_svg_image(text_as_path=True))
    return {"figure": source.stem, "distance_label_size_pt": 6.5,
            "distance_values": [v[2] for v in placements], "original_plot_paths_unchanged": True,
            "original_vector_labels_unchanged": True, "image_positions_and_sizes_unchanged": True,
            "molecular_pixels_outside_distance_text_regions_unchanged": True,
            "other_molecular_images_unchanged": True, "removed_old_label_pixels": removed}


