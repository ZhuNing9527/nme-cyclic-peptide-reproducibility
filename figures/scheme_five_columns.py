"""Restore the original five-column vector Scheme 1 and clarify test labels.

The numerical panels, molecular drawing, run matrix and endpoint markers are
copied from the approved original composition. No data marks are reconstructed.
"""
from pathlib import Path
import hashlib
import json
import fitz
from matplotlib import font_manager

WIDTH=491.76
INK=(0x21/255,0x27/255,0x2B/255)
MUTED=(0x53/255,0x5E/255,0x62/255)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_scheme(root, dest):
    root,dest=Path(root),Path(dest)
    source=root/'Scheme_1/Scheme_1.pdf'
    doc=fitz.open(source)
    page=doc[0]
    assert abs(page.rect.width-554.4)<.01
    labels=[]
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for s in line['spans']:
                if s['text'] in ['Val: not fitted','New-test RMSE (kcal/mol)']:
                    labels.append(s)
    assert len(labels)==2
    for s in labels:
        page.add_redact_annot(fitz.Rect(s['bbox'])+(-.1,-.1,.1,.1),fill=(1,1,1))
    page.apply_redactions(images=0,graphics=0,text=0)
    fs=7/(6.83/7.70)
    # Add plain text after all imported PDF panels; standard PDF fonts avoid
    # collisions with the subset fonts embedded in the historical composition.
    for x,y,text in [(5.76,180.48,'Val: retrospective'),(5.76,189.5,'not fitted'),
                     (108,165.0,'New-test RMSE'),(108,175.0,'(n = 144; kcal/mol)')]:
        page.insert_text((x,y),text,fontname='SubmissionSchemeSerif',
                         fontfile=font_manager.findfont('Liberation Serif'),
                         fontsize=fs,color=MUTED)
    # Scale the complete vector page uniformly to the manuscript's 6.83 in width.
    out=fitz.open()
    target=out.new_page(width=WIDTH,height=page.rect.height*WIDTH/page.rect.width)
    target.show_pdf_page(target.rect,doc,0,keep_proportion=True)
    out.set_metadata({'title':'Scheme 1 Fragment energetics fitting and ring ensemble validation',
                      'subject':'Five original vector columns with explicit earlier-selection and independent-test labels'})
    dest.parent.mkdir(parents=True,exist_ok=True)
    out.save(dest.with_suffix('.pdf'),garbage=4,deflate=True)
    with fitz.open(dest.with_suffix('.pdf')) as saved:
        pg=saved[0]
        pg.get_pixmap(dpi=600,alpha=False).save(dest.with_suffix('.png'))
        dest.with_suffix('.svg').write_text(pg.get_svg_image(text_as_path=True),encoding='utf-8')
        t=pg.get_text()
        for value in ['246 fit','144 selection','n = 144','2.30','2.00','14 runs','10.08','retrospective']:
            assert value in t,value
    verification={
        'status':'PASS','revision':'v3.4.3','layout':'Five-column original-vector composition',
        'source_file':'Figure_inputs_v3.1.3.zip / Figures/Scheme_1/Scheme_1.pdf',
        'source_sha256':sha(source),'numerical_marks_redrawn':False,
        'coverage_sets':['246 fitting geometries','144 earlier selection geometries'],
        'independent_test_locations_in_coverage':False,
        'independent_test_count':144,'new_test_RMSE_kcal_mol':{'pairwise':2.30,'joint':2.00},
        'Val':'72 retrospective fragment geometries; not fitted',
        'A':'Trans-plane backbone map','J':'Trans-referenced amide increment',
        'runs':14,'repeat_design':'M0 and Full: mono/di/tri × R1/R2; A only: tri × R1/R2',
        'replica_summed_time_microseconds':10.08,
        'exchange_heatmap':'Original Figure S2a panel, 14 runs, 11 adjacent-state pairs, lower-state indexing',
        'endpoint':'Equal-repeat arm-mean M2 occupancy, M0/A-only/Full',
        'canvas_points':[WIDTH,float(target.rect.height)],
        'minimum_new_label_size_at_6_83_in_pt':7,
        'current_v3_4_2_scientific_revisions_retained':True,
    }
    dest.with_name('Scheme_1_Data_Verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
    return verification


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    print(json.dumps(build_scheme(a.inputs,a.out),ensure_ascii=False,indent=2))
