"""Build the v3.4.5 artwork from the frozen v3.1.3 figure PDFs and numerical data.

Usage: python build_figures.py --inputs path/to/Figures --out path/to/Figures
With the repository, omit --inputs to use the adjacent frozen inputs directory.
"""
from pathlib import Path
from io import BytesIO
from tempfile import TemporaryDirectory
import argparse
import csv
import shutil
import zipfile

import fitz
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
import numpy as np

W = 491.76
INK = (0.09, 0.16, 0.20)
TEAL = (0.13, 0.34, 0.48)
GREY = (0.31, 0.37, 0.40)
RULE = (.78, .83, .86)
FONT = font_manager.findfont('Liberation Serif')
BOLD = font_manager.findfont(font_manager.FontProperties(family='Liberation Serif', weight='bold'))
R = fitz.Rect

def paper(h):
    doc = fitz.open()
    return doc, doc.new_page(width=W, height=h)

def label(pg, x, baseline, text, size=8.4, bold=False, color=INK):
    pg.insert_text((x,baseline), text, fontname='NMe342SerifB' if bold else 'NMe342SerifR',
                   fontfile=BOLD if bold else FONT, fontsize=size, color=color)

def erase(pg, box):
    pg.add_redact_annot(R(box), fill=(1,1,1))
    pg.apply_redactions(images=0, graphics=0, text=0)

def copy_region(dst, source, box, target):
    # Retain vector marks and original aspect ratio.
    dst.show_pdf_page(R(target), source, 0, clip=R(box), keep_proportion=True)

def export(doc, dest):
    dest.parent.mkdir(exist_ok=True, parents=True)
    doc.save(dest.with_suffix('.pdf'), garbage=4, deflate=True)
    with fitz.open(dest.with_suffix('.pdf')) as saved:
        pg = saved[0]
        pg.get_pixmap(dpi=400, alpha=False).save(dest.with_suffix('.png'))
        dest.with_suffix('.svg').write_text(pg.get_svg_image(text_as_path=True),encoding='utf8')

def old(root, key, suffix=''):
    return fitz.open(root/key/(key+suffix+'.pdf'))

def unchanged(root, source, dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/source/(source+'.pdf'),dest.with_suffix('.pdf'))
    with fitz.open(dest.with_suffix('.pdf')) as saved:
        pg = saved[0]
        pg.get_pixmap(dpi=600, alpha=False).save(dest.with_suffix('.png'))
        dest.with_suffix('.svg').write_text(pg.get_svg_image(text_as_path=True),encoding='utf8')

def scheme(root, dest):
    from scheme_five_columns import build_scheme
    build_scheme(root, dest)

def ala(root,dest):
    doc,pg=paper(520)
    with old(root,'Figure_1') as source:
        erase(source[0],(49,485,224,503))
        copy_region(pg,source,source[0].rect,(0,0,W,504))
    with (root/'Figure_1'/'holdout_metrics.tsv').open() as f:
        allrow=next(csv.DictReader(f,delimiter='\t'))
    lo,hi=(float(allrow[k]) for k in ['paired_CI95_low_kcal_mol','paired_CI95_high_kcal_mol'])
    label(pg,52,492,'Cis slice (9): RMS = 3.019 kcal/mol',8.0)
    label(pg,52,504,'All non-trans cells (45): RMS = 3.523 → 2.961 kcal/mol',8.0)
    label(pg,328,514,f'RMSE gain: 0.300 (95% CI {lo:.3f}–{hi:.3f}) kcal/mol',7.8)
    export(doc,dest)

def occupancy_paths(root):
    with (root/'Figure_2'/'peptide_repeat_endpoints.tsv').open() as f:
        data=list(csv.DictReader(f,delimiter='\t'))
    fig,axes=plt.subplots(1,3,figsize=(W/72,2.43),sharex=True,sharey=True)
    fig.subplots_adjust(left=.115,right=.975,top=.71,bottom=.235,wspace=.18)
    fig.text(.024,.965,'c   Amide-window occupancy and backbone match',fontsize=10.2,
             fontweight='bold',va='top')
    fig.text(.024,.865,'Arrows run from M0 to Full.  Grey curves mark M2 occupancies of 10%, 40% and 80%.',
             color='#40535b',fontsize=8.15)
    xx=np.linspace(.2,100,500)
    for ax,system in zip(axes,['mono-NMe','di-NMe','tri-NMe']):
        for z in [10,40,80]:
            ys=100*z/xx;keep=ys<=100
            ax.plot(xx[keep],ys[keep],color='#c5ced2',lw=.65,zorder=1)
        coords=[]
        for model in ['M0','Full']:
            rows=[r for r in data if (r['system'],r['model'])==(system,model)]
            p,m=np.mean([[float(r['M1_pct']),float(r['M2_pct'])] for r in rows],axis=0)
            coords.append((p,100*m/p))
        ax.annotate('',xy=coords[1],xytext=coords[0],
                    arrowprops=dict(arrowstyle='-|>',shrinkA=2,shrinkB=3,color='#183747',lw=1.5))
        for (x,y),color in zip(coords,['#efd0c7','#164b75']):
            ax.scatter(x,y,s=26,facecolor=color,edgecolor='#415663',linewidth=.65,zorder=3)
        ax.set(xlim=(-3,105),ylim=(-3,105),xticks=[0,50,100],yticks=[0,50,100])
        ax.set_title(system,fontsize=8.7,fontweight='bold',pad=5)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(length=2,width=.55,labelsize=7.5)
    axes[0].set_ylabel('Backbone match within\nwindow (%)',fontsize=8.3)
    fig.text(.545,.045,'All five amides in window (%)',ha='center',fontsize=8.6)
    stream=BytesIO();fig.savefig(stream,format='pdf');plt.close(fig)
    return fitz.open(stream=stream.getvalue(),filetype='pdf')

def ring_series(root,dest):
    doc,pg=paper(570)
    with old(root,'Figure_2') as orig:
        copy_region(pg,orig,(0,0,W,394),(0,0,W,394))
    with occupancy_paths(root) as sub:
        copy_region(pg,sub,sub[0].rect,(0,394,W,569))
    export(doc,dest)

def relabel_first(pg,x,y,letter):
    erase(pg,(x-.5,y-.3,x+6.2,y+11.1))
    label(pg,x,y+9.0,letter,10.3,True)

def landscapes(root,dest):
    from promote_s9 import landscapes_with_blocks
    landscapes_with_blocks(root,dest)

def extra_diagnostics(root,dest):
    from promote_s9 import repeat_diagnostics
    repeat_diagnostics(root,dest)

def structures(root,dest):
    from promote_s9 import structures_with_rmsd
    structures_with_rmsd(root,dest)

def classes(root,dest):
    doc,pg=paper(388)
    with old(root,'Figure_S4') as source:
        copy_region(pg,source,source[0].rect,(0,0,W,197.6))
    label(pg,7,18,'a',10.0,True)
    with old(root,'Figure_2','_c') as source:
        relabel_first(source[0],12,13.08,'b')
        copy_region(pg,source,source[0].rect,(78,197,372,387))
    export(doc,dest)

def rama(root,dest):
    unchanged(root,'Figure_S6',dest/'Figure_S6_part1')
    with old(root,'Figure_5') as source:
        pg=source[0]
        for block in pg.get_text('dict')['blocks']:
            for ln in block.get('lines',[]):
                s=''.join(k['text'] for k in ln['spans']).strip()
                if s in ('a','b','c') and ln['bbox'][1]<100:
                    x,y=ln['bbox'][:2]
                    relabel_first(pg,x,y,{'a':'c','b':'d','c':'e'}[s])
        export(source,dest/'Figure_S6_part2')

def tri_roe(root, dest):
    with Path(__file__).with_name('ALL20_CONTROL_TEST_COMPARISON_v1.tsv').open() as f:
        rows=list(csv.DictReader(f,delimiter='\t'))
    assert len(rows)==len({r['restraint_id'] for r in rows})==20
    def col(k):
        return np.array([float(r[k]) for r in rows])
    lo,hi=col('lower_bound_A'),col('upper_bound_A')
    m0,full=col('CONTROL_r_eff_A'),col('TEST_r_eff_A')
    violations=[]
    for model,ds,expected in [('CONTROL',m0,11),('TEST',full,10)]:
        v=np.maximum.reduce([lo-ds,ds-hi,np.zeros(len(ds))])
        np.testing.assert_allclose(v,col(model+'_violation_A'),rtol=0,atol=1e-12)
        assert (v==0).sum()==expected
        violations.append(v)
    change=violations[1]-violations[0]
    np.testing.assert_allclose(change,col('violation_A_TEST_minus_CONTROL'),rtol=0,atol=1e-12)
    assert ((change<0).sum(),((violations[0]==0)&(violations[1]==0)).sum(),(change>0).sum())==(5,9,6)
    f=plt.figure(figsize=(6.83,4.80),facecolor='white')
    f.text(.04,.953,'d   tri-NMe',fontsize=11,fontweight='bold')
    f.text(.04,.900,'Mean violation  %.3f → %.3f Å     |     Within bounds  11/20 → 10/20'%tuple(v.mean() for v in violations),fontsize=9)
    ax=f.add_axes([.255,.116,.383,.635]);bx=f.add_axes([.715,.116,.252,.635],sharey=ax)
    y=np.arange(20)
    lbl=[f"{i:02}   "+' · '.join(r[c].split(':')[1][-1]+r[c].split(':')[2] for c in ['source_endpoint_1','source_endpoint_2']) for i,r in enumerate(rows,1)]
    for a in (ax,bx):
        a.set_ylim(19.6,-.6);a.spines[['top','right','left']].set_visible(False)
        a.tick_params(axis='y',length=0);a.tick_params(axis='x',length=3,labelsize=8)
        for n in [4,9,14]:a.axhline(n+.5,c='#dee6eb',lw=.5)
    ax.set_yticks(y,lbl,fontsize=8.4);bx.tick_params(labelleft=False)
    ax.hlines(y,lo,hi,color='#9cc4e5',lw=2.7,label='Source interval')
    ax.scatter(m0,y-.14,s=18,marker='o',color='#f9d4ca',edgecolor='#6f7f89',lw=.55,label='M0',zorder=3)
    ax.scatter(full,y+.14,s=16,marker='D',color='#184581',edgecolor='white',lw=.35,label='Full',zorder=3)
    ax.set(xlim=(1.42,7.03),xticks=[2,4,6],xlabel='Effective distance (Å)')
    bx.axvline(0,lw=.7,c='#91a3b0')
    colors=['#1b6dc0' if x<0 else '#f9d4ca' if x>0 else '#a9b6be' for x in change]
    bx.hlines(y,0,change,color=colors,lw=1.5)
    bx.scatter(change,y,s=13,c=colors,edgecolor='#73818a',lw=.35)
    bx.set(xlim=(-1.45,.83),xticks=[-1,0,.5],xlabel='Δ violation (Å)')
    f.legend(*ax.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.55,.827),frameon=False,ncol=3,fontsize=8.5)
    for fmt in ['pdf','svg','png']:
        f.savefig(dest.with_suffix('.'+fmt),dpi=400)
    plt.close(f)

def build(root,out):
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':9,'pdf.fonttype':42,
                         'axes.edgecolor':'#718087','axes.linewidth':.65,'text.color':'#1b303b'})
    main=out/'Main';si=out/'Supporting';toc=out/'TOC'
    scheme(root,main/'Scheme_1');ala(root,main/'Figure_1')
    unchanged(root,'Figure_7',main/'Figure_2');ring_series(root,main/'Figure_3')
    landscapes(root,main/'Figure_4');structures(root,main/'Figure_5')
    for i in [1,2,3,5,7,8]:
        unchanged(root,f'Figure_S{i}',si/f'Figure_S{i}')
    tri_roe(root,si/'Figure_S5_part2')
    classes(root,si/'Figure_S4');rama(root,si);extra_diagnostics(root,si/'Figure_S9')
    # Supply a complete two-page PDF for each continued SI figure.
    for ext in ['.pdf','.png','.svg']:
        (si/('Figure_S5'+ext)).replace(si/('Figure_S5_part1'+ext))
    for number in [5,6]:
        with fitz.open() as joined:
            for part in [1,2]:
                with fitz.open(si/f'Figure_S{number}_part{part}.pdf') as pages:
                    joined.insert_pdf(pages)
            joined.save(si/f'Figure_S{number}.pdf',garbage=4,deflate=True)
    toc.mkdir(exist_ok=True,parents=True)
    for f in (root/'TOC').glob('NMe_TOC_energy_surface_3D.*'):
        if f.suffix in {'.pdf','.png','.svg','.tif'}:shutil.copyfile(f,toc/f.name)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path)
    ap.add_argument('--out',type=Path,default=Path('Figures'))
    args=ap.parse_args()
    build(args.inputs or Path(__file__).resolve().parent/'inputs', args.out)
