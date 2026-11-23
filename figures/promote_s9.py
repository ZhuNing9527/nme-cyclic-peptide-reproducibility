"""Compose v3.4.5 with compact, taller half-width time-block and RMSD panels."""
from pathlib import Path
from io import BytesIO
import csv
import hashlib
import json
import fitz
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, patheffects as pe, font_manager
from matplotlib.lines import Line2D
import numpy as np

W = 491.76
PANEL_W = 245.88
BLOCK_H = 190
RMSD_H = 190
INK = '#182B36'
EDGE = '#6F7F89'
GRID = '#E0E8ED'
AXIS = '#8FA2AC'
COLORS = {'M0':'#F9D4CA', 'A only':'#4396CD', 'Full':'#184581'}
MODELS = tuple(COLORS)
R = fitz.Rect
PRINT_SERIF = font_manager.findfont('Liberation Serif')
PRINT_SERIF_BOLD = font_manager.findfont(font_manager.FontProperties(family='Liberation Serif', weight='bold'))

def setup():
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,
                         'pdf.fonttype':42,'svg.fonttype':'path','text.color':INK,
                         'axes.labelcolor':INK,'axes.edgecolor':AXIS,'axes.linewidth':.65,
                         'xtick.color':INK,'ytick.color':INK,'lines.scale_dashes':False})

def table(root, name):
    with (root/'Figure_4'/name).open() as f:
        return list(csv.DictReader(f,delimiter='\t'))

def pdf_figure(fig):
    b=BytesIO();fig.savefig(b,format='pdf');plt.close(fig)
    return fitz.open(stream=b.getvalue(),filetype='pdf')

def export(doc,dest):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    doc.save(dest.with_suffix('.pdf'),garbage=4,deflate=True)
    with fitz.open(dest.with_suffix('.pdf')) as saved:
        saved[0].get_pixmap(dpi=600,alpha=False).save(dest.with_suffix('.png'))
        dest.with_suffix('.svg').write_text(saved[0].get_svg_image(text_as_path=True))
    doc.close()

def text(pg,x,y,s,size=8,bold=False,rotate=0,color=(.09,.16,.20)):
    pg.insert_text((x,y),s,fontname='SubmissionSerifBold' if bold else 'SubmissionSerifRegular',
                   fontfile=PRINT_SERIF_BOLD if bold else PRINT_SERIF, fontsize=size,
                   color=color,rotate=rotate)

def erase(pg,box):
    pg.add_redact_annot(R(box),fill=(1,1,1))
    pg.apply_redactions(images=0,graphics=0,text=0)

def paste(pg,src,box,dest):
    pg.show_pdf_page(R(dest),src,0,clip=R(box),keep_proportion=True)

def bar(pg,top,bottom):
    # Same discrete scale as the frozen PMF maps; no map values are recomputed.
    levels=[0,.5,1,1.5,2,2.5,3,4]
    colors=['#184581','#1B6DC0','#4396CD','#9CC4E5','#F9D4CA','#FAE0B7','#FBECD7']
    for lo,hi,c in zip(levels[:-1],levels[1:],colors):
        rgb=tuple(int(c[i:i+2],16)/255 for i in (1,3,5))
        a=bottom-(hi/4)*(bottom-top);b=bottom-(lo/4)*(bottom-top)
        pg.draw_rect(R(452,a,460,b),color=None,fill=rgb,width=0)
    pg.draw_rect(R(452,top,460,bottom),color=(.55,.63,.68),width=.45)
    for val in range(5):
        y=bottom-val*(bottom-top)/4
        pg.draw_line((460,y),(462,y),color=(.35,.43,.49),width=.5)
        text(pg,464,y+2.3,str(val),7.5)
    text(pg,443,top-18,'Relative F',8,True)
    text(pg,443,top-7,'(kcal/mol)',7.5)

def isolated_style(function):
    def wrapped(*args,**kwargs):
        with plt.rc_context():
            return function(*args,**kwargs)
    return wrapped

@isolated_style
def m2_blocks(root):
    setup();rows=table(root,'tri_NMe_M2_blocks.tsv')
    series={}
    for model in MODELS:
        for rep in ['R1','R2']:
            group=sorted((r for r in rows if (r['model'],r['repeat'])==(model,rep)),key=lambda r:int(r['block']))
            assert len(group)==5
            assert all(int(r['total_frames'])==10000 and abs(100*int(r['in_M2_frames'])/10000-float(r['M2_pct']))<1e-10 for r in group)
            series[model,rep]=np.array([float(r['M2_pct']) for r in group])
    assert all(np.all(series['Full',r]>series['A only',r]) for r in ['R1','R2'])
    fig=plt.figure(figsize=(PANEL_W/72,BLOCK_H/72),facecolor='white')
    high=fig.add_axes([.19,.43,.78,.37])
    low=fig.add_axes([.19,.18,.78,.17],sharex=high)
    for ax,models,lim,ticks in [(high,['A only','Full'],(85,94),[86,88,90,92]),(low,['M0'],(5,10),[6,8])]:
        ax.set(xlim=(0,50),ylim=lim,yticks=ticks)
        ax.set_axisbelow(True);ax.grid(axis='y',color=GRID,lw=.5)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=7.5,length=2.2,width=.6,pad=2)
        for model in models:
            for rep in ['R1','R2']:
                ax.plot(np.arange(5,50,10),series[model,rep],color=COLORS[model],
                        ls='-' if rep=='R1' else (0,(3.6,2.15)),lw=.85,
                        marker='o' if rep=='R1' else 'D',ms=3.1,mew=.55,
                        mec=EDGE if model=='M0' else COLORS[model],
                        path_effects=[pe.withStroke(linewidth=1.2,foreground=EDGE)] if model=='M0' else None)
    high.spines['bottom'].set_visible(False)
    high.tick_params(axis='x',bottom=False,labelbottom=False)
    low.set_xticks(np.arange(5,50,10),['0–10','10–20','20–30','30–40','40–50'])
    low.set_xlabel('Production interval (ns)',fontsize=8,labelpad=3)
    fig.text(.048,.49,'M2 occupancy (%)',rotation=90,va='center',ha='center',fontsize=8)
    for ax,offset in [(high,0),(low,1)]:
        ax.plot([-.005,.005],[offset-.025,offset+.025],transform=ax.transAxes,c=AXIS,lw=.75,clip_on=False)
    fig.text(.19,.975,'d   M2 across 10 ns blocks',fontsize=9.5,fontweight='bold',va='top')
    handles=[Line2D([],[],color=COLORS[m],marker='o',ls='none',ms=3.4,mec=EDGE if m=='M0' else COLORS[m],label=m) for m in MODELS]
    repeats=[Line2D([],[],color=INK,ls='-' if r=='R1' else (0,(3.6,2.15)),lw=.85,marker='o' if r=='R1' else 'D',ms=2.8,label=r) for r in ['R1','R2']]
    fig.legend(handles=handles+repeats,ncol=5,frameon=False,loc='upper right',bbox_to_anchor=(.99,.91),
               fontsize=7.5,columnspacing=.6,handletextpad=.3,handlelength=1.2)
    return pdf_figure(fig)

@isolated_style
def rmsd(root):
    setup();rows=table(root,'tri_NMe_RMSD_KDE.tsv');quant=table(root,'tri_NMe_RMSD_quantiles.tsv')
    assert len(rows)==2700 and len(quant)==6
    scale=max(float(r['density_per_A']) for r in rows)
    fig=plt.figure(figsize=(PANEL_W/72,RMSD_H/72),facecolor='white')
    ax=fig.add_axes([.20,.16,.75,.64])
    for j,model in enumerate(MODELS):
        for rep in ['R1','R2']:
            group=[r for r in rows if (r['model'],r['repeat'])==(model,rep)]
            ys=np.array([float(r['RMSD_A']) for r in group])
            density=np.array([float(r['density_per_A']) for r in group])
            width=density/scale*.397
            supported=density>.002*scale
            direction=-1 if rep=='R1' else 1
            ax.fill_betweenx(ys,j,j+direction*width,where=supported,facecolor=COLORS[model],edgecolor=EDGE if model=='M0' else COLORS[model],
                             hatch=None if rep=='R1' else '////',linewidth=.7,zorder=2)
            # White hatch marks and repeat-specific boundary styles follow the original split violin.
            if rep=='R2':
                ax.fill_betweenx(ys,j,j+width,where=supported,facecolor='none',edgecolor='white',hatch='////',linewidth=0,zorder=3)
            ax.plot(np.where(supported,j+direction*width,np.nan),ys,color=EDGE if model=='M0' else COLORS[model],lw=.85,
                    ls='-' if rep=='R1' else (0,(3.6,2.15)),zorder=4)
            q=next(r for r in quant if (r['model'],r['repeat'])==(model,rep))
            assert int(q['n_frames'])==50000
            xp=j+direction*.081
            ax.plot([xp,xp],[float(q['q25_A']),float(q['q75_A'])],c=INK,lw=1.1,
                    path_effects=[pe.withStroke(linewidth=2.5,foreground='white')],zorder=5)
            ax.scatter([xp],[float(q['median_A'])],s=17,marker='o' if rep=='R1' else 'D',
                       facecolor='white',edgecolor=INK,linewidth=.7,zorder=6)
    ax.set(xlim=(-.5,2.5),ylim=(0,1.3),xticks=[0,1,2],xticklabels=MODELS,yticks=[0,.25,.5,.75,1,1.25])
    ax.set_yticklabels(['0.00','0.25','0.50','0.75','1.00','1.25'])
    ax.set_ylabel('Backbone RMSD (Å)',fontsize=8,labelpad=5)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',c=GRID,lw=.5);ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5,length=2.2,pad=2,width=.6)
    fig.text(.20,.975,'d   Backbone RMSD',fontsize=9.5,fontweight='bold',va='top')
    fig.text(.20,.89,'To the common frozen starting backbone',fontsize=7.5,va='top',color='#506575')
    handles=[Line2D([],[],c=INK,ls='-' if r=='R1' else '--',marker='o' if r=='R1' else 'D',
                    mfc='white',ms=3.5,lw=.8,label=f'{r}: '+('left, solid' if r=='R1' else 'right, hatched')) for r in ['R1','R2']]
    ax.legend(handles=handles,loc='upper right',frameon=False,fontsize=7.5,ncol=1,handlelength=1.8,borderpad=.15)
    return pdf_figure(fig)

def landscapes_with_blocks(root,dest):
    extra=BLOCK_H-118
    doc=fitz.open();pg=doc.new_page(width=W,height=486+extra)
    with fitz.open(root/'Figure_3/Figure_3.pdf') as src:
        erase(src[0],(0,70,25,250))
        erase(src[0],(441,58,W,250))
        paste(pg,src,(0,0,W,153),(0,0,W,153))
        paste(pg,src,(55,250.5,440,276),(55,153.5,440,179))
        paste(pg,src,(45,277,W,292),(45,180,W,195))
    with m2_blocks(root) as block:
        left=(W-PANEL_W)/2
        paste(pg,block,block[0].rect,(left,199,left+PANEL_W,199+BLOCK_H))
    with fitz.open(root/'Figure_4/Figure_4.pdf') as src:
        # Remove panel letters only, leaving the original curves and waterfall unchanged.
        erase(src[0],(51.5,185.7,58.5,198))
        erase(src[0],(295.5,185.7,303,198))
        paste(pg,src,(0,182,W,205),(0,322+extra,W,345+extra))
        paste(pg,src,(0,220,W,370),(0,336+extra,W,486+extra))
    # Add text after all imported panels so embedded font aliases cannot collide.
    text(pg,15,148,'Backbone cRMS deviation (°)',8,rotate=90)
    bar(pg,68,146)
    text(pg,52,336+extra,'e',10.3,True);text(pg,296,336+extra,'f',10.3,True)
    export(doc,dest)

def structures_with_rmsd(root,dest):
    doc=fitz.open();pg=doc.new_page(width=W,height=389+RMSD_H)
    with fitz.open(root/'Figure_6/Figure_6.pdf') as src:
        # Preserve all original manual molecular renders, contact bars, and numerical marks.
        paste(pg,src,src[0].rect,(0,0,W,380.24))
    with rmsd(root) as sub:
        left=(W-PANEL_W)/2
        paste(pg,sub,sub[0].rect,(left,389,left+PANEL_W,389+RMSD_H))
    export(doc,dest)

def repeat_diagnostics(root,dest):
    doc=fitz.open();pg=doc.new_page(width=W,height=393)
    with fitz.open(root/'Figure_3/Figure_3.pdf') as src:
        erase(src[0],(0,70,25,250));erase(src[0],(441,58,W,250))
        for box in [(58.8,317.8,65.9,330),(189,317.8,196,330),(319.1,317.8,325,330)]:
            erase(src[0],box)
        paste(pg,src,(0,295,W,483.12),(0,0,W,188.12))
        paste(pg,src,(45,157,W,292),(45,251,W,386))
    for x,name,letter in [(59.5,'M0','a'),(189.7,'A only','b'),(319.8,'Full','c')]:
        text(pg,x,32.1,letter,10.3,True);text(pg,x+12.5,32.1,name,8.5,True)
    text(pg,59.5,215,'R2 free-energy landscapes',10,True)
    for x,name,letter in [(59.5,'M0','d'),(189.7,'A only','e'),(319.8,'Full','f')]:
        text(pg,x,240,letter,10.3,True);text(pg,x+37,240,name,9,True)
    text(pg,15,340,'Backbone cRMS deviation (°)',8,rotate=90)
    bar(pg,258,336)
    export(doc,dest)

def verification(root):
    names=['tri_NMe_M2_blocks.tsv','tri_NMe_RMSD_KDE.tsv','tri_NMe_RMSD_quantiles.tsv']
    return {'data_sources':{n:hashlib.sha256((root/'Figure_4'/n).read_bytes()).hexdigest() for n in names},
            'new_simulations':False,'new_density_estimation':False,
            'block_rows':30,'density_rows':2700,'quantile_rows':6,
            'panel_mapping':{'S9d':'Figure 4d','S9e':'Figure 5d','Figure 4d':'Figure 4e','Figure 4e':'Figure 4f',
                             'S9a–c':'S9a–c','Figure 4a–c R2 row':'S9d–f'}}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    landscapes_with_blocks(args.inputs,args.out/'Main/Figure_4')
    structures_with_rmsd(args.inputs,args.out/'Main/Figure_5')
    repeat_diagnostics(args.inputs,args.out/'Supporting/Figure_S9')
    (args.out/'S9_Promotion_Verification.json').write_text(json.dumps(verification(args.inputs),indent=2)+'\n')
