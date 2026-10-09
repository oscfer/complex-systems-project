"""Static ensemble figures and matched saved-network animations; no simulations."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from src.control.study_analysis import load_raw
from src.visualization.network_layout import _seeded_spring_layout


def attention_node_areas(influence, N):
    """Fixed, bounded contrast scale in points squared; never frame-normalized.

    Reduced size floor separates weak nodes; a gentle nonlinear response makes
    strong nodes legible without allowing hubs to grow without bound. This is
    an attention encoding, not proportional physical area or institutional power.
    """
    r = np.maximum(np.asarray(influence, dtype=float), 0.)
    contrast = np.power(r / 3., 1.4)
    fraction = contrast / (1. + contrast)
    return (4. + 86. * fraction) if N > 100 else (12. + 360. * fraction)


def ensemble_figures(data,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);paths=[]
    s=data['summary'];initial=data['initial'];groups=data['groups']
    if not s.empty:
        fig,ax=plt.subplots(figsize=(10,max(4,.32*len(s))))
        labels=[f"{r.role} ({r.persistent}/{r.valid}; expected {r.expected})" for r in s.itertuples()]
        ax.barh(labels,s.persistent_rate.fillna(0),color=['#427ba8' if x else '#999999' for x in s.complete]);ax.invert_yaxis()
        ax.set(xlim=(0,1),xlabel='Persistent success fraction (futures nested within networks)',title=f'N={data["N"]}: all allocated selectors; '+('complete' if data['complete'] else 'PARTIAL'))
        fig.tight_layout();p=out/'ensemble_success.png';fig.savefig(p,dpi=140);plt.close(fig);paths.append(p)
    if not initial.empty:
        fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(initial))
        ax.bar(x-.18,initial.private_blue,.36,label='Private',color='#427ba8');ax.bar(x+.18,initial.public_blue,.36,label='Public',color='#ca823b')
        ax.axhline(.5,color='gray',ls=':');ax.axhline(.75,color='black',ls='--',label='Target')
        ax.set(xticks=x,xticklabels=initial.origin,ylim=(0,1),ylabel='Initial blue fraction',title='All origins retained, including initial blue majorities');ax.tick_params(axis='x',rotation=35);ax.legend();fig.tight_layout()
        p=out/'initial_conditions.png';fig.savefig(p,dpi=140);fig.savefig(out/'initial_conditions.pdf');plt.close(fig);paths.append(p)
    if not groups.empty and 'true_received_attention_mean' in groups:
        groups=groups.assign(_membership=groups.nodes.map(lambda x:tuple(x))).drop_duplicates(['origin','_membership'])
        fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,metric in zip(axs,['mean','max']):
            for count,rs in groups[groups['count']>0].groupby('count'):
                ax.scatter(rs[f'true_received_attention_{metric}'],rs.persistent_rate,label=f'{count} nodes',s=32,alpha=.7)
            ax.set(xlabel=f'Initial full-network influence: group {metric}',ylabel='Persistent fraction across audit futures',ylim=(-.03,1.03));ax.axvline(1,ls=':',color='gray');ax.legend(fontsize=8)
        fig.suptitle('All audited groups, including failures; descriptive associations, not causal effects');fig.tight_layout()
        p=out/'influence_outcomes.png';fig.savefig(p,dpi=140);fig.savefig(out/'influence_outcomes.pdf');plt.close(fig);paths.append(p)
    return paths


def comparison_choices(data):
    return data['roles'][data['roles']['count']>0][['origin','future','role','nodes','persistent']].copy()


def render_comparison(data,out_dir,*,origin=None,role=None,future=0,max_frames=24):
    """Two arms, same origin/future. Default first origin, coverage10, never success-picked."""
    rows=data['roles'];m=data['manifest']
    if rows.empty:raise ValueError('No completed audit comparisons yet')
    origin=origin or sorted(rows.origin.unique())[0]
    options=rows[(rows.origin==origin)&(rows.future==future)]
    if role is None:
        role='audience_coverage10' if 'audience_coverage10' in set(options.role) else sorted(set(options.role)-{'zero'})[0]
    selected=options[options.role.isin(['zero',role])].set_index('role')
    if not all(x in selected.index for x in ('zero',role)):raise ValueError('Both matched arms must be saved')
    units=[load_raw(data['bundle'],selected.loc[r,'physical_unit']) for r in ('zero',role)]
    frames=[u.get('network_frames',[]) for u in units]
    if not all(frames):raise ValueError('Frames were retained only for future 0 in main studies; choose future=0')
    times=[float(f['t']) for f in frames[0]]
    if times!=[float(f['t']) for f in frames[1]]:raise ValueError('Frame grids differ; cannot synchronize')
    for key in ('W','x','y'):
        if not np.array_equal(frames[0][0][key],frames[1][0][key]):raise ValueError('Origins differ')
    pos=_seeded_spring_layout(np.array(frames[0][0]['W']),iterations=50)
    cmap=plt.get_cmap('coolwarm');N=data['N']
    def draw(ax,frame):
        W=np.array(frame['W'],dtype=float);tot=W.sum(axis=1,keepdims=True)
        A=np.divide(W,tot,out=np.zeros_like(W),where=tot>0);influence=A.sum(axis=0)
        ii,jj=np.nonzero(W);segments=np.stack([pos[ii],pos[jj]],axis=1)
        ax.add_collection(LineCollection(segments,colors='#64748b',alpha=.09,linewidths=.4))
        area=attention_node_areas(influence, N)
        x=np.array(frame['x']).reshape(-1);y=np.array(frame['y']).reshape(-1)
        active=np.array(frame.get('controlled_mask',np.zeros(N)),bool)
        ax.scatter(pos[:,0],pos[:,1],s=area,c=cmap((y+1)/2),edgecolors='white',linewidths=.4)
        ax.scatter(pos[:,0],pos[:,1],s=area*.42,c=cmap((x+1)/2),edgecolors='white',linewidths=.3)
        ax.scatter(pos[active,0],pos[active,1],s=area[active]*1.3,facecolors='none',edgecolors='#eeb500',linewidths=1.3)
        ax.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3));ax.set_aspect('equal');ax.axis('off')
        return f'Private {np.mean(x<0):.0%}; public {np.mean(y<0):.0%}; actuated {active.sum()}'
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    chosen=[min(range(len(times)),key=lambda i:abs(times[i]-t)) for t in (0,20,320)]
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for row,(fs,label) in enumerate(zip(frames,['Matched zero',role])):
        for col,i in enumerate(chosen):
            info=draw(axes[row,col],fs[i]);axes[row,col].set_title(f'{label}; t={times[i]:g}\n{info}',fontsize=9)
    legend='Center: private; outer disk: public. Blue = -1; red = +1. Size: current received attention (nonlinear, bounded; same scale across frames). Yellow ring: current field.\nLines show directed memory support without arrowheads. Fixed layout from initial memory; positions are not geographic.'
    fig.suptitle(f'{origin}, future {future}; recipients {list(selected.loc[role,"nodes"])}' if len(selected.loc[role,'nodes'])<=10 else f'{origin}, future {future}; {len(selected.loc[role,"nodes"])} recipients (full IDs in table)',fontsize=10)
    fig.text(.5,.02,legend,ha='center',fontsize=8);fig.tight_layout(rect=(0,.07,1,.95));static=out/'network_comparison.png';fig.savefig(static,dpi=130);plt.close(fig)
    indices=sorted(set(np.linspace(0,len(times)-1,min(max_frames,len(times)),dtype=int))|{min(range(len(times)),key=lambda i:abs(times[i]-5)),min(range(len(times)),key=lambda i:abs(times[i]-20))})
    fig,axes=plt.subplots(1,2,figsize=(10,5));fig.subplots_adjust(bottom=.22,top=.82)
    fig.text(.5,.02,legend+'\nSampled frames do not certify uninterrupted holding.',ha='center',fontsize=8)
    def update(k):
        i=indices[k];fig.suptitle(f'{origin}, future {future}; t={times[i]:g}; release 20; horizon 320')
        for ax,fs,label in zip(axes,frames,['Matched zero',role]):
            ax.clear();info=draw(ax,fs[i]);ax.set_title(f'{label}\n{info}',fontsize=9)
    anim=FuncAnimation(fig,update,frames=len(indices),interval=300)
    from src.visualization.animation_player import inline_player_fixups
    html=inline_player_fixups(anim.to_jshtml());plt.close(fig);animation=out/'network_progress.html'
    style='<style>.modular-player{max-width:100%;width:960px;margin:auto}.modular-player .animation{max-width:100%}.modular-player img{max-width:100%;height:auto}.modular-player .anim-controls{max-width:100%;overflow-x:auto}</style>'
    animation.write_text(style+'<div class="modular-player">'+html+'</div>')
    # Paired scalar curves from exactly these saved frames.
    fig,axes=plt.subplots(1,2,figsize=(10,3),sharey=True)
    for ax,fs,label in zip(axes,frames,['Matched zero',role]):
        for layer,style,name in [('x','-','Private'),('y','--','Public')]:ax.plot(times,[np.mean(np.array(f[layer])<0) for f in fs],style,label=name)
        ax.axhline(.75,color='gray',ls=':');ax.axvspan(1,20,color='gold',alpha=.15);ax.set(title=label,xlabel='Time',ylim=(0,1.05));ax.legend()
    axes[0].set_ylabel('Blue support');fig.tight_layout();curves=out/'paired_support.png';fig.savefig(curves,dpi=140);plt.close(fig)
    return dict(static=static,animation=animation,curves=curves,origin=origin,role=role,future=future,nodes=list(selected.loc[role,'nodes']))


def interactive_comparisons(data, out_dir, *, style='study', max_bytes=None, inline_bytes=None):
    """Every saved matched comparison as one self-contained, offline page.

    The page is built once from the saved `network_frames` and then stepped
    through by its own buttons, so walking the whole ensemble costs no further
    simulation and no re-rendering. It is written to `out_dir/player.html` and
    also embedded here, which is the same document either way. Matplotlib
    re-rendering per selection (and the cascading dropdowns it needed) is gone.

    Returns the written paths and a size report. `render_comparison` still
    produces the static panels and support curves on request.
    """
    from IPython.display import display, HTML
    from src.visualization.animation_player import (PayloadBudgetError, INLINE_MAX_BYTES,
                                                    payload_from_data, prepare_comparisons,
                                                    render_player_html, write_player_html,
                                                    write_player_metadata)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    budget = max_bytes  # None preserves every saved comparison on disk.
    inline_budget = INLINE_MAX_BYTES if inline_bytes is None else inline_bytes
    title = f'Persistent transformation — {data["N"]} nodes, saved matched comparisons'
    ctx = prepare_comparisons(data)  # one verified read, reused by both budgets
    # Save the full export before attempting the optional notebook-sized view.
    payload, report = payload_from_data(data, style=style, max_bytes=budget, prepared=ctx)
    page = write_player_html(payload, out / 'player.html', title=title)
    write_player_metadata(payload, page, report=report, source=str(data['bundle']))
    inline_payload = inline_report = None
    if report['bytes'] > inline_budget:
        try:
            inline_payload, inline_report = payload_from_data(
                data, style=style, max_bytes=inline_budget, prepared=ctx)
        except PayloadBudgetError:
            print(f'The notebook byte budget cannot hold one comparison. '
                  f'Open the self-contained player at {page.resolve()} '
                  f'({len(payload["comparisons"])} comparisons).')
            return dict(page=page, metadata=page.with_name('player_metadata.json'),
                        shown=None, report=report, inline_report=None)
    shown = payload if inline_payload is None else inline_payload
    if inline_payload is not None:
        print(f'The embedded view shows {len(inline_payload["comparisons"])} of '
              f'{len(payload["comparisons"])} comparisons to keep this notebook small; '
              f'{page.name} holds {len(payload["comparisons"])} '
              f'({report["bytes"] / 1e6:.1f} MB, {report["frames_per_arm"]} sampled frames per arm).')
    else:
        print(f'All {len(payload["comparisons"])} comparisons embedded '
              f'({report["bytes"] / 1e6:.1f} MB, {report["frames_per_arm"]} sampled frames per arm).')
    print(f'Player written to {page} — open it directly, offline; no resource is fetched.')
    if report['dropped']:
        print(f'Byte budget omitted {len(report["dropped"])} comparison(s) from the declared order.')
    # Isolate IDs, CSS and keyboard handlers from the notebook and other players.
    import html
    document = html.escape(render_player_html(shown, title=title), quote=True)
    display(HTML('<iframe title="Saved network comparisons" sandbox="allow-scripts" '
                 'style="width:100%;height:1000px;border:0" srcdoc="' + document + '"></iframe>'))
    return dict(page=page, metadata=page.with_name('player_metadata.json'), shown=shown,
                report=report, inline_report=inline_report)
