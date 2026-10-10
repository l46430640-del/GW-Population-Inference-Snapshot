"""Posterior density curves following the published PRX Fig. 16 convention.

Every interval is a quantile of density draws, never a spread of optimizers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from analysis_results import lisa_baselines

ROOT = Path(__file__).resolve().parents[1]
COLOR = '#147d92'


def lisa(morphologies, output):
    matplotlib.rcParams.update({'font.size': 12.5, 'axes.labelsize': 13.75,
                               'axes.titlesize': 12.5, 'xtick.labelsize': 12.5,
                               'ytick.labelsize': 12.5, 'pdf.fonttype': 42,
                               'ps.fonttype': 42, 'axes.linewidth': .75})
    labels = [r'$\ell_1$', r'$z$', r'$q$']
    ylabels = [r'$p(\ell_1)$', r'$p(z)$', r'$p(q)$']
    low, high = np.array([5., .21, .1]), np.array([9., 7.5, 1.])
    fig, axes = plt.subplots(len(morphologies), 3, figsize=(7.5, 2.5*len(morphologies)+.55), squeeze=False)
    selected = lisa_baselines()
    for row, name in enumerate(morphologies):
        directory = ROOT/selected[name]
        data = np.load(directory/'density_summary.npz')
        for column, ax in enumerate(axes[row]):
            width = high[column]-low[column]
            x = low[column]+width*data['marginal_x']
            draws = data['marginal_draws'][column]/width
            lower, upper = np.quantile(draws, [.05, .95], axis=0)
            ax.fill_between(x, lower, upper, color=COLOR, alpha=.22, linewidth=0)
            ax.plot(x, draws.mean(axis=0), color=COLOR, lw=1.5)
            ax.plot(x, data['marginal_truth'][column]/width, color='#333333', lw=1.4, ls='--')
            floor = .001
            ceiling = 10**np.ceil(np.log10(max(upper.max(), (data['marginal_truth'][column]/width).max())))
            ax.set(xlim=(low[column], high[column]), yscale='log', ylim=(floor, ceiling), xlabel=labels[column], ylabel=ylabels[column])
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(direction='out', length=3)
            ax.set_title(f'({chr(97+3*row+column)}) '+{'correlated':'Correlated','bimodal':'Bimodal','curved':'Curved'}[name], loc='left', pad=9)
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.yaxis.set_minor_locator(NullLocator())
            if column == 0:
                ax.set_xticks([5, 6, 7, 8, 9])
    handles = [Line2D([], [], color='#333333', ls='--', lw=1.4, label='Injected'),
               Line2D([], [], color=COLOR, lw=1.5, label='Posterior mean'),
               Patch(facecolor=COLOR, alpha=.22, label='90% interval')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5,.995))
    fig.subplots_adjust(left=.12, right=.97, bottom=.085 if len(morphologies)>1 else .20,
                        top=.90 if len(morphologies)>1 else .76, wspace=.48, hspace=.68)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix('.png'), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--morphologies', nargs='+', default=['correlated','bimodal','curved'])
    parser.add_argument('--output', default='figures/lisa_marginals.pdf')
    parser.add_argument('--lvk-directory')
    args = parser.parse_args()
    if args.lvk_directory:
        lvk(ROOT/args.lvk_directory,ROOT/args.output)
    else:
        lisa(args.morphologies, ROOT/args.output)


def lvk(directory,output):
    matplotlib.rcParams.update({'font.size':12.5,'axes.labelsize':13.75,'xtick.labelsize':12.5,
                               'ytick.labelsize':12.5,'pdf.fonttype':42,'axes.linewidth':.75})
    fitted = np.load(directory/'density_summary.npz')
    reference = np.load(ROOT/'results/analysis/lvk_reference/density_summary.npz')
    fig,axes = plt.subplots(1,3,figsize=(7.5,3.1))
    for column,ax in enumerate(axes):
        upper_bound = 0.
        for data,color in ((reference,'#b25845'),(fitted,COLOR)):
            curves = data['marginal_draws'][column]
            x = data['marginal_x'][column]
            if column == 0:
                x = 10**x
            low,high = np.quantile(curves,[.05,.95],axis=0)
            upper_bound = max(upper_bound,float(high.max()))
            ax.fill_between(x,low,high,color=color,alpha=.18,lw=0)
            ax.plot(x,curves.mean(0),color=color,lw=1.5)
        ax.set(xlim=([3,150],[0,2.3],[0,1])[column],
               yscale='log',ylim=(.001,10**np.ceil(np.log10(upper_bound))),
               xlabel=(r'$m_1/M_\odot$',r'$z$',r'$q$')[column],
               ylabel=(r'$p(\ell)$',r'$p(z)$',r'$p(q)$')[column])
        if column == 0:
            ax.set_xscale('log')
            ax.set_xticks([5,10,30,100],labels=['5','10','30','100'])
            ax.xaxis.set_minor_locator(NullLocator())
        if column == 2:
            ax.set_xticks([0,.5,1],labels=['0','0.5','1'])
        ax.spines[['top','right']].set_visible(False)
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_title(f'({chr(97+column)})',loc='left',fontsize=12.5,pad=9)
    handles = [Line2D([],[],color=COLOR,lw=1.5,label='Copula mixture'),
               Line2D([],[],color='#b25845',lw=1.5,label='LVK Power Law + Peak')]
    fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,.995))
    fig.subplots_adjust(left=.12,right=.97,bottom=.23,top=.76,wspace=.50)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix('.png'),dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
