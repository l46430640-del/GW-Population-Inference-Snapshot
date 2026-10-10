"""Display conditional density recovery using the verified PRX Fig. 16 convention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plot(directory, destination):
    data = np.load(directory / 'conditional_densities.npz')
    summary = json.loads((directory / 'comparison.json').read_text())
    starts = summary['flow_starts']
    selected = {mode: min((name for name in starts if name.startswith(mode + '_')),
                          key=lambda name: starts[name]['last_1000_mean_negative_elbo'])
                for mode in ('vi', 'independent')}
    matplotlib.rcParams.update({'font.size': 12.5, 'axes.labelsize': 13.5,
                               'axes.titlesize': 12.5, 'xtick.labelsize': 12,
                               'ytick.labelsize': 12, 'pdf.fonttype': 42,
                               'ps.fonttype': 42, 'axes.linewidth': .75})
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.6))
    x = (np.arange(96) + .5) / 96
    intervals = [r'$0<x_1<1/3$', r'$1/3<x_1<2/3$', r'$2/3<x_1<1$']
    for row, axis in enumerate((1, 2)):
        for column, ax in enumerate(axes[row]):
            upper = 0.
            for name, color in ((selected['independent'], '#b25845'), (selected['vi'], '#147d92')):
                density = data[f'{name}_axis{axis}'][column]
                np.testing.assert_allclose(density.sum(-1)/96, 1., atol=1e-8)
                if not np.isfinite(density).all() or density.min() < -1e-10:
                    raise ValueError('Invalid conditional density')
                low, high = np.quantile(density, [.05, .95], axis=0)
                mean = density.mean(0)
                upper = max(upper, float(high.max()), float(mean.max()))
                ax.fill_between(x, low, high, color=color, alpha=.17, lw=0)
                ax.plot(x, mean, color=color, lw=1.4)
            truth = data[f'truth_axis{axis}'][column, 0]
            upper = max(upper, float(truth.max()))
            ax.plot(x, truth, color='#333333', ls='--', lw=1.4)
            ax.set(xlim=(0, 1), ylim=(0, upper*1.07),
                   xlabel=rf'$x_{axis+1}$',
                   ylabel=rf'$p(x_{axis+1}\mid I_{column+1})$')
            ax.set_xticks([0, .5, 1], labels=['0', '0.5', '1'])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(direction='out', length=3)
            ax.set_title(f'({chr(97+3*row+column)}) ' + intervals[column], loc='left', pad=9)
    handles = [Line2D([], [], color='#333333', ls='--', lw=1.4, label='Injected'),
               Line2D([], [], color='#147d92', lw=1.4, label='Copula mixture'),
               Line2D([], [], color='#b25845', lw=1.4, label='Independent components')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.51, .995), handlelength=1.7, columnspacing=1.1)
    fig.subplots_adjust(left=.11, right=.98, bottom=.105, top=.86, wspace=.51, hspace=.73)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    fig.savefig(destination.with_suffix('.png'), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='results/analysis/selected_mixture')
    parser.add_argument('--output', default='figures/selected_mixture_conditionals.pdf')
    args = parser.parse_args()
    plot(ROOT / args.directory, ROOT / args.output)


if __name__ == '__main__':
    main()
