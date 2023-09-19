import numpy as np
import pandas as pd
import pathlib
import pickle

import matplotlib.pyplot as plt
import matplotlib.lines as lines

import common
import plotHelp

plt.style.use(f'./paper.mplstyle')
colors = common.color_scheme()
labels = common.labels()
markers = common.markers()

binnedXiP, binnedXiM, binnedXi, size1d, shear1d = plotHelp.getTuples()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-sumPath', type=str, default='../data/')
    args = parser.parse_args()

    summaries = pathlib.Path(args.sumPath)

    subplots = [['p', 'r', 'm', 'h']]

    f, a = plt.subplot_mosaic(subplots, figsize=(7.5, 2.5), sharey=True,
                              gridspec_kw={'wspace': 0, 'width_ratios': [3, 3, 3, 2]})
    f.set_tight_layout(False)

    for kind, c, s, l, ls in zip(['rand', 'match', 'psfws'],
                                 [colors.b, colors.m, colors.p],
                                 [markers.b, markers.m, markers.p],
                                 [labels.b, labels.m, labels.p],
                                 ['--', ':', '-']):
        # load wind and variance data for simulation type
        gl_winds, avg_wind = plotHelp.getWindSpeeds(kind, args.sumPath)

        psf_path = pathlib.Path.joinpath(summaries, f'shearPlus_polarSummary_{kind}.p')
        summary = pickle.load(open(psf_path, 'rb'))
        summary = pd.DataFrame([d for k, d in summary.items()], index=summary.keys())
        variances = summary.sort_index()['autocorr_e1']
        psf_path = pathlib.Path.joinpath(summaries, f'shearMinus_polarSummary_{kind}.p')
        summary = pickle.load(open(psf_path, 'rb'))
        summary = pd.DataFrame([d for k, d in summary.items()], index=summary.keys())
        variances += summary.sort_index()['autocorr_e2']

        ylims = [1e-5, 4e-3]
        bins = np.logspace(np.log10(ylims[0]), np.log10(ylims[1]), 15)

        # scatter
        a[kind[0]].scatter(gl_winds, variances, marker=s, color=c, zorder=1,
                           label=l, alpha=0.3, edgecolors='None',
                           s=plt.rcParams['lines.markersize']+6)
        # projected hist
        a['h'].hist(variances, bins, color=c, ls=ls, histtype='step',
                    orientation='horizontal', label=l, lw=1.25)

        a[kind[0]].grid(zorder=2, alpha=0.75, lw=0.5)
        a[kind[0]].text(.25, .000025, l, fontsize='large')

    a['h'].legend(handlelength=1, bbox_to_anchor=(0.2, 1), loc='upper left')
    a['p'].set_yscale('log')
    a['p'].set_ylim(ylims)

    [a[k].set_xscale('log') for k in ['p', 'm', 'r']]
    [a[k].set_xlim([2e-1, 25]) for k in ['p', 'r', 'm']]

    a['p'].set_ylabel(r'Var(e)')
    a['r'].set_xlabel(r'$v$(GL) (m/s)')
    a['h'].set_xlabel('\# of simulations')

    plt.subplots_adjust(left=0.1, bottom=0.2, top=0.95, right=.95)
    plt.savefig(f'./../figures/f4_e1var_speed.png', dpi=300)
    plt.show()
