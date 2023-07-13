import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.ticker import MultipleLocator

import common
import plotHelp

plt.style.use(f'./paper.mplstyle')
colors = common.color_scheme()
labels = common.labels()
markers = common.markers()

binnedXiP, binnedXiM, binnedXi, size1d, shear1d = plotHelp.getTuples()

histlw = plt.rcParams['lines.linewidth']+0.5
scatterS = plt.rcParams['lines.markersize'] + 8
plt.rcParams['patch.linewidth'] = 0.5

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-sumPath', type=str, default='../data/')
    args = parser.parse_args()

    psfparam = 'shearP'
    pname = 'shearPlus'

    subplots = [['glhist', 'fahist', '.'],
                ['glbig', 'fabig', 'bighist'],
                ['glsmall', 'fasmall', 'smallhist']]

    f, a = plt.subplot_mosaic(subplots, figsize=(7.2, 7.2),
                              gridspec_kw={'height_ratios': [0.5, 1, 1],
                                           'width_ratios': [1, 1, 0.5],
                                           'wspace': 0.1, 'hspace': 0.1})
    f.set_tight_layout(False)

    for kind, c, s, ls in zip(['rand', 'match', 'psfws'],
                              [colors.b, colors.m, colors.p],
                              [markers.b, markers.m, markers.p],
                              ['--', ':', '-']):
        # load wind directions
        gl_winds, avg_wind = plotHelp.getWinds(kind, args.sumPath)
        bins = np.linspace(0, 180, 18)

        # wind histograms
        a['glhist'].hist(gl_winds, bins, lw=histlw, ls=ls, color=c, histtype='step')
        a['fahist'].hist(avg_wind, bins, lw=histlw, ls=ls, color=c, histtype='step')

        for size in ['big', 'small']:
            # load alpha_max
            maxima, maxbins = plotHelp.getMaxs(kind, size, pname, args.sumPath)
            bins = np.linspace(0, 180, int(len(maxbins)/2)+1)

            # calculate correlation coefficients
            GLccs = plotHelp.jacknife_correlation(gl_winds, maxima, 1000)
            FAccs = plotHelp.jacknife_correlation(avg_wind, maxima, 1000)
            g = fr'\boldmath$\rho={np.mean(GLccs):+.2f} \pm {np.std(GLccs):.2f}$'
            f = fr'\boldmath$\rho={np.mean(FAccs):+.2f} \pm {np.std(FAccs):.2f}$'

            if kind == 'psfws':
                a['gl'+size].scatter(gl_winds, maxima, marker=s, label=g,
                                     color=c, s=scatterS-2, alpha=0.5, edgecolors='None')
                a['fa'+size].scatter(avg_wind, maxima, marker=s, label=f,
                                     color=c, s=scatterS-2, alpha=0.5, edgecolors='None')
            else:
                a['gl'+size].scatter(gl_winds, maxima, marker=s, label=g, lw=0.75,
                                     color=c, s=scatterS, facecolors='None', alpha=0.5)
                a['fa'+size].scatter(avg_wind, maxima, marker=s, label=f, lw=0.75,
                                     color=c, s=scatterS, facecolors='None', alpha=0.5)

            # alpha_max histograms
            a[size+'hist'].hist(maxima, bins, lw=histlw, color=c, ls=ls,
                                histtype='step', orientation='horizontal')

    [a[l].legend(frameon=True, loc='upper left', scatterpoints=0,
                 labelcolor=[colors.b, colors.m, colors.p], labelspacing=0.75,
                 handlelength=0, borderaxespad=0.25, handletextpad=0) 
     for l in ['fasmall', 'fabig', 'glsmall', 'glbig']]

    ll = [lines.Line2D([0], [0], color=colors.b, lw=0, marker='v', mfc='None', mew=0.9),
          lines.Line2D([0], [0], color=colors.m, lw=0, marker='D', mfc='None', mew=0.9),
          lines.Line2D([0], [0], color=colors.p, lw=0, marker='o', mec='None')]

    a['fahist'].legend(ll, [labels.b, labels.m, labels.p], handletextpad=0.1,
                       loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # labels
    a['glsmall'].set_ylabel(r'$\alpha_{\text{max}}$ for ' + labels.small)
    a['glbig'].set_ylabel(r'$\alpha_{\text{max}}$ for ' + labels.big)
    a['glhist'].set_ylabel('\# of simulations')
    a['smallhist'].set_xlabel('\# of simulations')
    a['glsmall'].set_xlabel(r'$\phi$ (GL)')
    a['fasmall'].set_xlabel(r'$\phi$ (FA)')

    scatters = ['glsmall', 'glbig', 'fasmall', 'fabig']
    ticks = [0, 45, 90, 135, 180]

    # set ticks and axis limits
    for l in scatters + ['bighist', 'smallhist']:
        a[l].set_yticks(ticks)
        a[l].yaxis.set_minor_locator(MultipleLocator(45))
    [a[l].set_yticklabels([]) for l in ['fasmall', 'fabig', 'bighist', 'smallhist', 'fahist']]

    for l in scatters + ['fahist', 'glhist']:
        a[l].set_xticks(ticks)
        a[l].xaxis.set_minor_locator(MultipleLocator(45))
    [a[l].set_xticklabels([]) for l in ['fabig', 'glbig', 'fahist', 'glhist', 'bighist']]

    [a[l].set_ylim(top=220) for l in ['fahist', 'glhist']]
    [a[l].set_xlim(right=220) for l in ['bighist', 'smallhist']]

    [a[l].set_ylim([0, 180]) for l in ['bighist', 'smallhist']+scatters]
    [a[l].set_xlim([0, 180]) for l in ['glhist', 'fahist']+scatters]

    plt.subplots_adjust(bottom=0.1, left=0.1, top=0.95, right=0.95)
    plt.savefig(f'./../figures/f7_alphamax_scatter.png', dpi=300)
    # plt.show()
