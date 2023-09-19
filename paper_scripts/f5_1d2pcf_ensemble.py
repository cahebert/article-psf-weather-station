import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

import common
import plotHelp

plt.style.use(f'./paper.mplstyle')
colors = common.color_scheme()
seed = common.seed()
labels = common.labels()
markers = common.markers()

binnedXiP, binnedXiM, binnedXi, size1d, shear1d = plotHelp.getTuples()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-sumPath', type=str, default='../data/')
    args = parser.parse_args()

    subplots = [['s'],
                ['p'],
                ['m']]

    f, a = plt.subplot_mosaic(subplots, sharex=True, figsize=(3.25, 5),
                              gridspec_kw={'hspace': 0.2})
    f.set_tight_layout(False)

    for k, c, ls, l in zip(['psfws', 'rand', 'match'],
                           [colors.p, colors.b, colors.m],
                           ['-', '--', ':'],
                           [labels.p, labels.b, labels.m]):
        # load 1d 2pcfs
        Sstack, _, bins = plotHelp.get2pcf1d(k, 'big', 'size', args.sumPath, 1)
        Mstack, _, bins = plotHelp.get2pcf1d(k, 'big', 'shearMinus', args.sumPath, 1)
        Pstack, _, bins = plotHelp.get2pcf1d(k, 'big', 'shearPlus', args.sumPath, 1)

        # set lower limit of xvalues
        xcp = bins*60 >= 1
        xplot = bins[xcp] * 60

        for r, stack in zip(['p', 'm', 's'], [Pstack, Mstack, Sstack]):
            # if r == 's':
            #     # convert size to arcsec from pixels
            #     stack = np.array(stack) * (0.2)**2
            # get and plot 2pcf quantile values
            xi25th, xi50th, xi75th = np.quantile(stack, [0.25, 0.50, 0.75], axis=0)
            a[r].plot(xplot, xi50th[xcp], color=c, label=l, zorder=2, lw=1.25, ls=ls)
            if k == 'psfws':
                a[r].fill_between(xplot, xi25th[xcp], xi75th[xcp], color=c, alpha=0.1)

    [ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) for k, ax in a.items()]

    a['p'].set_ylabel(r'median $\xi_+(\theta)$')
    a['m'].set_ylabel(r'median $\xi_-(\theta)$')
    a['s'].set_ylabel(r'median $C(\theta)$ (arcsec$^2$)')

    a['p'].set_xscale('log')
    a['p'].set_xlim(left=1, right=60*max(bins))
    a['m'].set_xlabel(r'$\theta$ (arcmin)')

    a['m'].axhline(0, color='k', ls='--')
    a['s'].axhline(0, color='k', ls='--')
    a['p'].axhline(0, color='k', ls='--')

    a['s'].legend(ncol=1, loc='upper right', borderaxespad=0.75)

    plt.subplots_adjust(left=0.2, top=.95, bottom=0.1)
    plt.savefig(f'./../figures/f5_1d2pcf_ensemble.png', dpi=300)
    plt.show()
