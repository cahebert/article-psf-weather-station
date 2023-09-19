import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import common
import plotHelp
import pickle
import seaborn as sns

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

    degToAmin = 60

    # load raw output for example
    d = common.open_example()

    thx, thy = np.array(d['thx'])*degToAmin, np.array(d['thy'])*degToAmin
    e = np.hypot(d['e1'], d['e2'])
    beta = 0.5*np.arctan2(d['e2'], d['e1'])
    dx = e*np.cos(beta)
    dy = e*np.sin(beta)

    # get thetax,thetay 2pcf
    xyXi = plotHelp.compGG2pcf(d['thx'], d['thy'], d['e1'], d['e2'], 'big', 'TwoD')

    # polar coordinate images
    _, xipImS, coordsS = plotHelp.getImgs('psfws', 'shearP', args.sumPath, 'small')
    _, xipImB, coordsB = plotHelp.getImgs('psfws', 'shearP', args.sumPath, 'big')

    subplots = [['shape', 'sc'],
                ['.', '.'],
                ['xyxi', 'xyc'],
                ['.', '.'],
                ['bxi', 'c'],
                ['sxi', 'c']]

    f, a = plt.subplot_mosaic(subplots, figsize=(3.25, 6.5),
                              gridspec_kw={'hspace': 0.25,
                                           'height_ratios': [1, 0.001, 1, 0.001, 0.45, 0.45],
                                           'width_ratios': [1, .05]})
    f.set_tight_layout(False)

    # plot ellipticity over FOV
    display = np.random.choice(range(50000), 20000)
    e = np.hypot(d['e1'], d['e2'])[display]
    beta = 0.5*np.arctan2(d['e2'], d['e1'])[display]
    dx = e*np.cos(beta)
    dy = e*np.sin(beta)

    qdict = dict(alpha=1, angles='uv', headlength=0, headwidth=0, headaxislength=0,
                 minlength=0, pivot='middle', width=0.002, color=sns.color_palette("Purples")[-1])
    q = a['shape'].quiver(thx[display], thy[display], dx, dy, scale=1, **qdict)
    a['shape'].quiverkey(q, 160, -10, 0.03, r'$e$ = 0.03', coordinates='data', labelpos='N')

    a['sc'].clear()
    a['sc'].axis('off')

    a['shape'].set_xlabel(r'$\theta_x$ (arcmin)', labelpad=-.25)
    a['shape'].set_ylabel(r'$\theta_y$ (arcmin)')
    a['shape'].set_xlim(-1.9*degToAmin, 1.9*degToAmin)
    a['shape'].set_ylim(-1.9*degToAmin, 1.9*degToAmin)
    a['shape'].set_aspect('equal')

    # plot 2PCF in thetax/thetay
    exp = 4
    pcfAdjust = 10**exp
    exps = f'10^{exp}'

    extent = [-xyXi.max_sep, xyXi.max_sep, -xyXi.max_sep, xyXi.max_sep]
    extent = [e*degToAmin for e in extent]

    vmax = np.max(xyXi.xip*pcfAdjust)
    m = a['xyxi'].imshow(xyXi.xip*pcfAdjust, origin='lower', extent=extent,
                         cmap=colors.lCmap, vmax=vmax, vmin=0)
    label = r'$\xi_+(\Delta \theta_x, \Delta \theta_y)\times$ '+f'${exps}$'
    cb = plt.colorbar(m, cax=a['xyc'], label=label)

    a['xyxi'].set_xlabel(r'$\Delta \theta_x$ (arcmin)', labelpad=-.25)
    a['xyxi'].set_ylabel(r'$\Delta \theta_y$ (arcmin)')

    # plot 2PCFs in theta, alpha
    for size, xiim, coord in zip(['small', 'big'], [xipImS, xipImB], [coordsS, coordsB]):
        if size == 'small':
            limits = [.025*degToAmin, 0.1*degToAmin]
            grid = (14, 7)
            hlines = [.05 * degToAmin, .099 * degToAmin]
        elif size == 'big':
            limits = [0.29*degToAmin, 2*degToAmin]
            grid = (24, 13)
            hlines = [1.79 * degToAmin, 1.98 * degToAmin]

        alpha, theta, img = plotHelp.arrange2pcf(coord, xiim*pcfAdjust, size=size)
        theta = np.array(theta) * degToAmin
        cx = a[size[0]+'xi'].hexbin(alpha, theta, img, gridsize=grid,
                                    cmap=colors.lCmap, vmin=0, vmax=vmax)
        a[size[0]+'xi'].set_ylim(limits)

    cb = plt.colorbar(cx, cax=a['c'], label=r'$\xi_{+}(\theta,\alpha)\times$ '+f'${exps}$')

    [a[s+'xi'].set_ylabel(r'$\theta$ (arcmin)') for s in ['s', 'b']]
    [a[s+'xi'].set_xlim([0, 180]) for s in ['s', 'b']]
    [a[s+'xi'].set_xticks([0, 45, 90, 135, 180]) for s in ['s', 'b']]
    a['s'+'xi'].set_xlabel(r'$\alpha$ (deg)')

    plt.subplots_adjust(bottom=0.075, left=0.2, right=0.825, top=0.975)
    plt.savefig(f'./../figures/f6_2d2pcf_example.png', dpi=300)
    plt.show()
