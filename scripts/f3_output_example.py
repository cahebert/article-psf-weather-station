import matplotlib.pyplot as plt
import numpy as np
import psfws
import common
import pickle
import seaborn as sns
import treecorr
from collections import namedtuple
import plotHelp

plt.style.use(f'./paper.mplstyle')
colors = common.color_scheme()

if __name__ == '__main__':
    f, (asize, ashape) = plt.subplots(1, 2, sharex=True, figsize=(3.35, 2.25))
    f.set_tight_layout(False)
    [a.set_aspect('equal') for a in [asize, ashape]]

    degToAmin = 60
    d = common.open_example()
    thx, thy = np.array(d['thx'])*degToAmin, np.array(d['thy'])*degToAmin

    ## PSF size
    sigma = (d['sigma'] - np.mean(d['sigma'])) * 0.2
    vmax = max(abs(sigma))
    sc = asize.hexbin(thx, thy, C=sigma, cmap=colors.dCmap, linewidths=0.2,
                      gridsize=25, vmax=vmax, vmin=-vmax)
    plotHelp.make_h_cbar(asize, sc, r'$\delta \sigma$ (arcsec)')

    asize.set_xlabel(r'$\theta_x$ (arcmin)')
    asize.set_ylabel(r'$\theta_y$ (arcmin)', labelpad=-0.8)
    asize.set_xlim(-1.9*degToAmin, 1.9*degToAmin)
    asize.set_ylim(-1.9*degToAmin, 1.9*degToAmin)

    ## PSF ellipticity
    e = np.hypot(d['e1'], d['e2'])
    beta = 0.5*np.arctan2(d['e2'], d['e1'])
    dx = e*np.cos(beta)
    dy = e*np.sin(beta)

    qdict = dict(alpha=1, angles='uv', headlength=0, headwidth=0, headaxislength=0,
                 minlength=0, pivot='middle', width=0.0025, cmap=colors.pCmap)
    q = ashape.quiver(thx, thy, dx, dy, e, scale=1, **qdict)
    plotHelp.make_h_cbar(ashape, q, r'$|e|$')

    ashape.set_xlabel(r'$\theta_x$ (arcmin)')
    ashape.set_yticklabels([])
    ashape.set_xlim(-1.9*degToAmin, 1.9*degToAmin)
    ashape.set_ylim(-1.9*degToAmin, 1.9*degToAmin)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.95)
    plt.savefig(f'./../figures/f3_output_example.png', dpi=300)
    plt.show()
