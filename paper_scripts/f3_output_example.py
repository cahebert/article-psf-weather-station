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
    f, (asize, ashape) = plt.subplots(2,1, sharex=True, figsize=(3.35, 4.5), gridspec_kw={'hspace':0.01})
    f.set_tight_layout(False)
    [a.set_aspect('equal') for a in [asize, ashape]]

    degToAmin = 60
    d = common.open_example()
    thx, thy = np.array(d['thx'])*degToAmin, np.array(d['thy'])*degToAmin

    ## PSF size
    sigma = d['sigma'] - np.mean(d['sigma'])
    vmax = max(abs(sigma))
    sc = asize.hexbin(thx, thy, C=sigma, cmap=colors.dCmap, linewidths=0.2,
                      gridsize=40, vmax=vmax, vmin=-vmax)
    plotHelp.make_h_cbar(asize, sc, r'$\delta \sigma$ (arcsec)', orientation='w')

    # asize.set_xlabel(r'$\theta_x$ (arcmin)')
    asize.set_ylabel(r'$\theta_y$ (arcmin)')
    asize.set_xlim(-1.9*degToAmin, 1.9*degToAmin)
    asize.set_ylim(-1.9*degToAmin, 1.9*degToAmin)
    # asize.set_yticks([-100,0,100])

    ## PSF ellipticity
    display = np.random.choice(range(50000), 15000)
    e = np.hypot(d['e1'], d['e2'])[display]
    beta = 0.5*np.arctan2(d['e2'], d['e1'])[display]
    dx = e*np.cos(beta)
    dy = e*np.sin(beta)

    qdict = dict(alpha=1, angles='uv', headlength=0, headwidth=0, headaxislength=0,
                 minlength=0, pivot='middle', width=0.002, color='k')#,cmap=colors.pCmap)
    q = ashape.quiver(thx[display], thy[display], dx, dy, scale=1, **qdict)
    ashape.quiverkey(q, 160, -10, 0.03, r'$e$ = 0.03', coordinates='data', labelpos='N')

    cbarax = plotHelp.make_h_cbar(ashape, q, r'$|e|$', orientation='w')
    cbarax.clear()
    cbarax.axis('off')
    

    ashape.set_xlabel(r'$\theta_x$ (arcmin)')
    ashape.set_ylabel(r'$\theta_y$ (arcmin)')
    # ashape.set_yticks([-100,0,100])
    ashape.set_xlim(-1.9*degToAmin, 1.9*degToAmin)
    ashape.set_ylim(-1.9*degToAmin, 1.9*degToAmin)

    plt.subplots_adjust(left=0.2, right=0.8, top=0.95)
    plt.savefig(f'./../figures/f3_output_example.png', dpi=300)
    plt.show()
