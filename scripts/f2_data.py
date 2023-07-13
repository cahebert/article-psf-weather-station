import matplotlib.pyplot as plt
import numpy as np
import psfws
import common
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pickle

plt.style.use(f'./paper.mplstyle')
colors = common.color_scheme()

if __name__ == '__main__':
    f = plt.figure(figsize=(3.35, 3.35))
    f.set_tight_layout(False)

    gs = f.add_gridspec(4, 2, width_ratios=[1.75, 0.5], height_ratios=[0.1, 1, 1, 1])
    gs.update(hspace=0.35, wspace=0.1)

    ## top legend
    aleg = f.add_subplot(gs[0, :])
    aleg.axis('off')
    gl_line = Line2D([0], [0], label='ground layer', color=colors.gl,
                     lw=plt.rcParams['lines.linewidth']+0.25)
    fa_line = Line2D([0], [0], label='free atmosphere', color=colors.fa,
                     lw=plt.rcParams['lines.linewidth']+0.25)
    aleg.legend(handles=[gl_line, fa_line], ncol=2, loc='lower center',
                markerfirst=False, borderaxespad=-0.5, borderpad=0, handletextpad=0.4)

    ## wind plots
    afa = [f.add_subplot(gs[i, 0]) for i in [1, 2]]
    afah = [f.add_subplot(gs[i, 1]) for i in [1, 2]]

    d = common.open_example()
    seed = d['args'].atmSeed
    ws = psfws.ParameterGenerator(seed=seed)

    bins = {'speed': np.linspace(0, 80, 10),
            'phi': np.linspace(0, 360, 10),
            't': np.linspace(180, 300, 10)}

    example = ws.data_gl.index[s]

    for af, ah, col in zip(afa, afah, ['speed', 'phi']):
        # plot all FA profiles
        for ix in ws.data_fa.index:
            data = ws.data_fa.at[ix, col][ws.fa_start:]
            if col == 'phi':
                data = psfws.utils.smooth_dir(data)
                if data.mean() < 0:
                    data += 360
            af.plot(ws.h[ws.fa_start:], data, color=colors.fa, alpha=0.1,
                    lw=plt.rcParams['lines.linewidth']*0.75)

        # plot example
        af.plot(ws.h[ws.fa_start:], ws.data_fa.at[example, col][ws.fa_start:],
                color=colors.fa_accent)
        # plot GL points
        af.plot(np.ones(538)*ws.h0, ws.data_gl[col], 'o',
                ms=plt.rcParams['lines.markersize']-2, color=colors.gl, alpha=0.2)
        # plot GL example point
        af.plot(ws.h0, ws.data_gl.at[example, col], 'o', color=colors.gl_accent,
                ms=plt.rcParams['lines.markersize']-1.75)

        # histogram projections
        ah.hist(ws.data_gl[col], bins=bins[col], color=colors.gl, histtype='step',
                lw=plt.rcParams['lines.linewidth']*1.5, orientation='horizontal')
        ah.hist(np.concatenate(ws.data_fa[col]), bins=bins[col], histtype='step',
                lw=plt.rcParams['lines.linewidth']*1.5, color=colors.fa,
                orientation='horizontal', weights=np.ones(538*69)/69)

    # labels
    afa[0].set_ylabel(r'$v$ (m/s)')
    afa[1].set_ylabel(r'$\phi$ ($^\circ$ E of N)')
    afa[1].set_yticks([0, 90, 180, 270, 360], [r'$0$', '', r'$180$', '', r'$360$'])
    afah[1].set_yticks([0, 90, 180, 270, 360], [r'$0$', '', r'$180$', '', r'$360$'])

    # axis limits/ticks/etc
    afa[0].set_xlim(0, max(ws.h))
    [ax.set_xlim(afah[-1].get_xlim()) for ax in afah[:-1]]
    [ax.set_xlim(afa[0].get_xlim()) for ax in afa[1:]]
    [ax.set_xticklabels([]) for ax in afah[:-1] + afa]
    [afa[i].set_ylim(afah[i].get_ylim()) for i in range(2)]
    [ax.set_yticklabels([]) for ax in afah]

    ## plot Cn2 and Js
    at = f.add_subplot(gs[-1, 0])

    cn2, h = ws._get_fa_cn2(example)
    j, h_screens, edges = ws.get_turbulence_integral(pt=example, nl=6, location='com')
    cn2_int = psfws.utils.integrate_in_bins(cn2, h, np.array([edges[1], edges[-1]]))
    cn2_cal = cn2 * sum(j[1:]) / cn2_int

    at.plot(h, cn2, color=colors.fa_accent, alpha=0.3)
    at.plot(h, cn2_cal, color=colors.fa_accent)
    [at.axvline(e, color=colors.fa_accent, linestyle='--') for e in edges[1:]]
    at.axvline(edges[0], color=colors.gl_accent, linestyle='--')

    at.plot(h_screens[1:], j[1:] / 1e3 / (h_screens[-1]-h_screens[-2]),
            'o', color=colors.fa_accent, ms=plt.rcParams['lines.markersize']-1)
    at.plot(h_screens[0] + 0.4, j[0] / 1e3 / (h_screens[1]-h_screens[0]),
            'o', color=colors.gl, ms=plt.rcParams['lines.markersize']-1)

    at.set_yscale('log')
    at.set_xlim([0, max(ws.h)])
    at.set_xlabel(r'Altitude (km)')
    at.set_ylabel(r'$C_n^2$ (m$^{-2/3}$)')

    ## bottom legend
    alegt = f.add_subplot(gs[-1, -1])
    alegt.axis('off')
    uncal = Line2D([0], [0], label=r'$C_n^2$ uncal.', color=colors.fa_accent,
                   lw=plt.rcParams['lines.linewidth'], alpha=0.3)
    cal = Line2D([0], [0], label=r'$C_n^2$ cal. ', color=colors.fa_accent,
                 lw=plt.rcParams['lines.linewidth'])
    j_point = Line2D([0], [0], marker='o', lw=0, color=colors.fa_accent,
                     ms=plt.rcParams['lines.markersize']-1, label=r'$J$ / $\Delta h$')
    alegt.legend(handles=[uncal, cal, j_point], loc='center', handlelength=0.8,
                 borderaxespad=-0.1, borderpad=0, handletextpad=0.4)

    plt.subplots_adjust(bottom=0.15, top=0.95, left=0.2, right=0.95)
    plt.savefig(f'./../figures/f2_data.png', dpi=300)
    plt.show()
