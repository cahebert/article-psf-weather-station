import numpy as np
import pandas as pd
import sklearn
import pickle
import pathlib
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import common
import treecorr
seed = common.seed()


def getTuples():
    global binnedXiP
    global binnedXiM
    global binnedXi
    global size1d
    global shear1d
    binnedXiP  = namedtuple('binnedXiP', 'xiP xiPim bins')
    binnedXiM = namedtuple('binnedXiM', 'xiM xiMim bins')
    binnedXi = namedtuple('binnedXi', 'kk bins')
    size1d = namedtuple('size1d', 'kk rnom')
    shear1d = namedtuple('shear1d', 'xi xiIm rnom')
    return binnedXiP, binnedXiM, binnedXi, size1d, shear1d


def compGG2pcf(x, y, e1, e2, size, bin_type):
    """Calculate shear 2pcf."""
    cat = treecorr.Catalog(x=x, y=y, g1=e1, g2=e2, w=None)
    if size=='big':
        mxsep = 2
        nbins = 101
        mnsep = 1e-4
    elif size=='small':
        mxsep = 0.1
        nbins = 101
        mnsep = 1e-4
    gg = treecorr.GGCorrelation(min_sep=mnsep, max_sep=mxsep, nbins=nbins,
                                bin_type=bin_type, bin_slop=0)
    gg.process(cat)
    return gg


def arrange2pcf(coord, img, size='big'):
    """Arrange 2pcf data to polar view."""
    alphas, thetas = coord

    neg = alphas < 0
    if size == 'big':
        pos = (alphas >= 0) & (thetas <= 2)
    elif size == 'small':
        pos = (alphas >= 0) & (thetas <= 0.1)

    alphas = list(alphas[pos])+list(alphas[neg] % 180)
    thetas = list(thetas[pos])+list(thetas[neg])
    img = list(img[pos])+list(img[neg])

    return alphas, thetas, img


def make_h_cbar(ax, maple, label, orientation='h'):
    divider = make_axes_locatable(ax)
    if orientation=='h':
        ax_cb = divider.append_axes("top", size="5%", pad="15%")
        cbar = plt.colorbar(maple, cax=ax_cb, orientation='horizontal')
        cbar.set_label(label=label)
        ax_cb.xaxis.set_label_position("top")
    else:
        ax_cb = divider.append_axes("right", size="5%", pad="10%")
        cbar = plt.colorbar(maple, cax=ax_cb)
        cbar.set_label(label=label)       
    return ax_cb


def getImgs(kind, psfparam, sum_path, size='big'):
    f = sum_path + f'/polarSummary_images_{kind}.p'
    imgs = pickle.load(open(f, 'rb'))
    stacked = []
    for s in imgs.keys():
        if s != 'coord':
            stacked.append(imgs[s][f'{psfparam}_{size}'])

    return (np.median(stacked, axis=0),
            imgs[seed][f'{psfparam}_{size}'],
            (imgs[seed][f'size_{size}_theta'], imgs[seed][f'size_{size}_r']))


def get2pcf1d(kind, size, pname, sum_path, dim=2):
    f = sum_path + f'/{pname}_polarSummary_{kind}.p'
    allXi = pickle.load(open(f, 'rb'))

    if dim == 1:
        bins = allXi[seed][f'cf_1d_{size}'].rnom
        if 'shear' in pname:
            indXi = allXi[seed][f'cf_1d_{size}'].xi
            stackXi = [np.array(allXi[s][f'cf_1d_{size}'].xi) for s in allXi.keys()]
        elif pname == 'size':
            indXi = allXi[seed][f'cf_1d_{size}'].kk
            stackXi = [np.array(allXi[s][f'cf_1d_{size}'].kk) for s in allXi.keys()]
    elif dim == 2:
        bins = allXi[seed][f'cf_slice_{size}'].bins
        if pname == 'size':
            indXi = allXi[seed][f'cf_slice_{size}'].kk
            stackXi = [np.array(allXi[s][f'cf_slice_{size}'].kk) for s in allXi.keys()]
        elif pname == 'shearPlus':
            indXi = allXi[seed][f'cf_slice_{size}'].xiP
            stackXi = [np.array(allXi[s][f'cf_slice_{size}'].xiP) for s in allXi.keys()]
        elif pname == 'shearMinus':
            indXi = allXi[seed][f'cf_slice_{size}'].xiM
            stackXi = [np.array(allXi[s][f'cf_slice_{size}'].xiM) for s in allXi.keys()]
        stackXi = np.median(stackXi, axis=0)

    return stackXi, indXi, bins


def getMaxs(kind, size, pname, sum_path):
    f = sum_path + f'/{pname}_polarSummary_{kind}.p'
    allXi = pickle.load(open(f, 'rb'))

    if pname == 'size':
        xis = [allXi[s][f'cf_slice_{size}'].kk for s in allXi.keys()]
    if pname == 'shearPlus':
        xis = [allXi[s][f'cf_slice_{size}'].xiP for s in allXi.keys()]
    if pname == 'shearMinus':
        xis = [allXi[s][f'cf_slice_{size}'].xiM for s in allXi.keys()]

    bins = allXi[seed][f'cf_slice_{size}'].bins

    maxPosition = np.array([np.argmax(s) for s in xis])
    maxima = bins[maxPosition]

    sortexps = np.argsort([k for k in allXi.keys()])

    return np.array(maxima)[sortexps] % 180, bins


def phi_avg(speeds, directions):
    vx_bar = np.mean([v*np.cos(d) for v, d in zip(speeds, directions)][1:])
    vy_bar = np.mean([v*np.sin(d) for v, d in zip(speeds, directions)][1:])

    phi_bar = np.arctan2(vy_bar, vx_bar) * 180 / np.pi
    return phi_bar / 2


def getWinds(kind, sum_path):
    atm_path = sum_path + f'atm_polarSummary_{kind}.p'
    atm_summary = pickle.load(open(atm_path, 'rb'))
    atm_summary = pd.DataFrame([d for k, d in atm_summary.items()],
                               index=atm_summary.keys())
    gl_winds = np.array([dl[0].rad * 180 / np.pi
                        for dl in atm_summary['direction']]) % 180
    avg_wind = np.array([phi_avg(sp, dl)
                        for sp, dl in zip(atm_summary['speed'],
                                          atm_summary['direction'])]) % 180
    sortexps = np.argsort(atm_summary.index)
    return gl_winds[sortexps], avg_wind[sortexps]


def getWindSpeeds(kind, sum_path):
    atm_path = sum_path + f'atm_polarSummary_{kind}.p'
    atm_summary = pickle.load(open(atm_path, 'rb'))
    atm_summary = pd.DataFrame([d for k, d in atm_summary.items()],
                               index=atm_summary.keys())

    gl_speed = np.array([s[0] for s in atm_summary['speed']])
    avg_fa = np.array([np.mean(sp[1:]) for sp in atm_summary['speed']])
    sortexps = np.argsort(atm_summary.index)
    return gl_speed[sortexps], avg_fa[sortexps]


def jacknife_correlation(thing1, thing2, B):
    '''
    Bootstrap a correlation coefficient between thing1 and thing2, sampling B times.
    '''
    if len(thing1) != len(thing2):
        raise ValueError('things to correlate must have the same length!')

    idx = range(len(thing1))
    boot_corr_coefs = np.zeros(B)
    for i in range(B):
        resampled = sklearn.utils.resample(idx, replace=True, n_samples=len(idx)-1)
        boot_corr_coefs[i] = np.corrcoef(thing1[resampled],
                                         thing2[resampled], rowvar=False)[0, -1]
    return boot_corr_coefs
