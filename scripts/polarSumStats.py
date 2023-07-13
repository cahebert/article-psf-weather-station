import numpy as np
import pickle
import treecorr
import galsim
import argparse
import pandas as pd
import pathlib
from collections import namedtuple
import os
import matplotlib.pyplot as plt
import seaborn as sns

binnedXiP  = namedtuple('binnedXiP', 'xiP xiPim bins')
binnedXiM = namedtuple('binnedXiM', 'xiM xiMim bins')
binnedXi = namedtuple('binnedXi', 'kk bins')
size1d = namedtuple('size1d', 'kk rnom')
shear1d = namedtuple('shear1d', 'xi xiIm rnom')
 
def compKK2pcf(x, y, k, size, bin_type):
    """Calculate 2pcf for scalar k."""
    cat = treecorr.Catalog(x=x, y=y, k=k, w=None)
    if size=='big':
        mxsep = 2
        nbins = 101
        mnsep = 1e-4
    elif size=='small':
        mxsep = 0.1
        nbins = 101
        mnsep = 1e-4
    elif size=='data':
        mxsep = 1.0
        mnsep = 0.05
        nbins = 23
    kk = treecorr.KKCorrelation(min_sep=mnsep, max_sep=mxsep, nbins=nbins,
                                bin_type=bin_type, bin_slop=0)
    kk.process(cat)
    return kk

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
    elif size=='data':
        mxsep = 1.0
        mnsep = 0.05
        nbins = 23
    gg = treecorr.GGCorrelation(min_sep=mnsep, max_sep=mxsep, nbins=nbins,
                                bin_type=bin_type, bin_slop=0)
    gg.process(cat)
    return gg

def get2pcfPolar(kk, shear=False):
    """Cast PSF parameter 2pcf into polar coordinates."""
    dx = np.linspace(-kk.max_sep, kk.max_sep, kk.nbins)
    dy = np.linspace(-kk.max_sep, kk.max_sep, kk.nbins)
    dX, dY = np.meshgrid(dx, dy)
    r = np.hypot(dX, dY).ravel()
    theta = np.arctan2(dY, dX).ravel() * 180 / np.pi
    keep = (theta > 0)
    theta = 90 - theta
    if shear:
        polar = namedtuple('polar', 'r theta xiP xiPim xiM xiMim')
        return polar(r[keep], 
                     theta[keep], 
                     kk.xip.ravel()[keep],
                     kk.xip_im.ravel()[keep],
                     kk.xim.ravel()[keep],
                     kk.xim_im.ravel()[keep])
    else:
        polar = namedtuple('polar', 'r theta xi')
        return polar(r[keep], theta[keep], kk.xi.ravel()[keep])


def get2pcfAnnulus(kkPol, radius, width, shear=False):
    """Return annulus of radius and width from 2pcf."""
    inRing = (kkPol.r > radius) & (kkPol.r < radius + width)
    if shear:
        annulus = namedtuple('annulus', 'r theta xiP xiPim xiM xiMim')
        return annulus(kkPol.r[inRing], 
                       kkPol.theta[inRing], 
                       kkPol.xiP[inRing], 
                       kkPol.xiPim[inRing], 
                       kkPol.xiM[inRing],
                       kkPol.xiMim[inRing])
    else:
        annulus = namedtuple('annulus', 'r theta xi')
        return annulus(kkPol.r[inRing], kkPol.theta[inRing], kkPol.xi[inRing])


def get2pcfSlice(annulus, nBins, shear=False):
    """Return an averaged 2pcf within binned version of annulus."""
    theEdges = np.linspace(-90, 90, nBins + 1)
    theBins = (theEdges[1:] + theEdges[:-1])/2
    conditions = [(annulus.theta>=theEdges[i])&(annulus.theta<theEdges[i+1])
                  for i in range(nBins)]

    # want to convert this to bins from 0 to 180 for display purposes
    posBins = (theBins+180)%180
    order = np.argsort(posBins)

    if shear:
        xip = [np.mean(annulus.xiP[conditions[i]]) for i in range(nBins)]
        xim = [np.mean(annulus.xiM[conditions[i]]) for i in range(nBins)]
        xipim = [np.mean(annulus.xiPim[conditions[i]]) for i in range(nBins)]
        ximim = [np.mean(annulus.xiMim[conditions[i]]) for i in range(nBins)]
        return (binnedXiP(np.array(xip)[order], np.array(xipim)[order], posBins[order]), 
                binnedXiM(np.array(xim)[order], np.array(ximim)[order], posBins[order]))
    else:
        kk = [np.mean(annulus.xi[conditions[i]]) for i in range(nBins)]
        return binnedXi(np.array(kk)[order], posBins[order])


def getOutputSummary(d, outputImgs=True, outputCoords=True):
    """Extract summary parameters from simulation output file."""
    
    d_sigma = (d['sigma'] - np.mean(d['sigma'])) * 0.2 # convert to arcsec
    thx, thy = d['thx'], d['thy']
    
    size, shearP, shearM, imgs = {}, {}, {}, {}

    zippedargs = zip(['big', 'small'], [1.8, 0.05], [0.2, 0.05], [30, 18])

    for pcf_size, radius, width, nbins in zippedargs:
        for psf_param in ['size', 'shear']:
            if psf_param == 'size':
                cf = compKK2pcf(thx, thy, d_sigma, pcf_size, bin_type='TwoD')
                cf1d = compKK2pcf(thx, thy, d_sigma, pcf_size, bin_type='Log')
            elif psf_param == 'shear':
                cf = compGG2pcf(thx, thy, d['e1'], d['e2'], pcf_size, bin_type='TwoD')
                cf1d = compGG2pcf(thx, thy, d['e1'], d['e2'], pcf_size, bin_type='Log')
            cfPolar = get2pcfPolar(cf, (psf_param=='shear'))
            cfAnnulus = get2pcfAnnulus(cfPolar, radius, width, (psf_param=='shear'))
            cfSlice = get2pcfSlice(cfAnnulus, nbins, (psf_param=='shear'))

            if psf_param == 'size':
                size['cf_1d_' + pcf_size] = size1d(cf1d.xi, cf1d.rnom)
                size['cf_slice_' + pcf_size] = cfSlice
                imgs[psf_param+'_' + pcf_size] = cfPolar.xi
                
                if outputCoords:
                    imgs[psf_param + '_' + pcf_size + '_theta'] = cfPolar.theta
                    imgs[psf_param + '_' + pcf_size + '_r'] = cfPolar.r
            elif psf_param == 'shear':
                shearP['cf_1d_'+pcf_size] = shear1d(cf1d.xip, cf1d.xip_im, cf1d.rnom)
                shearM['cf_1d_'+pcf_size] = shear1d(cf1d.xim, cf1d.xim_im, cf1d.rnom)
                shearP['cf_slice_'+pcf_size], shearM['cf_slice_'+pcf_size] = cfSlice
                imgs['shearP_'+pcf_size] = cfPolar.xiP
                imgs['shearM_'+pcf_size] = cfPolar.xiM
                imgs['shearPim_'+pcf_size] = cfPolar.xiPim
                imgs['shearMim_'+pcf_size] = cfPolar.xiMim

    size['autocorr'] = np.var(d_sigma)
    shearP['autocorr_e1'] = np.var(d['e1'])
    shearM['autocorr_e2'] = np.var(d['e2'])
    
    return size, shearP, shearM, imgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", type=str)
    parser.add_argument("--nPool", type=int, default=1)
    parser.add_argument("--outdir", type=str, default='/home/users/chebert/validate-psfws/summaries/allSixMonths_shear/')
    parser.add_argument("--simdir", type=str, default='/home/groups/burchat/chebert/psfwsPaperSims/')
    args = parser.parse_args()

    if args.kind not in ['psfws', 'rand', 'match']:
        raise ValueError('kind input must be "psfws", "rand", "match"')

    N = range(538)

    size_sum, shearP_sum, shearM_sum, images = {}, {}, {}, {}

    def f(num):
        fname = args.kind + f'{num}.p'
        file_path = pathlib.Path.joinpath(pathlib.Path(args.simdir), fname)
        try:
            d = pickle.load(open(file_path, 'rb'))
        except FileNotFoundError:
            print(f'Warning: file with seed {num} not found!!')

        out = getOutputSummary(d)
        return num, out

    if args.nPool > 1:
        from multiprocessing import Pool
        with Pool(args.nPool) as pool:
            for out in pool.imap_unordered(f, N):
                num = out[0]
                size_sum[num], shearP_sum[num], shearM_sum[num], images[num] = out[1]
    else:
        for num in N:
            num, out = f(num)
            size_sum[num], shearP_sum[num], shearM_sum[num], images[num] = out
    
    f_name = f'polarSummary_images_{args.kind}.p'
    save_path = pathlib.Path.joinpath(pathlib.Path(args.outdir), f_name)
    pickle.dump(images, open(save_path, 'wb'))

    for summary, name in zip([size_sum, shearP_sum, shearM_sum],
                             ['size', 'shearPlus', 'shearMinus']):
        f_name = f'{name}_polarSummary_{args.kind}.p'
        save_path = pathlib.Path.joinpath(pathlib.Path(args.outdir), f_name)
        pickle.dump(summary, open(save_path, 'wb'))
