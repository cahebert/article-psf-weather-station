# define colors (and other?) to be used by the individual figure scripts

from collections import namedtuple
import seaborn as sns
import colorcet as cc
import pickle


def color_scheme():
    ColorScheme = namedtuple('ColorScheme',
                             'fa, fa_accent, gl, gl_accent, p, b, m, dCmap,lCmap,pCmap')

    colors = ColorScheme('#8073ac', '#542788',
                         '#e08214', '#b35806',
                         '#C83E4D', '#5A7FAF', '#d8973c',
                         sns.color_palette("vlag", as_cmap=True),
                         cc.cm['CET_L17'],
                         sns.color_palette("Purples", as_cmap=True)
                         )
    return colors


def labels():
    legendLabel = namedtuple('legendLabel', 'p, b, m big small')
    labels = legendLabel(r'\textsc{psfws}', r'\textsc{bench}',  r'\textsc{match}',
                         r'$\theta=(108\,\mbox{-}\,120)$ arcmin',
                         r'$\theta=(3\,\mbox{-}\,6)$ arcmin')
    return labels


def markers():
    Markers = namedtuple('Markers', 'p, b, m')
    markers = Markers('o', 'v', 'D')
    return markers


def seed():
    return 287


def open_example():
    s = seed()
    return pickle.load(open(f'./../data/psfws{s}.p', 'rb'))
