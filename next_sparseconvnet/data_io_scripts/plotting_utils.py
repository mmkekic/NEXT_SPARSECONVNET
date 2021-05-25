import numpy  as np
import pandas as pd
import tables as tb
import matplotlib.pyplot  as plt
import matplotlib         as mpl
from mpl_toolkits.mplot3d import Axes3D
import itertools

def plot_projections(hits, value='energy', coords = ['x', 'y', 'z'], cmap = mpl.cm.jet, th = 0):
    fig, axs = plt.subplots(nrows=1, ncols=3,
                                        figsize=(12, 6))
    coors_pairs = itertools.combinations(coords, 2)
    cmap.set_under('white')
    for i, coor_pair in enumerate(coors_pairs):
        sel = hits.groupby(list(coor_pair))[value].sum()
        ind0 = sel.index.get_level_values(coor_pair[0])
        ind1 = sel.index.get_level_values(coor_pair[1])
        newind0 = np.arange(ind0.min(), ind0.max()+1)
        newind1 = np.arange(ind1.min(), ind1.max()+1)
        xx, yy = np.meshgrid(newind0, newind1)
        newind = pd.Index(list(zip(xx.flatten(), yy.flatten())), name=tuple(coor_pair))
        sel = sel.reindex(newind,  fill_value=0).reset_index()
        sel = pd.pivot_table(sel, values=value, index=[coor_pair[0]],
                        columns=[coor_pair[1]], aggfunc=np.sum)
        #print((newind0.min(),newind0.max(), newind1.min(),  newind1.max()))
        axs[i].imshow(sel.T, origin='lower', vmin=th+np.finfo(float).eps, extent=(newind0.min(),newind0.max(), newind1.min(),  newind1.max()),
                      cmap=cmap, aspect='auto')
        axs[i].set_xlabel(coor_pair[0])
        axs[i].set_ylabel(coor_pair[1])
    fig.tight_layout()

    plt.show()

def plot_3d_vox(hits_digitized, value='energy', coords = ['x', 'y', 'z'], th=0, edgecolor=None, cmap=mpl.cm.jet):

    xmin, xmax = hits_digitized[coords[0]].min(), hits_digitized[coords[0]].max()
    ymin, ymax = hits_digitized[coords[1]].min(), hits_digitized[coords[1]].max()
    zmin, zmax = hits_digitized[coords[2]].min(), hits_digitized[coords[2]].max()

    nbinsX = int(np.ceil((xmax-xmin))) + 2
    nbinsY = int(np.ceil((ymax-ymin))) + 2
    nbinsZ = int(np.ceil((zmax-zmin))) + 2
    xarr = np.ones(shape=(nbinsX, nbinsY, nbinsZ))*th

    nonzeros = np.vstack([hits_digitized[coords[0]].values-xmin+1,
                          hits_digitized[coords[1]].values-ymin+1,
                          hits_digitized[coords[2]].values-zmin+1])
    xarr[tuple(nonzeros)] = hits_digitized[value].values
    dim     = xarr.shape
    voxels  = xarr > th

    fig  = plt.figure(figsize=(15, 15), frameon=False)
    gs   = fig.add_gridspec(2, 40)
    ax   = fig.add_subplot(gs[0, 0:16], projection = '3d')
    axcb = fig.add_subplot(gs[0, 18])
    norm = mpl.colors.Normalize(vmin=xarr.min(), vmax=xarr.max())
    m    = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = np.asarray(np.vectorize(m.to_rgba)(xarr))
    colors = np.rollaxis(colors, 0, 4)

    ax.voxels(voxels, facecolors=colors, edgecolor=edgecolor)
    cb = mpl.colorbar.ColorbarBase(axcb, cmap=cmap, norm=norm, orientation='vertical')

    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    cb.set_label (value)

    plt.show()

def plot_3d_hits(hits, value='energy', coords = ['x', 'y', 'z'], cmap = mpl.cm.jet):
    fig  = plt.figure(figsize=(15, 15), frameon=False)
    gs   = fig.add_gridspec(2, 40)
    ax   = fig.add_subplot(gs[0, 0:16], projection = '3d')
    axcb = fig.add_subplot(gs[0, 18])
    norm = mpl.colors.Normalize(vmin=hits.loc[:, value].min(), vmax=hits.loc[:, value].max())

    m    = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = np.asarray(np.vectorize(m.to_rgba)(hits.loc[:, value]))
    colors = np.rollaxis(colors, 0, 2)

    ax.scatter(hits[coords[0]], hits[coords[1]], hits[coords[2]], c=colors, marker='o')
    cb = mpl.colorbar.ColorbarBase(axcb, cmap=cmap, norm=norm, orientation='vertical')


    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    cb.set_label (value)

    plt.show()


def read_event(fname, datid, table='Voxels', group='DATASET', df=True):
    with tb.open_file(fname) as h5in:
        hits = h5in.root[group][table].read_where('dataset_id==datid')
        if df:
            return pd.DataFrame.from_records(hits)
        return hits
