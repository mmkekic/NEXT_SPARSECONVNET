"""
This script creates hdf5 files that contains:
 - DATASET/BinClassHits - voxelized hits table with labels (binary classification)
 - DATASET/SegClassHits - voxelized hits table with labels (segmentation)
 - DATASET/BinInfo      - table that stores info about bins
 - DATASET/EventsInfo   - table that contains EventID and binary classification label
"""
import sys
import os
import tables as tb
import numpy  as np
import pandas as pd

from glob import glob
from invisible_cities.io import mcinfo_io as mio
from invisible_cities.io import dst_io    as dio

from invisible_cities.core.configure import configure

import dataset_labeling_utils as utils


def get_tables(filename, config):
    min_x, max_x = config.xlim
    min_y, max_y = config.ylim
    min_z, max_z = config.zlim
    bins_x = np.linspace(min_x, max_x, config.nbins_x)
    bins_y = np.linspace(min_y, max_y, config.nbins_y)
    bins_z = np.linspace(min_z, max_z, config.nbins_z)
    bins = (bins_x, bins_y, bins_z)

    hits      = mio.load_mchits_df(filename)
    particles = mio.load_mcparticles_df(filename)


    hits_clf = utils.add_clf_labels(hits, particles)
    hits_clf = utils.get_bin_indices(hits_clf, bins, label = 'binclass')

    if config.segmentation:
        hits_seg = utils.add_seg_labels(hits, particles, delta_t=config.blob_delta_t, delta_e=config.blob_delta_e)
        hits_seg = utils.get_bin_indices(hits_seg, bins, label = 'segclass')
    else:
        hits_seg = None

    eventInfo = hits_clf[['event_id', 'binclass']].drop_duplicates().reset_index(drop=True)

    binsInfo = pd.Series({'min_x'   : min_x ,
                          'max_x'   : max_x ,
                          'nbins_x' : config.nbins_x,
                          'min_y'   : min_y ,
                          'max_y'   : max_y ,
                          'nbins_y' : config.nbins_y,
                          'min_z'   : min_z ,
                          'max_z'   : max_z ,
                          'nbins_z' : config.nbins_z,
                          }).to_frame().T

    return eventInfo, binsInfo, hits_clf, hits_seg


if __name__ == "__main__":
    config  = configure(sys.argv).as_namespace
    filesin = glob(os.path.expandvars(config.files_in))
    for f in filesin:
        eventInfo, binsInfo, hits_clf, hits_seg = get_tables(f, config)
        with tb.open_file(os.path.expandvars(config.file_out), 'w') as h5out:
            dio.df_writer(h5out, eventInfo, 'DATASET', 'EventsInfo', columns_to_index=['event_id'])
            dio.df_writer(h5out, binsInfo, 'DATASET', 'BinsInfo')
            if hits_clf is not None:
                dio.df_writer(h5out, hits_clf, 'DATASET', 'BinClassHits', columns_to_index=['event_id'])
            if hits_seg is not None:
                dio.df_writer(h5out, hits_seg, 'DATASET', 'SegClassHits', columns_to_index=['event_id'])
