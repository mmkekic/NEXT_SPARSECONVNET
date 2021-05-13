#!/usr/bin/env python
"""
This script merges dataset files such that dataset_id is always increasing consecutive integers array.
The final file contains
 - DATASET/BinClassHits - voxelized hits table with labels (binary classification)
 - DATASET/SegClassHits - voxelized hits table with labels (segmentation)
 - DATASET/BinInfo      - table that stores info about bins
 - DATASET/EventsInfo   - table that contains EventID and binary classification label
"""


import tables as tb
from glob import glob
import re
from invisible_cities.core  .configure import configure


if __name__ == "__main__":
    config  = configure(sys.argv).as_namespace
    filesin = glob(os.path.expandvars(config.files_in))
    files_to_merge = sorted(filesin, key = lambda x:int(re.findall(r'\d+',x)[0]))
    fout = os.path.expandvars(config.file_out)

    with tb.open_file(files_to_merge[0], 'r') as h5in:
        h5in.copy_file(fout, overwrite=True)

    #set dataset_id to start from 0 in h5out

    with tb.open_file(fout, 'r+') as h5out:
        min_dataset_id = h5out.root.DATASET.EventsInfo.cols.dataset_id[0]

        if min_dataset_id>0:
            h5out.root.DATASET.EventsInfo.cols.dataset_id[:]-=min_dataset_id
            h5out.root.DATASET.Voxels.cols.dataset_id[:]-=min_dataset_id
            h5out.root.DATASET.EventsInfo.flush()
            h5out.root.DATASET.Voxels.flush()


    for filein in files_to_merge[1:]:
        with tb.open_file(fout, 'a') as h5out:
            with tb.open_file(filein, 'r') as h5in:
                prev_id =  h5out.root.DATASET.EventsInfo.cols.dataset_id[-1]+1

                evs = h5in.root.DATASET.EventsInfo[:]
                file_start_id = evs['dataset_id'][0]
                evs['dataset_id']+=prev_id-file_start_id
                h5out.root.DATASET.EventsInfo.append(evs)
                h5out.root.DATASET.EventsInfo.flush()
                del(evs)
                voxs = h5in.root.DATASET.Voxels[:]
                voxs['dataset_id']+=prev_id
                h5out.root.DATASET.Voxels.append(voxs)
                h5out.root.DATASET.Voxels.flush()
                del(voxs)

    with tb.open_file(fout, 'r+') as h5out:
        h5out.root.DATASET.EventsInfo.cols.dataset_id.create_index()
        h5out.root.DATASET.Voxels.cols.dataset_id.create_index()
