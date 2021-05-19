import tables as tb
import numpy  as np
import pandas as pd
import torch
import warnings
from enum import auto

from invisible_cities.io   .dst_io  import load_dst
from invisible_cities.types.ic_types import AutoNameEnumBase

from . data_io import get_3d_input


class LabelType(AutoNameEnumBase):
    Classification = auto()
    Segmentation   = auto()


class DataGen_classification(torch.utils.data.Dataset):
    def __init__(self, labels, binsX, binsY, binsZ):
        self.binsX  = binsX
        self.binsY  = binsY
        self.binsZ  = binsZ
        self.labels = labels

    def __getitem__(self, idx):
        filename = self.labels.iloc[idx].filename
        event    = self.labels.iloc[idx].event
        label    = self.labels.iloc[idx].label
        x, y, z, ener = get_3d_input(filename, event, self.binsX, self.binsY, self.binsZ)
        return x, y, z, ener, [label], event #tener eventid puede ser util



class DataGen(torch.utils.data.Dataset):
    def __init__(self, filename, label_type, nevents=None):
        """ This class yields events from pregenerated MC file.
        Parameters:
            filename : str; filename to read
            table_name : str; name of the table to read
                         currently available BinClassHits and SegClassHits
        """
        self.filename   = filename
        if not isinstance(label_type, LabelType):
            raise ValueError(f'{label_type} not recognized!')
        self.label_type = label_type
        self.events     = load_dst(filename, 'DATASET', 'EventsInfo')
        if nevents is not None:
            if nevents>=len(self.events):
               warnings.warn(UserWarning(f'length of dataset smaller than {nevents}, using full dataset'))
            else:
                self.events = self.events.iloc[:nevents]
        #self.bininfo    = load_dst(filename, 'DATASET', 'BinsInfo')
        self.h5in = None

    def __getitem__(self, idx):
        idx_ = self.events.iloc[idx].dataset_id
        if self.h5in is None:#this opens a table once getitem gets called
            self.h5in = tb.open_file(self.filename, 'r')
        hits  = self.h5in.root.DATASET.Voxels.read_where('dataset_id==idx_')
        if self.label_type == LabelType.Classification:
            label = np.unique(hits['binclass'])
        elif self.label_type == LabelType.Segmentation:
            label = hits['segclass']
        return hits['xbin'], hits['ybin'], hits['zbin'], hits['energy'], label, idx_

    def __len__(self):
        return len(self.events)
    def __del__(self):
        if self.h5in is not None:
            self.h5in.close()

def collatefn(batch):
    coords = []
    energs = []
    labels = []
    events = torch.zeros(len(batch)).int()
    for bid, data in enumerate(batch):
        x, y, z, E, lab, event = data
        batchid = np.ones_like(x)*bid
        coords.append(np.concatenate([x[:, None], y[:, None], z[:, None], batchid[:, None]], axis=1))
        energs.append(E)
        labels.append(lab)
        events[bid] = event

    coords = torch.tensor(np.concatenate(coords, axis=0), dtype = torch.long)
    energs = torch.tensor(np.concatenate(energs, axis=0), dtype = torch.float).unsqueeze(-1)
    labels = torch.tensor(np.concatenate(labels, axis=0), dtype = torch.long)

    return  coords, energs, labels, events


def weights_loss_segmentation(fname, nevents):
    with tb.open_file(fname, 'r') as h5in:
        dataset_id = h5in.root.DATASET.Voxels.read_where('dataset_id<nevents', field='dataset_id')
        segclass   = h5in.root.DATASET.Voxels.read_where('dataset_id<nevents', field='segclass')

    df = pd.DataFrame({'dataset_id':dataset_id, 'segclass':segclass})
    nclass = max(df.segclass)+1
    df = df.groupby('dataset_id').segclass.apply(lambda x:np.bincount(x, minlength=nclass)/len(x))
    mean_freq = df.mean()
    inverse_freq = 1./mean_freq
    return inverse_freq/sum(inverse_freq)
