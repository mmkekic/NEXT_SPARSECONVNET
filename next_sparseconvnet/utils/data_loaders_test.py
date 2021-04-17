import pytest

from . data_loaders import *

def test_DataGen_bin(MCdataset):
    datagen = DataGen(MCdataset, 'BinClassHits')
    with datagen as dg:
        data = datagen[0]
    assert len(data) == 6 #x, y, z, ener, label, event

    assert data[0].dtype == np.int64
    assert data[1].dtype == np.int64
    assert data[2].dtype == np.int64
    assert data[3].dtype == np.float
    assert len(data[0])==len(data[1])==len(data[2])==len(data[3])

    assert data[4][0] in [0, 1]
    assert isinstance(data[5], np.int64)

def test_DataGen_seg(MCdataset):
    datagen = DataGen(MCdataset, 'SegClassHits')
    with datagen as dg:
        data = datagen[0]
    assert len(data) == 6 #x, y, z, ener, label, event

    assert data[0].dtype == np.int64
    assert data[1].dtype == np.int64
    assert data[2].dtype == np.int64
    assert data[3].dtype == np.float
    assert len(data[0])==len(data[1])==len(data[2])==len(data[3]==len(data[4]))

    assert data[4].dtype == np.int
    assert set(np.unique(data[4])) == set([0, 1, 2])
    assert isinstance(data[5], np.int64)


def test_collatefn_bin(MCdataset):
    datagen = DataGen(MCdataset, 'BinClassHits')
    with datagen as dg:
        data = [datagen[i] for i in range(3)]
    batch = collatefn(data)

    assert len(batch) == 4 #coords, energies, labels, events
    coords = batch[0]
    energs = batch[1]
    labels = batch[2]
    assert coords.shape[1] == 4 #x, y, z, bid
    assert coords.dtype == torch.long
    assert coords.shape[0] == energs.shape[0]
    assert energs.dtype == torch.float
    assert len(labels) == 3

def test_collatefn_seg(MCdataset):
    datagen = DataGen(MCdataset, 'SegClassHits')
    with datagen as dg:
        data = [datagen[i] for i in range(3)]
    batch = collatefn(data)

    assert len(batch) == 4 #coords, energies, labels, events
    coords = batch[0]
    energs = batch[1]
    labels = batch[2]
    assert coords.shape[1] == 4 #x, y, z, bid
    assert coords.dtype == torch.long
    assert coords.shape[0] == energs.shape[0] == labels.shape[0]
    assert energs.dtype == torch.float
