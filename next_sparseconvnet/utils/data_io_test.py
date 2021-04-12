from .data_io import *

def test_get_mc_hits(labels_df):
    filename, event = labels_df.iloc[0][['filename', 'event']]
    mchits = get_mchits(filename, event)

    assert mchits.shape[1]==4

def test_get_3d_input(labels_df):

    binsX = binsY = np.linspace(-200, 200, 11)
    binsZ = np.linspace(0, 550, 11)
    filename, event = labels_df.iloc[0][['filename', 'event']]
    data = get_3d_input(filename, event, binsX, binsY, binsZ)
    assert len(data) ==4

    assert data[0].dtype == np.int64
    assert data[1].dtype == np.int64
    assert data[2].dtype == np.int64
    assert data[3].dtype == np.float
    assert len(data[0])==len(data[1])==len(data[2])==len(data[3])
