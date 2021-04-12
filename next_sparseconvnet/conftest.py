import pytest
import pandas as pd

@pytest.fixture(scope = 'session')
def labels_df():
    return pd.read_csv('./test_files/cut_labels.cvs')
