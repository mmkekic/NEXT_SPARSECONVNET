import pytest
import os
import pandas as pd

@pytest.fixture(scope = 'session')
def tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('output_files')


@pytest.fixture(scope = 'session')
def TEST_DATA():
    return os.environ['TEST_DATA']

@pytest.fixture(scope = 'session')
def MCdataset(TEST_DATA):
    return os.path.join(TEST_DATA, "MC_dataset.h5")
