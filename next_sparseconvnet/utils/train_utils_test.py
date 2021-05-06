import torch
import numpy as np
from .data_loaders import DataGen, collatefn, LabelType
from next_sparseconvnet.networks.architectures import UNet
from .train_utils import *

def test_IoU():
    a = [0, 1, 1, 2, 0, 0, 2]
    b = [1, 1, 2, 0, 0, 1, 2]
    iou_by_hand = [1/4, 1/4, 1/3]
    iou = IoU(a, b)
    np.testing.assert_allclose(iou, iou_by_hand)

    n = 5
    iou = IoU(a, b, nclass = n)
    assert len(iou) == n
