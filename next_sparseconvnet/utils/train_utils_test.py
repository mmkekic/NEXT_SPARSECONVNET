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



def test_train_one_epoch_segmentation(MCdataset):
    datagen = DataGen(MCdataset, LabelType.Segmentation, nevents = 3)
    loader = torch.utils.data.DataLoader(datagen, batch_size = 1, shuffle = True, num_workers=1, collate_fn=collatefn, drop_last=True, pin_memory=False)

    spatial_size = (51, 51, 51)
    init_conv_nplanes = 4
    init_conv_kernel = 3
    kernel_sizes = [7, 5, 3]
    stride_sizes = [2, 2]
    basic_num = 3
    nclasses = 3

    net = UNet(spatial_size, init_conv_nplanes, init_conv_kernel, kernel_sizes, stride_sizes, basic_num, nclasses = nclasses)
    net = net.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-6, weight_decay=0)

    train_one_epoch_segmentation(0, net, criterion, optimizer, loader)



def test_valid_one_epoch_segmentation(MCdataset):
    datagen = DataGen(MCdataset, LabelType.Segmentation, nevents = 3)
    loader = torch.utils.data.DataLoader(datagen, batch_size = 1, shuffle = True, num_workers=1, collate_fn=collatefn, drop_last=True, pin_memory=False)

    spatial_size = (51, 51, 51)
    init_conv_nplanes = 4
    init_conv_kernel = 3
    kernel_sizes = [7, 5, 3]
    stride_sizes = [2, 2]
    basic_num = 3
    nclasses = 3

    net = UNet(spatial_size, init_conv_nplanes, init_conv_kernel, kernel_sizes, stride_sizes, basic_num, nclasses = nclasses)
    net = net.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    valid_one_epoch_segmentation(net, criterion, loader)
