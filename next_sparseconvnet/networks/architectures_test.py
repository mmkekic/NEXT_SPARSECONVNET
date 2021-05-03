import pytest
import torch
import sparseconvnet as scn

from next_sparseconvnet.utils.data_loaders import DataGen, collatefn, LabelType
from .architectures import UNet


def test_UNet(MCdataset):
    datagen = DataGen(MCdataset, LabelType.Classification)
    data = [datagen[i] for i in range(3)]
    coord, ener, lab, ev = collatefn(data)
    spatial_size = (51, 51, 51)
    init_conv_nplanes = 4
    init_conv_kernel = 3
    kernel_sizes = [7, 5, 3]
    stride_sizes = [2, 2]
    basic_num = 3
    nclasses = 3

    net = UNet(spatial_size, init_conv_nplanes, init_conv_kernel, kernel_sizes, stride_sizes, basic_num, nclasses = nclasses)

    last_basic = []
    net.basic_up[0][2].add.register_forward_hook(lambda model, input, output: last_basic.append([output.spatial_size, output.features.shape]))

    assert len(net.downsample) == len(kernel_sizes) - 1
    assert len(net.upsample)   == len(kernel_sizes) - 1
    assert len(net.basic_down) == len(kernel_sizes) - 1
    assert len(net.basic_up)   == len(kernel_sizes) - 1
    assert len(net.basic_down[0]) == basic_num

    out = net.forward((coord, ener))

    for i, size in enumerate(last_basic[0][0]):
        assert size == spatial_size[i]
    assert last_basic[0][1][1] == init_conv_nplanes
    assert out.size()[0] == coord.size()[0]
    assert out.size()[1] == nclasses
