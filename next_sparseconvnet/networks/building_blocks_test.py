import pytest
import torch
import sparseconvnet as scn

from next_sparseconvnet.utils.data_loaders import DataGen, collatefn, LabelType
from .building_blocks import *

def test_ResidualBlock_downsample(MCdataset):
    datagen = DataGen(MCdataset, LabelType.Classification)
    data = [datagen[i] for i in range(3)]
    coord, ener, lab, ev = collatefn(data)
    spatial_size = (50, 50, 50)
    dim = 3
    inplanes = 1
    kernel = 2
    stride = 2

    x = scn.InputLayer(dim, spatial_size)((coord, ener))
    out = ResidualBlock_downsample(inplanes, kernel, stride)(x)

    assert out.features.shape[1] == 2 * inplanes
    for i, size in enumerate(spatial_size):
        assert out.spatial_size[i] == (size - kernel)/stride + 1


def test_ResidualBlock_basic(MCdataset):
    datagen = DataGen(MCdataset, LabelType.Classification)
    data = [datagen[i] for i in range(3)]
    coord, ener, lab, ev = collatefn(data)
    spatial_size = (50, 50, 50)
    dim = 3
    inplanes = 1
    kernel = 2

    x = scn.InputLayer(dim, spatial_size)((coord, ener))
    out = ResidualBlock_basic(inplanes, kernel)(x)

    assert out.features.shape[1] == inplanes
    for i, size in enumerate(spatial_size):
        assert out.spatial_size[i] == size


def test_ResidualBlock_upsample(MCdataset):
    datagen = DataGen(MCdataset, LabelType.Classification)
    data = [datagen[i] for i in range(3)]
    coord, ener, lab, ev = collatefn(data)
    spatial_size = (50, 50, 50)
    dim = 3
    inplanes = 1
    outplanes = 6
    kernel = 2
    stride = 2

    x = scn.InputLayer(dim, spatial_size)((coord, ener))
    x = scn.SubmanifoldConvolution(dim, inplanes, outplanes, kernel, False)(x)

    inplanes = x.features.shape[1]
    out = ResidualBlock_upsample(inplanes, kernel, stride)(x)

    assert out.features.shape[1] == inplanes / 2
    for i, size in enumerate(spatial_size):
        assert out.spatial_size[i] == kernel + stride * (size - 1)


def test_ConvBNBlock(MCdataset):
    datagen = DataGen(MCdataset, LabelType.Classification)
    data = [datagen[i] for i in range(3)]
    coord, ener, lab, ev = collatefn(data)
    spatial_size = (50, 50, 50)
    dim = 3
    inplanes = 1
    outplanes = 3
    kernel = 2
    stride = 2

    x = scn.InputLayer(dim, spatial_size)((coord, ener))
    out = ConvBNBlock(inplanes, outplanes, kernel)(x)
    out_with_stride = ConvBNBlock(inplanes, outplanes, kernel, stride = stride)(x)

    for i, size in enumerate(spatial_size):
        assert out.spatial_size[i] == size
    assert out_with_stride.spatial_size[i] == (size - kernel) / stride + 1


def test_calculate_ouput_dimension():
    spatial_size = (561, 561, 561)
    kernel_sizes = [9, 7, 5, 3, 3]
    stride_sizes = [4, 2, 2, 2]
    with pytest.raises(AssertionError):
        calculate_output_dimension(spatial_size, kernel_sizes, stride_sizes)

    spatial_size = (289, 417, 449)
    assert calculate_output_dimension(spatial_size, kernel_sizes, stride_sizes) == [7, 11, 12]
