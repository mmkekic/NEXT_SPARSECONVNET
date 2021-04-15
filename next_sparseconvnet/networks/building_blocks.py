import torch
import sparseconvnet as scn


class ResidualBlock_downsample(torch.nn.Module):
    def __init__(self, inplanes, kernel, stride, bias = False, dim = 3):
        torch.nn.Module.__init__(self)

        #f1
        self.bnr1    = scn.BatchNormReLU(inplanes)
        self.conv1   = scn.Convolution(dim, inplanes, 2 * inplanes, kernel, stride, bias)
        self.bnr2    = scn.BatchNormReLU(2 * inplanes)
        self.subconv = scn.SubmanifoldConvolution(dim, 2 * inplanes, 2 * inplanes, kernel, bias)

        #f2
        self.conv2   = scn.Convolution(dim, inplanes, 2 * inplanes, kernel, stride, bias)

        self.add     = scn.AddTable()

    def forward(self, x):
        x = self.bnr1(x)

        #f1
        y1 = self.conv1(x)
        y1 = self.bnr2(y1)
        y1 = self.subconv(y1)

        #f2
        y2 = self.conv2(x)

        #sum
        out = self.add([y1, y2])

        return out

class ResidualBlock_basic(torch.nn.Module):
    def __init__(self, inplanes,  kernel, dim=3):
        torch.nn.Module.__init__(self)
        self.bnr1 = scn.BatchNormReLU(inplanes)
        self.subconv1 = scn.SubmanifoldConvolution(dim, inplanes, inplanes, kernel, 0)
        self.bnr2 = scn.BatchNormReLU(inplanes)
        self.subconv2 = scn.SubmanifoldConvolution(dim, inplanes, inplanes, kernel, 0)
        self.add = scn.AddTable()

    def forward(self, x):
        y = self.bnr1(x)
        y = self.subconv1(y)
        y = self.bnr2(y)
        y = self.subconv2(y)
        x = self.add([x,y])

        return x
