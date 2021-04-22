import torch
import sparseconvnet as scn


class ResidualBlock_downsample(torch.nn.Module):
    def __init__(self, inplanes, kernel, stride, bias = False, dim = 3):
        torch.nn.Module.__init__(self)

        outplanes = 2 * inplanes

        #f1
        self.bnr1    = scn.BatchNormReLU(inplanes)
        self.conv1   = scn.Convolution(dim, inplanes, outplanes, kernel, stride, bias)
        self.bnr2    = scn.BatchNormReLU(outplanes)
        self.subconv = scn.SubmanifoldConvolution(dim, outplanes, outplanes, kernel, bias)

        #f2
        self.conv2   = scn.Convolution(dim, inplanes, outplanes, kernel, stride, bias)

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

class ResidualBlock_upsample(torch.nn.Module):
    def __init__(self, inplanes, kernel, stride, bias = False, dim = 3):
        torch.nn.Module.__init__(self)

        outplanes = int(inplanes / 2)

        #f1
        self.bnr1      = scn.BatchNormReLU(inplanes)
        self.deconv1   = scn.Deconvolution(dim, inplanes, outplanes, kernel, stride, bias)
        self.bnr2      = scn.BatchNormReLU(outplanes)
        self.subconv   = scn.SubmanifoldConvolution(dim, outplanes, outplanes, kernel, bias)

        #f2
        self.deconv2   = scn.Deconvolution(dim, inplanes, outplanes, kernel, stride, bias)

        self.add       = scn.AddTable()

    def forward(self, x):
        x = self.bnr1(x)

        #f1
        y1 = self.deconv1(x)
        y1 = self.bnr2(y1)
        y1 = self.subconv(y1)

        #f2
        y2 = self.deconv2(x)

        #sum
        out = self.add([y1, y2])

        return out


class ConvBNBlock(torch.nn.Module):
    def __init__(self, inplanes, outplanes, kernel, stride = 1, bias = False, dim = 3):
        torch.nn.Module.__init__(self)
        if stride==1:
            self.conv = scn.SubmanifoldConvolution(dim, inplanes, outplanes, kernel, bias)
        else:
            self.conv = scn.Convolution(dim, inplanes, outplanes, kernel, stride, bias)
        self.bnr = scn.BatchNormReLU(outplanes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bnr(x)
        return x
