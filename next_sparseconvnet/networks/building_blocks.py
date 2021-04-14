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


class ResidualBlock(torch.nn.Module):
    def __init__(self, inplanes,  kernel):
        # f
        #BNRelu
        #SubmanifoldConvolution
        #BNRELU
        #SubmanifoldConvolution
        pass


    def forward(self, x):
        #x+f(x)
        pass
