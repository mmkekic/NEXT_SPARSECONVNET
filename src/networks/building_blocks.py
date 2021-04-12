import torch
import sparseconvnet as scn


class ResidualBlock(torch.nn.Module):
    def __init__(self, inplanes,  kernel, stride):
        # f1:
        #BNRelu
        #Convolution (stride) outplanes = 2*inplanes
        #BNRELU
        #SubmanifoldConvolution

        # f2:
        # Convolution (stride) outplanes = 2*inplanes
        pass


    def forward(self, x):
        #f1(x)+f2(x)
        pass


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
