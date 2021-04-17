
import torch
import sparseconvnet as scn
from .building_blocks import ResidualBlock_downsample, ResidualBlock_basic, ResidualBlock_upsample

class UNet(torch.nn.Module):
        def __init__(self, spatial_size, inplanes, kernel, stride, basic_num, level_depth, dim = 3):
            torch.nn.Module.__init__(self)

            self.basic_num = basic_num
            self.level_depth = level_depth

            self.inp = scn.InputLayer(dim, spatial_size)

            #Creo las layers de la estructura en U con un par de bucles

            self.downsample = []
            self.upsample = []
            self.basic_down = []
            self.basic_up = []

            for i in range(level_depth):
                for j in range(basic_num):
                    if i != level_depth - 1: #es decir, mientras no llegue al nivel final, hacer 3 basic en ambos lados
                        self.basic_down.append(ResidualBlock_basic(inplanes, kernel))
                        self.basic_up.append(ResidualBlock_basic(inplanes, kernel))
                    else: #cuando ya estemos en el último nivel, basta con añadirle 3 basic solo a una de las ramas...
                        self.basic_down.append(ResidualBlock_basic(inplanes, kernel))
                if i != level_depth - 1: #lo pongo porque si no me añade en la lista un downsample y un upsample del último nivel (el que considero yo último vamos, en el que solo deberia haber basic)
                    self.downsample.append(ResidualBlock_downsample(inplanes, kernel, stride)) #lista de los downsamples
                    self.upsample.append(ResidualBlock_upsample(2 * inplanes, kernel, stride)) #lista de los upsamples, OJO porque está al revés (el primero que debería aplicar es el último de la lista)

                inplanes = inplanes * 2 #en el siguiente nivel tendremos el doble de inplanes...

            self.add = scn.AddTable()

        def forward(self, x):
            x = self.inp(x)

            out_layers = []
            c = 0 #contador para recorrer los basic
            for i in range(self.level_depth):
                for j in range(self.basic_num):
                    x = self.basic_down[c](x)
                    if j == self.basic_num - 1 and i != self.level_depth - 1:
                        out_layers.append(x) #guardamos el output tras pasarlo varias veces por el bloque basico
                    c = c + 1
                if i != self.level_depth - 1: #es decir, cuando lleguemos al ultimo nivel ya no hacemos downsample...
                    x = self.downsample[i](x)

            c = 1 #reseteo contador de lista basic (ahora de atrás hacia delante)

            for i in range(self.level_depth - 1): #ahora en la subida hay como un nivel menos por así decirlo...
                x = self.upsample[-(i+1)](x) #hago upsample
                x = self.add([x, out_layers[-(i+1)]]) #aqui sumo mi salida con la salida guardada...
                for j in range(self.basic_num):
                    x = self.basic_up[-c](x)
                    c = c + 1

            return x
