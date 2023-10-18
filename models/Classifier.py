import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import unittest
import numpy as np
import math
from models.utilities.quant_ops import QuantConv2d, QuantAct,QuantLinear
from utils import utility

def convLayer(in_planes, out_planes, useDropout = False):
    "3x3 convolution with padding"
    seq = nn.Sequential(
        QuantConv2d(in_planes, out_planes, kernel_size=3,
                    stride=1, padding=1, bias=True),
        # nn.Conv2d(in_planes, out_planes, kernel_size=3,
        #           stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True),
        QuantAct(k_bits=8),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    if useDropout: # Add dropout module
        list_seq = list(seq.modules())[1:]
        list_seq.append(nn.Dropout(0.1))
        seq = nn.Sequential(*list_seq)

    return seq

class Classifier(nn.Module):
    def __init__(self, layer_size, k_bits, nClasses = 0, num_channels = 1, useDropout = False, image_size = 28):
        super(Classifier, self).__init__()

        """
        Builds a CNN to produce embeddings
        :param layer_sizes: A list of length 4 containing the layer sizes
        :param nClasses: If nClasses>0, we want a FC layer at the end with nClasses size.
        :param num_channels: Number of channels of images
        :param useDroput: use Dropout with p=0.1 in each Conv block
        """
        self.layer1 = convLayer(num_channels, 64, useDropout)
        self.layer2 = convLayer(64, 128, useDropout)
        # self.layer3 = convLayer(128, 64, useDropout)

        finalSize = int(math.floor(image_size / (2 * 2)))
        self.outSize = finalSize * finalSize * 128
        self.k_bits = k_bits
        self.layer3 = QuantLinear(self.outSize, 8)
        self.layer4 = QuantLinear(8, self.k_bits)
        # self.layer3 = nn.Linear(self.outSize, 8)
        # self.layer4 = nn.Linear(8, self.k_bits)
        # self.layer4 = nn.BatchNorm1d(self.k_bits)

        if nClasses>0: # We want a linear
            self.useClassification = True
            self.layer5 = nn.Linear(self.outSize,nClasses)
            self.outSize = nClasses
        else:
            self.useClassification = False

        # self.weights_init(self.layer1)
        # self.weights_init(self.layer2)
        # self.weights_init(self.layer3)
        # self.weights_init(self.layer4)

    def weights_init(self,module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image_input):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """

        x = self.layer1(image_input)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = x.view(x.size(0), -1)
        # noise = torch.randn_like(x)
        # x *= (1 + noise*0.05)
        if self.useClassification:
            x = self.layer5(x)
        return x


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward(self):
        pass

if __name__ == '__main__':
    unittest.main()
