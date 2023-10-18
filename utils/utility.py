import os
import torch
import shutil
import math
import numpy as np
import torch.nn as nn

# device = torch.device('cuda:0')


def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    os.makedirs(checkpoint, exist_ok=True)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def hamming_distance(input_point, query_point):
    # same = torch.ones_like(input_point)*1000
    # diff = torch.ones_like(input_point)
    # res = torch.where(input_point == query_point, same, diff)
    # return torch.sum(res, dim=1)
    diff = torch.sum((input_point - query_point).abs()/2, dim=1)
    d = (input_point.size()[1] - diff) * 1000 + diff
    return d

def cosine_distance(input_point, query_point):
    return torch.sum(input_point*query_point,dim=1)*(torch.sum(input_point ** 2, dim=1).rsqrt()*torch.sum(query_point ** 2,dim=1).rsqrt())

def set_one(num_of_one, length):
    return torch.cat((torch.zeros(length - num_of_one), torch.ones(num_of_one)))


def TorchRound():
    """
    Apply STE to clamp function.
    """
    class identity_quant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            out = torch.round(input)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    return identity_quant().apply

class PROJECTION:
    def __init__(self,input_size, hash_size):
        self.device = torch.device('cuda:0')
        self.input_size = input_size
        self.hash_size = hash_size
        # self.uniform_planes = torch.randn(input_size, hash_size)
        self.uniform_planes = torch.Tensor(np.random.uniform(high=1,low=-1,size=(input_size, hash_size))).to(self.device)
        self.round = TorchRound()

    def hash(self, input_point):
        # lvl = self.hash_size // 8
        batch_projections = []
        for i in range(input_point.size()[0]):
            projections = torch.matmul(input_point[i], self.uniform_planes)
            # projections = input_point[i] / input_point[i].abs()
            zero = torch.zeros_like(projections)
            one = torch.ones_like(projections)
            projections = torch.where(projections > 0, one, projections)
            projections = torch.where(projections < 0, zero, projections)
            batch_projections.append(projections)
        # for i in range(input_point.size()[0]):
        #     projections = None
        #     quant_num = self.quant(input_point[i], lvl, input_point.max().item(), input_point.min().item())
        #     if projections is None:
        #         projections = quant_num
        #     else:
        #         projections = torch.cat((projections, quant_num), 1)
        #     batch_projections.append(projections)
        return torch.stack(batch_projections)

    def quant(self, input, lvl, max, min):
        input = input.detach().cpu() - min
        max = max - min
        quant_num = self.round(input / max * lvl)
        # res = torch.zeros(lvl * input.size()[0])
        res = torch.zeros(lvl * input.size()[0]).to(device)
        for i in range(input.size()[0]):
            num_of_one = int(quant_num[i])
            res[lvl*i:lvl*i+lvl] = set_one(num_of_one, lvl)
        return res


def proj(img):
    zero = torch.zeros_like(img)
    one = torch.ones_like(img)
    projections = torch.where(img > 0, one, img)
    projections = torch.where(img < 0, zero, projections)
    return projections
