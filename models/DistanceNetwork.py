import torch
import torch.nn as nn
import unittest
from utils import utility

device = torch.device('cuda:0')

class DistanceNetwork(nn.Module):
    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image, k_bits, test_only):

        """
        Produces pdfs over the support set classes for the target set image.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """
        similarities = []
        if not test_only:
            eps = 1e-10
            for support_image in support_set:
                sum_support = torch.sum(torch.pow(support_image, 2), 1)
                support_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
                dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
                cosine_similarity = dot_product * support_magnitude
                similarities.append(cosine_similarity)
            similarities = torch.stack(similarities)
        else:
            proj = utility.PROJECTION(input_image.size()[1],k_bits)
            input_embedding = proj.hash(input_image)
            print(input_embedding.size())
            import numpy as np
            np.save("5-5.npy", input_embedding.cpu().detach().numpy())
            exit()
            for support_image in support_set:
                support_embedding = proj.hash(support_image)
                hamming_similarity = utility.hamming_distance(input_embedding, support_embedding)
                similarities.append(hamming_similarity)
            similarities = torch.stack(similarities).to(device)
            # similarities = torch.stack(similarities)
        return similarities

class DistanceNetworkTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward(self):
        pass

if __name__ == '__main__':
    unittest.main()