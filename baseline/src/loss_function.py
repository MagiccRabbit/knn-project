import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Some ideas taken from https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py

class AAM_loss(nn.Module):
    def __init__(self, embed_dim, n_speakers, scale = 30.0, margin = 0.2):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.n_speakers = n_speakers
        self.embed_dim = embed_dim
        self.W = nn.Parameter(torch.FloatTensor(self.embed_dim, self.n_speakers), requires_grad=True)
        nn.init.xavier_normal_(self.W,gain= 1)

        # Precomputation
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self,x , labels):
        x = F.normalize(x, dim=1)
        W = F.normalize(self.W, dim=1)


        cos = F.linear(x, W) 

        sin = torch.sqrt((1.0 - torch.mul(cos,cos)).clamp(0,1))

        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)

        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cos)
        output *= self.scale

        loss = F.cross_entropy(output, labels)

        return loss
    