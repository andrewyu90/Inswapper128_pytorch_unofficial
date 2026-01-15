import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
from PIL import Image
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch import nn
from Model.Inswapper128_pytorch import Fusion_Encoder, Swapped_Face_Generator
from Model.arcface_resnet import resnet50, resnet_face18
import numpy as np
import pdb
#from Model.loss import GANLoss, AEI_Loss
batch_size = 1

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        x = x - self.mean
        x = x / self.std
        return x

class Inswapper128_Pytorch(pl.LightningModule):
    def __init__(self,source_dim):
        super(Inswapper128_Pytorch, self).__init__()
        self.source_dim = source_dim
        self.E = Fusion_Encoder(self.source_dim)
        self.G = Swapped_Face_Generator(1024,3)

   
    def forward(self, target, source):
        feature_map = self.E(target,source)
        output = self.G(feature_map)
        return output


class ArcFace_encoder(pl.LightningModule):
    def __init__(self, ema_path):
        super(ArcFace_encoder, self).__init__()
        self.resnet = resnet50()
        #pdb.set_trace()

        self.ema = nn.Linear(512, 512)
        self.ema.weight.data = torch.nn.Parameter(torch.from_numpy(np.load(ema_path)).permute(1,0)).cuda()
        self.ema.bias.data.fill_(0.0) 
        #self.ema = torch.from_numpy(np.load(ema_path)).cuda()

    def forward(self, inputs):
        x = self.resnet(inputs)
        x = self.ema(x)
        #x = torch.matmul(x,self.ema)
        out = torch.div(x, torch.linalg.norm(x, dim=1, keepdim=True))
        return out