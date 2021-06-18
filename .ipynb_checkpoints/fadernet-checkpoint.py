import os
import copy
import h5py as h5

import numpy as np
import pandas as pd
import nibabel as nib

from scipy import ndimage as nd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.utils.data as data

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold, 
                                     ShuffleSplit, LeaveOneGroupOut)
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()

device = 0


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Encoder3d(nn.Module):
    def __init__(self, 
                 img_shape=(1, 64, 64, 64), 
                 conv_model=[32, 64, 128, 256],
                 lrelu_slope=0.2,
                 noises=[0, 0, 0, 0]):
        super(self.__class__, self).__init__()
        img_shape = np.array(img_shape)
        self.n_layers = len(conv_model)
        
        self.model = []
        C_in = img_shape[0]
        for i, C_out in enumerate(conv_model):
            self.model += [
                nn.Dropout3d(noises[i]),
                nn.Conv3d(C_in, C_out, 4, 2, 1, bias=False),
                nn.BatchNorm3d(C_out),
                nn.LeakyReLU(lrelu_slope, inplace=True),
            ]
            C_in = C_out
            
#         print(C_out, "activations of size:", 
#               list(img_shape[1:] // (2 ** self.n_layers)))
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        return self.model(x)
    
class LatentDiscriminator(nn.Module):
    def __init__(self,
                 latent_shape, # takes decoder.latent_shape
                 n_attrs_outputs,
                 conv_model=[512],
                 fc_model=[],
                 dropout=0,
                 batchnorm=True):
        super().__init__()
        self.latent_shape = np.array(latent_shape)
        self.n_attrs_outputs = n_attrs_outputs
        self.n_outputs = sum(n_attrs_outputs)
        self.n_layers = len(conv_model)
        
        self.model = []
        C_in = latent_shape[0]
        for C_out in conv_model:
            self.model += [
                nn.Conv3d(C_in, C_out, 3, 2, 1, bias=False),
                nn.BatchNorm3d(C_out),
                nn.ReLU(inplace=True),
            ]
            C_in = C_out
            
        self.model += [
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            Flatten()
        ]
        
        n_in = C_in
        for n_out in fc_model:
            self.model += [nn.Dropout(dropout),
                           nn.Linear(n_in, n_out)]
            if batchnorm:
                self.model += [nn.BatchNorm1d(n_out)]
            self.model += [
                nn.ReLU(True),
            ]
            n_in = n_out
        self.model += [nn.Dropout(dropout),
                       nn.Linear(n_in, self.n_outputs)]
        self.model = nn.Sequential(*self.model)

    def forward(self, z):
        z = self.model(z)
        return z
    
###############################################################################    
    
class FaderDecoder3d(nn.Module):
    def __init__(self, 
                 latent_shape=(256, 4, 4, 4), 
#                  img_shape=(1, 64, 64, 64),
                 conv_model=[128, 64, 32, 1],
                 n_attrs_outputs=[],
                ):
        super(self.__class__, self).__init__()
        self.latent_shape = np.array(latent_shape)
#         img_shape = np.array(img_shape)
        self.n_layers = len(conv_model)
        self.n_attrs_outputs = n_attrs_outputs # list of n_outputs for each attr
        self.n_attrs_channels = sum(n_attrs_outputs)
        
        self.model = []
        C_in = latent_shape[0]
        for C_out in conv_model:
            self.model += [
                nn.ConvTranspose3d(C_in + self.n_attrs_channels, C_out, 4, 2, 1, bias=False),
                nn.BatchNorm3d(C_out),
                nn.ReLU(inplace=True),
            ]
            C_in = C_out
            
# logging activations
#         print(C_out, "activations of size:", 
#               list(self.latent_shape[1:] * (2 ** self.n_layers)))
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x, attrs):
        for i in range(len(self.model)):
            # we should add attrs channels to each x before apply conv
            # assume attrs here are already one-hot-encoded vectors
            # so just create 1s-channels and multiply them object-wise
            if i % 3 == 0:
                bs, C, h, w, d = x.size()
                attrs_C = torch.ones(bs, self.n_attrs_channels, h, w, d).to(x)
                attrs_C *= attrs.view(bs, self.n_attrs_channels, 1, 1, 1).float()
                x = torch.cat((x, attrs_C), dim=1) # channelwise
            x = self.model[i](x)
        return x
    
###############################################################################        

# encoder + decoder or aenc architecture

class FaderNet3d(nn.Module):
    def __init__(self, 
                 img_shape=(1, 64, 64, 64), 
                 enc_conv_model=[32, 64, 128, 256],
                 lrelu_slope=0.2,
                 noises=[0, 0, 0, 0, 0],
                 n_attrs_outputs=[],
                ):
        super(self.__class__, self).__init__()
        self.img_shape = np.array(img_shape)
        self.enc_conv_model = enc_conv_model
        self.dec_conv_model = enc_conv_model[-2::-1] + [img_shape[0]]
        self.n_layers = len(enc_conv_model)
        self.latent_shape = np.array([enc_conv_model[-1]] + list(self.img_shape[1:] // (2 ** self.n_layers)))
        self.lrelu_slope = lrelu_slope
        self.n_attrs_outputs = n_attrs_outputs
        
        self.encoder = nn.Sequential(
            Encoder3d(img_shape,
                      self.enc_conv_model,
                      lrelu_slope,
                      noises),
            nn.Conv3d(self.latent_shape[0], self.latent_shape[0], 3, 1, 1),
        )
        
        self.decoder = FaderDecoder3d(self.latent_shape, 
                                      self.dec_conv_model,
                                      n_attrs_outputs)
        self.decoder_output = nn.Sequential(
            nn.Conv3d(self.dec_conv_model[-1], 1, 3, 1, 1, bias=True),
#             nn.ReLU(inplace=True)
#             nn.Sigmoid() # -> normalize images to (0, 1)
        )
        
    def encode(self, x):
        z = self.encoder(x) # z repr is 4d tensor 
        return z
    
    def decode(self, z, attrs):
        x_rec = self.decoder(z, attrs)
        x_rec = self.decoder_output(x_rec)
        return x_rec

    def forward(self, x, attrs):        
        z = self.encode(x)
        x_rec = self.decode(z, attrs)
        return x_rec, z
    
def create_model(args):
    AE = FaderNet3d(
        args["img_shape"],
        args["conv_model"],
        args["lrelu_slope"],
        args["noises"],
        args["n_attrs_outputs"],
    )
    AE = AE.to(device)
    
    D = LatentDiscriminator(
        AE.decoder.latent_shape,
        args["n_attrs_outputs"],
        args["D_conv_model"],
        args["D_fc_model"],
        args["D_dropout"],
        args["D_batchnorm"]
    )
    D = D.to(device)
    
    AE_opt = torch.optim.Adam(AE.parameters(), lr=args["learning_rate"], 
                           weight_decay=args["weight_decay"])
    D_opt = torch.optim.Adam(D.parameters(), lr=args["D_learning_rate"], 
                           weight_decay=args["weight_decay"])
    return AE, D, AE_opt, D_opt