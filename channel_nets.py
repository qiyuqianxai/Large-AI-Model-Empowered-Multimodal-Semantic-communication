from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pickle
from CGE import channel_estimation
import random
x_data = torch.load("X_data.pt")
y_data = torch.load("Y_data.pt")
H_data = torch.load("H_data.pt")

def fading_channel(x, snr, CGE = False):
    [batch_size, feature_length] = x.shape
    x = torch.reshape(x, (batch_size, -1, 2))
    x_com = torch.complex(x[:, :, 0], x[:, :, 1])
    x_fft = torch.fft.fft(x_com)
    index = random.choice(range(50000))
    H = H_data[index]
    H = H.permute(1, 0)
    H = H.view(batch_size, feature_length//2, 2)
    h_fft = torch.complex(H[..., 0], H[..., 1]).to("cuda")
    y_fft = h_fft * x_fft
    xpower = torch.sum(y_fft ** 2) / (feature_length * batch_size // 2)
    npower = xpower / 10 ** (snr / 10.0)
    n = torch.randn(batch_size, feature_length // 2, device="cuda") * npower
    y_add = y_fft + n
    if CGE:
        # channel estimation
        # prepare pilots
        x_pilot = x_data[index]
        y_pilot = y_data[index]
        # channel estimation using CGE
        h_p = channel_estimation(x_pilot, y_pilot,snr)
        h_p = h_p.view(batch_size, feature_length//2, 2)
        h_p_fft = torch.complex(h_p[..., 0], h_p[..., 1])
        y_add = y_add / h_p_fft
    y = torch.fft.ifft(y_add)
    y_tensor = torch.zeros((y.shape[0], y.shape[1], 2), device="cuda")
    y_tensor[:, :, 0] = y.real
    y_tensor[:, :, 1] = y.imag
    y_tensor = torch.reshape(y_tensor, (batch_size, feature_length))
    return y_tensor

class channel_net(nn.Module):
    def __init__(self, in_dims=512, mid_dims=128, snr=15, CGE=False):
        super(channel_net, self).__init__()
        self.enc_fc = nn.Linear(in_dims, mid_dims)
        self.dec_fc = nn.Linear(mid_dims, in_dims)
        self.snr = snr
        self.use_CGE = CGE

    def forward(self, x):
        ch_code = self.enc_fc(x)
        ch_code_with_n = fading_channel(ch_code,self.snr,self.use_CGE)
        x = self.dec_fc(ch_code_with_n)
        return ch_code,ch_code_with_n,x

from torchsummary import summary

if __name__ == '__main__':
    net = channel_net()
    summary(net,device="cpu")
