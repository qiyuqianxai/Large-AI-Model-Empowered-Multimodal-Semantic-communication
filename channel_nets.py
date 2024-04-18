from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pickle

def AWGN_channel(x, snr):  # used to simulate additive white gaussian noise channel
    [batch_size, length] = x.shape
    x_power = torch.sum(torch.abs(x)) / (batch_size * length)
    n_power = x_power / (10 ** (snr / 10.0))
    noise = torch.rand(batch_size, length, device="cuda") *n_power
    return x + noise

def fading_channel(x, h_I, h_Q, snr,CGE = False):
    [batch_size, length, feature_length] = x.shape
    x = torch.reshape(x, (batch_size, -1, 2))
    x_com = torch.complex(x[:, :, 0], x[:, :, 1])
    x_fft = torch.fft.fft(x_com)
    h = torch.complex(torch.tensor(h_I), torch.tensor(h_Q))
    h_fft = torch.fft.fft(h, feature_length * length//2).to("cuda")
    y_fft = h_fft * x_fft
    snr = 10 ** (snr / 10.0)
    xpower = torch.sum(y_fft ** 2) / (length * feature_length * batch_size // 2)
    npower = xpower / snr
    n = torch.randn(batch_size, feature_length * length // 2, device="cuda") * npower
    y_add = y_fft + n
    # print(x.shape, h_fft.shape, x_fft.shape, y_add.shape)
    if CGE:
        # channel estimation
        h_p = (y_add, x_fft)
        y_add = y_add / h_p
    # y_add = y_add / h_fft
    y = torch.fft.ifft(y_add)
    y_tensor = torch.zeros((y.shape[0], y.shape[1], 2), device="cuda")
    y_tensor[:, :, 0] = y.real
    y_tensor[:, :, 1] = y.imag
    y_tensor = torch.reshape(y_tensor, (batch_size, length, feature_length))
    return y_tensor

def multipath_generator(num_sample):
    P_hdB = np.array([0, -8, -17, -21, -25])  # Power characteristics of each channels(dB)
    D_h = [0, 3, 5, 6, 8]  # Each channel delay(sampling point)
    P_h = 10 ** (P_hdB / 10)  # Power characteristics of each channels
    NH = len(P_hdB)  # Number of the multi channels
    LH = D_h[-1] + 1  # Length of the channels(after delaying)
    P_h = np.reshape(P_h, (len(D_h), 1))
    a = np.tile(np.sqrt(P_h / 2), num_sample)  # generate rayleigh stochastic variable
    A_h_I = np.random.rand(NH, num_sample) * a
    A_h_Q = np.random.rand(NH, num_sample) * a
    h_I = np.zeros((num_sample, LH))
    h_Q = np.zeros((num_sample, LH))

    i = 0
    for index in D_h:
        h_I[:, index] = A_h_I[i, :]
        h_Q[:, index] = A_h_Q[i, :]
        i += 1

    return h_I, h_Q

class channel_net(nn.Module):
    def __init__(self, in_dims=512, mid_dims=128, snr=25, CGE=False):
        super(channel_net, self).__init__()
        self.enc_fc = nn.Linear(in_dims, mid_dims)
        self.dec_fc = nn.Linear(mid_dims, in_dims)
        self.snr = snr
        self.use_CGE = CGE

    def forward(self, x, h_I, h_Q):
        ch_code = self.enc_fc(x)
        ch_code_with_n = fading_channel(ch_code,h_I,h_Q,self.snr,self.use_CGE)
            # ch_code_with_n/=H_p
        x = self.dec_fc(ch_code_with_n)
        return ch_code,ch_code_with_n,x

class MutualInfoSystem(nn.Module):  # mutual information used to maximize channel capacity
    def __init__(self):
        super(MutualInfoSystem, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, inputs):
        output = F.relu(self.fc1(inputs))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        return output

def sample_batch(batch_size, sample_mode, x, y):  # used to sample data for mutual info system
    length = x.shape[0]
    if sample_mode == 'joint':
        index = np.random.choice(range(length), size=batch_size, replace=False)
        batch_x = x[index, :]
        batch_y = y[index, :]
    elif sample_mode == 'marginal':
        joint_index = np.random.choice(range(length), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(length), size=batch_size, replace=False)
        batch_x = x[joint_index, :]
        batch_y = y[marginal_index, :]
    batch = torch.cat((batch_x, batch_y), 1)
    return batch

from torchsummary import summary

if __name__ == '__main__':
    net = channel_net()
    summary(net,device="cpu")
