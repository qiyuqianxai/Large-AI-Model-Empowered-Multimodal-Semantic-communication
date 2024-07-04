import os
import random

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
## CGAN
## CGAN

# Define the Generator Network
class G_Net(nn.Module):
    def __init__(self, indim=128):
        super(G_Net, self).__init__()
        self.epsilon = 1e-7

        # First encoder network
        self.encoder_1 = nn.Sequential(
            nn.Conv1d(indim, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
        )

        # Second encoder network
        self.encoder_2 = nn.Sequential(
            nn.Conv1d(indim, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
        )

        # Intermediate layers
        self.mid_hiddens = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, indim, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.Tanh(),
        )

    def forward(self, x, y):
        # Combine the outputs of the two encoders and pass through intermediate and decoder layers
        z = (self.encoder_1(x) + self.encoder_2(y)) / 2
        z = self.mid_hiddens(z)
        z = self.decoder(z)
        return z


# Define the Discriminator Network
class D_Net(nn.Module):
    def __init__(self, indim=128):
        super(D_Net, self).__init__()

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv1d(indim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Output layer
        self.output = nn.Linear(16, 1)

    def forward(self, z):
        z = self.encoder(z)
        z = z.view(z.shape[0], -1)
        z = F.sigmoid(self.output(z))
        return z


# Channel State Information (CSI) generator function.
# Code reference from https://github.com/Azul-9/DeepLearningEnabledSemanticCommunicationSystems.git
def CSI_generator(num_sample, feature_length):
    P_hdB = np.array([0, -8, -17, -21, -25])  # Power characteristics of each channel (dB)
    D_h = [0, 3, 5, 6, 8]  # Delay for each channel (sampling points)
    P_h = 10 ** (P_hdB / 10)  # Power characteristics of each channel
    NH = len(P_hdB)  # Number of multi channels
    LH = D_h[-1] + 1  # Length of the channels (after delay)
    P_h = np.reshape(P_h, (len(D_h), 1))
    a = np.tile(np.sqrt(P_h / 2), num_sample)  # Generate Rayleigh stochastic variable
    A_h_I = np.random.rand(NH, num_sample) * a
    A_h_Q = np.random.rand(NH, num_sample) * a
    h_I = np.zeros((num_sample, LH))
    h_Q = np.zeros((num_sample, LH))
    i = 0
    for index in D_h:
        h_I[:, index] = A_h_I[i, :]
        h_Q[:, index] = A_h_Q[i, :]
        i += 1
    h = torch.complex(torch.tensor(h_I), torch.tensor(h_Q))
    h_fft = torch.fft.fft(h, feature_length // 2)
    return h_fft


# Function to save images
def save_imgs(H, H_p, e):
    plt.figure()
    plt.imshow(H)
    plt.axis("off")
    plt.savefig(f"H_{e}.jpg", bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(H_p)
    plt.axis("off")
    plt.savefig(f"H_p_{e}.jpg", bbox_inches='tight', pad_inches=0)


# Parameters class
class params():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr_G = 2e-3
    lr_D = 2e-3
    num_sample = 8
    batchsize = 1024
    epoch = 1000
    feature_length = 128
    snr = 15
    weight_path = "checkpoints"


# Instantiate parameters
args = params()


# Data generator function
def data_generator(args: params):
    print("Data generation...")
    b_x, b_y, b_H = [], [], []
    for i in range(50000):
        x_pilot = torch.ones((args.num_sample, args.feature_length // 2, 2))
        x_com = torch.complex(x_pilot[:, :, 0], x_pilot[:, :, 1])
        x_fft = torch.fft.fft(x_com)
        h_fft = CSI_generator(args.num_sample, args.feature_length)
        y_fft = h_fft * x_fft
        snr = 10 ** (args.snr / 10.0)
        xpower = torch.sum(y_fft ** 2) / (args.num_sample * args.feature_length // 2)
        npower = xpower / snr
        n = torch.randn(args.num_sample, args.feature_length // 2) * npower
        y_fft = y_fft + n

        x_fft2tensor = torch.stack((x_fft.real, x_fft.imag), dim=-1)
        y_fft2tensor = torch.stack((y_fft.real, y_fft.imag), dim=-1)
        h_fft2tensor = torch.stack((h_fft.real, h_fft.imag), dim=-1)

        b_x.append(x_fft2tensor.view(args.feature_length, args.num_sample).float())
        b_y.append(y_fft2tensor.view(args.feature_length, args.num_sample).float())
        b_H.append(h_fft2tensor.view(args.feature_length, args.num_sample).float())
    b_x = torch.stack(b_x, dim=0)
    b_y = torch.stack(b_y, dim=0)
    b_H = torch.stack(b_H, dim=0)
    # Save data to local storage
    torch.save(b_x, "X_data.pt")
    torch.save(b_y, "Y_data.pt")
    torch.save(b_H, "H_data.pt")
    return b_x, b_y, b_H


# Training function
def train(G, D, args: params):
    # Set loss functions
    criterion_1 = nn.L1Loss()
    criterion_2 = nn.BCEWithLogitsLoss()
    G.to(args.device)
    D.to(args.device)
    # Set optimizers
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr_G)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr_D)
    # Training
    x_data, y_data, H_data = data_generator(args)
    x_data = torch.load("X_data.pt")
    y_data = torch.load("Y_data.pt")
    H_data = torch.load("H_data.pt")
    for epoch in range(args.epoch):
        nmses = []
        training_size = x_data.shape[0]
        for i in range(0, training_size, args.batchsize):
            if i + args.batchsize < training_size:
                b_x = x_data[i:i + args.batchsize]
                b_y = y_data[i:i + args.batchsize]
                b_H = H_data[i:i + args.batchsize]
            else:
                b_x = x_data[i:training_size]
                b_y = y_data[i:training_size]
                b_H = H_data[i:training_size]

            b_x = b_x.to(args.device)
            b_y = b_y.to(args.device)
            b_H = b_H.to(args.device)

            # Train the Discriminator
            G.eval()
            D.train()
            b_H_p = G(b_x, b_y).detach()  # Generate fake samples
            True_labels = torch.ones((b_H_p.shape[0], 1))
            fake_labels = torch.zeros((b_H_p.shape[0], 1))
            True_labels = True_labels.to(args.device).detach()
            fake_labels = fake_labels.to(args.device).detach()
            D_p_1 = D(b_H)  # Discriminator output for real samples
            D_p_2 = D(b_H_p)  # Discriminator output for fake samples
            loss_D = criterion_2(D_p_1, True_labels) + criterion_2(D_p_2, fake_labels)
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # Train the Generator
            G.train()
            D.eval()
            b_H_p = G(b_x, b_y)  # Generate fake samples
            D_p = D(b_H_p)  # Discriminator output for fake samples
            loss_G = criterion_2(D_p, True_labels) + 100 * criterion_1(b_H_p, b_H)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Calculate NMSE (Normalized Mean Square Error)
            nmse = 10 * torch.log10(torch.var(b_H - b_H_p) / torch.var(b_H))
            nmses.append(nmse.cpu().detach().numpy())

            print(f'Epoch:{epoch + 1}, G Loss:{loss_G.item():.4f}, D Loss:{loss_D.item()}, NMSE:{np.mean(nmses)}')

            # Save model weights periodically
        torch.save(G.state_dict(), os.path.join(args.weight_path, f"G_{args.snr}.pth"))
        torch.save(D.state_dict(), os.path.join(args.weight_path, f"D_{args.snr}.pth"))

        if epoch % 100 == 0:
            H = b_H[0].cpu().detach().numpy()
            H_p = b_H_p[0].cpu().detach().numpy()
            save_imgs(H, H_p, epoch)

# Function for channel estimation
@torch.no_grad()
def channel_estimation(x, y, snr):
    G = G_Net(args.feature_length)
    weight = torch.load(os.path.join(args.weight_path, f"G_{snr}.pth"), map_location="cpu")
    G.load_state_dict(weight)
    G.to(args.device)
    G.eval()
    x = x.unsqueeze(0).to(args.device)
    y = y.unsqueeze(0).to(args.device)
    H_p = G(x, y)
    H_p = H_p.squeeze()
    H_p = H_p.permute(1, 0)
    return H_p


if __name__ == '__main__':
    G = G_Net(args.feature_length)
    D = D_Net(args.feature_length)
    # training
    train(G,D,args)

