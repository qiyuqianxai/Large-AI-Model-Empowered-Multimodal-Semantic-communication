from torchsummary import summary
import torch
from torch import nn
from channel_nets import channel_net


class base_net(nn.Module):
    def __init__(self, isc_model, channel_model):
        super(base_net, self).__init__()
        self.isc_model = isc_model
        self.ch_model = channel_model

    def forward(self,x):
        encoding = self.isc_model(x)
        encoding_with_noise = self.ch_model(encoding)
        decoding = self.isc_model(x, encoding_with_noise)
        return encoding,encoding_with_noise,decoding

if __name__ == '__main__':
    SC_model = VAE()
    channel_model = channel_net(M=3072)
    # summary(tst_model,(3,224,224),device="cpu")
    model = base_net(SC_model, channel_model).to("cuda")
    summary(model,(3,64,64),device="cuda")
