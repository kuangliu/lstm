'''Simple LSTM example.

Reference: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(self, cfg):
        super(RNNModel, self).__init__()
        self.cfg = cfg
        in_channels = cfg['in_channels']
        mid_channels = cfg['mid_channels']
        out_channels = cfg['out_channels']

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(True),
            nn.Linear(mid_channels, mid_channels),
        )
        self.rnn = nn.LSTM(mid_channels, mid_channels,
                           cfg['num_layers'], batch_first=True)
        self.decoder = nn.Linear(mid_channels, out_channels)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = (
            weight.new_zeros(self.cfg['num_layers'],
                             batch_size, self.cfg['mid_channels']),
            weight.new_zeros(self.cfg['num_layers'],
                             batch_size, self.cfg['mid_channels']),
        )
        return hidden

    def forward(self, x, hidden):
        out = self.encoder(x)
        out, hidden = self.rnn(out, hidden)
        out = self.decoder(out)
        return out


def test():
    cfg = {
        'in_channels': 4,
        'mid_channels': 128,
        'out_channels': 2,
        'num_layers': 2,
    }
    model = RNNModel(cfg)

    batch = 1
    seq = 3
    in_channels = 4
    x = torch.randn(batch, seq, in_channels)
    hidden = model.init_hidden(batch)
    y = model(x, hidden)
    print(y.shape)


if __name__ == '__main__':
    test()
