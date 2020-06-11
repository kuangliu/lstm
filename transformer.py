'''Simple transformer example.

Reference: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(self, cfg):
        super(TransformerModel, self).__init__()
        self.cfg = cfg
        in_channels = cfg['in_channels']
        mid_channels = cfg['mid_channels']
        out_channels = cfg['out_channels']

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(True),
            nn.Linear(mid_channels, mid_channels),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mid_channels, nhead=8)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg['num_layers'])
        self.decoder = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        out = self.encoder(x)
        out = self.transformer(out)
        out = self.decoder(out)
        return out


def test():
    cfg = {
        'in_channels': 4,
        'mid_channels': 128,
        'out_channels': 2,
        'num_layers': 8,
    }
    model = TransformerModel(cfg)

    batch = 1
    seq = 3
    in_channels = 4
    x = torch.randn(seq, batch, in_channels)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    test()
