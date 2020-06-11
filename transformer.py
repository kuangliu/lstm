'''Simple transformer example.

Reference: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=8):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, cfg):
        super(TransformerModel, self).__init__()
        self.cfg = cfg
        in_channels = cfg['in_channels']
        mid_channels = cfg['mid_channels']
        out_channels = cfg['out_channels']

        self.encoder = nn.Linear(in_channels, mid_channels)
        self.emb = PositionalEncoding(mid_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mid_channels, nhead=8)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg['num_layers'])
        self.decoder = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        out = self.encoder(x)
        out = self.emb(out)
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
