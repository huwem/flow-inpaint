import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freq = torch.exp(-torch.arange(half_dim, dtype=torch.float32) * 
                         torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        angle = t[:, None] * freq[None, :]
        embedding = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        return embedding

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, width=64, num_blocks=2):
        super().__init__()
        self.time_embed = TimeEmbedding(32)
        self.width = width

        self.input_conv = nn.Conv2d(in_channels * 2, width, 3, padding=1)

        self.enc = nn.ModuleList([
            nn.ModuleList([ConvBlock(width, width) for _ in range(num_blocks)]),
            nn.ModuleList([ConvBlock(width, width * 2) for _ in range(num_blocks)]),
            nn.ModuleList([ConvBlock(width * 2, width * 4) for _ in range(num_blocks)])
        ])
        self.downsample = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(width * 4, width * 4)

        self.dec = nn.ModuleList([
            nn.ModuleList([ConvBlock(width * 8, width * 4) for _ in range(num_blocks)]),
            nn.ModuleList([ConvBlock(width * 4, width * 2) for _ in range(num_blocks)]),
            nn.ModuleList([ConvBlock(width * 2, width) for _ in range(num_blocks)])
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.time_proj = nn.ModuleList([
            nn.Linear(32, width),
            nn.Linear(32, width * 2),
            nn.Linear(32, width * 4),
            nn.Linear(32, width * 4),
            nn.Linear(32, width * 2),
            nn.Linear(32, width)
        ])

        self.output = nn.Conv2d(width, in_channels, 1)

    def forward(self, x, t, x_cond):
        B = x.shape[0]
        t_emb = self.time_embed(t)

        h = torch.cat([x, x_cond], dim=1)
        h = self.input_conv(h)
        skips = []

        for i, blocks in enumerate(self.enc):
            for block in blocks:
                h = block(h)
            h += self.time_proj[i](t_emb)[:, :, None, None]
            skips.append(h)
            h = self.downsample(h)

        h = self.bottleneck(h)
        h += self.time_proj[3](t_emb)[:, :, None, None]

        for i, blocks in enumerate(self.dec):
            h = self.upsample(h)
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h)
            h += self.time_proj[4+i](t_emb)[:, :, None, None]

        return self.output(h)
