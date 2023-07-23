import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool = False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            #original: normalization by 32 groups
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(out_channels, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(out_channels, out_channels),
            nn.ReLU(),
        )

        self.is_res = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(ConvBlock(in_channels, out_channels), nn.MaxPool2d(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ConvBlock(out_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), 1)
        x = self.layers(x)

        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        #original: in channels different
        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        #original: activation and embedding different
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class UnetModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_size: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.hidden_size = hidden_size

        self.init_conv = ConvBlock(in_channels, hidden_size, residual=True)

        self.down1 = DownBlock(hidden_size, hidden_size)
        self.down2 = DownBlock(hidden_size, 2 * hidden_size)
        self.down3 = DownBlock(2 * hidden_size, 2 * hidden_size)

        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())

        self.timestep_embedding = TimestepEmbedding(2 * hidden_size)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * hidden_size, 2 * hidden_size, 4, 4),
            nn.GroupNorm(8, 2 * hidden_size),
            nn.ReLU(),
        )

        self.up1 = UpBlock(4 * hidden_size, 2 * hidden_size)
        self.up2 = UpBlock(4 * hidden_size, hidden_size)
        self.up3 = UpBlock(2 * hidden_size, hidden_size)
        self.out = nn.Conv2d(2 * hidden_size, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x) #B x C x H x W

        down1 = self.down1(x) #B x C x H/2 x W/2
        down2 = self.down2(down1) #B x 2C x H/4 x W/4
        down3 = self.down3(down2) #B x 2C x H/8 x W/8

        thro = self.to_vec(down3) #B x 2C x H/32 x B/32
        temb = self.timestep_embedding(t) #B x 2C

        thro = self.up0(thro + temb[:, :, None, None]) #B x 2C x H/8 x B/8

        up1 = self.up1(thro, down3) + temb[:, :, None, None]
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)

        out = self.out(torch.cat((up3, x), 1))

        return out
