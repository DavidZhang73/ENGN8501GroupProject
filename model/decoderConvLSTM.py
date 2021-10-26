import torch
import torch.nn as nn
import torch.nn.functional as F

from model.convLSTM import ConvLSTM


def interpolate(input, size):
    return F.interpolate(input, size=size, mode='bilinear', align_corners=False)


class Decoder(nn.Module):
    """
    Decoder upsamples the image by combining the feature maps at all resolutions from the encoder.
    
    Input:
        x4: (B, C, H/16, W/16) feature map at 1/16 resolution.
        x3: (B, C, H/8, W/8) feature map at 1/8 resolution.
        x2: (B, C, H/4, W/4) feature map at 1/4 resolution.
        x1: (B, C, H/2, W/2) feature map at 1/2 resolution.
        x0: (B, C, H, W) feature map at full resolution.
        
    Output:
        x: (B, C, H, W) upsampled output at full resolution.
    """

    def __init__(self, channels, feature_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_channels[0] + channels[0], channels[1], 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.convLSTM1 = ConvLSTM(input_dim=channels[1], hidden_dim=channels[1], batch_first=True)
        self.conv2 = nn.Conv2d(feature_channels[1] + channels[1], channels[2], 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.convLSTM2 = ConvLSTM(input_dim=channels[2], hidden_dim=channels[2], batch_first=True)
        self.conv3 = nn.Conv2d(feature_channels[2] + channels[2], channels[3], 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.convLSTM3 = ConvLSTM(input_dim=channels[3], hidden_dim=channels[3], batch_first=True)
        self.conv4 = nn.Conv2d(feature_channels[3] + channels[3], channels[4], 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x4, x3, x2, x1, x0, b, t):
        x = interpolate(x4, x3.shape[2:])
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.unflatten(0, (b, t))
        (x,), _ = self.convLSTM1(x, None)
        x = x.flatten(0, 1)

        x = interpolate(x, x2.shape[2:])
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.unflatten(0, (b, t))
        (x,), _ = self.convLSTM2(x, None)
        x = x.flatten(0, 1)

        x = interpolate(x, x1.shape[2:])
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = x.unflatten(0, (b, t))
        (x,), _ = self.convLSTM3(x, None)
        x = x.flatten(0, 1)

        x = interpolate(x, x0.shape[2:])
        x = torch.cat([x, x0], dim=1)
        x = self.conv4(x)
        x = x.unflatten(0, (b, t))
        return x
