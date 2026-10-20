import torch
from torch import nn

class Conv3k(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1)
        
    def forward(self, x):
        return self.conv1(x)
    
class DoubleConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv3k(channels_in, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            Conv3k(channels_out, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.ReLU()
        )
        
    def forward(self,x):
        return self.double_conv(x)
    
class DownConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2,2),
            DoubleConv(channels_in, channels_out)
        )
        
    def forward(self, x):
        return self.encoder(x)
    
class UpConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic'),
            nn.Conv2d(channels_in, channels_in//2, kernel_size=1, stride=1)
        )
        
        self.decoder = DoubleConv(channels_in, channels_out)
        
    def forward(self, x1, x2):
        """
        x1 volume from upsample
        x2 volume from downsample to concatenate
        """
        
        x1 = self.upsample_layer(x1)
        x = torch.cat([x2, x1], dim=1)
        
        return self.decoder(x)
    
class UNet(nn.Module):
    def __init__(self, channels_in, channels, num_classes):
        super().__init__()
        
        self.first_conv = DoubleConv(channels_in, channels)
        self.down_conv1 = DownConv(channels, channels*2)
        self.down_conv2 = DownConv(channels*2, channels*4)        
        self.down_conv3 = DownConv(channels*4, channels*8)
        self.down_conv4 = DownConv(channels*8, channels*16)
        
        self.middle_conv = DownConv(channels*16, channels*32)
        
        self.up_conv1 = UpConv(channels*32, channels*16)
        self.up_conv2 = UpConv(channels*16, channels*8)
        self.up_conv3 = UpConv(channels*8, channels*4)
        self.up_conv4 = UpConv(channels*4, channels*2)
        self.up_conv5 = UpConv(channels*2, channels)
        
        self.last_conv = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)
        
    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)
        x5 = self.down_conv4(x4)
        
        x6 = self.middle_conv(x5)
        
        u1 = self.up_conv1(x6, x5)
        u2 = self.up_conv2(u1, x4)
        u3 = self.up_conv3(u2, x3)
        u4 = self.up_conv4(u3, x2)
        u5 = self.up_conv5(u4, x1)
        
        return self.last_conv(u5)
                