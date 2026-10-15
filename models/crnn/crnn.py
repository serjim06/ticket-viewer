from torch import nn
import torch.nn.functional as F
import torch

conv_layer = lambda in_channel, out_channel : nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)

class CRNN(nn.Module):
    def __init__(self, in_channels, output_classes, width = 256):
        """Initialize the OCR Model

        Args:
            in_channels(int): Number of channels in the input image (1 for grayscale, 3 for RGB)
            output_classes(int): Number of output classes (number of characters + 1 for the blank token)
            width(int): Width of the input image (default: 256)
        """
        super().__init__()
        
        self.output_classes = output_classes
        
        self.cnn = nn.Sequential(
            conv_layer(in_channels, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            conv_layer(16, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.MaxPool2d(2,2),
            
            # W/2, 16
            
            conv_layer(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            conv_layer(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            
            nn.MaxPool2d(2,2),
            
            # (W/2)/2, 8
            
            conv_layer(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),   

            conv_layer(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),   
            
            nn.MaxPool2d(2,2),   
            
            # ((W/2)/2)/2, 4                          
        )
        
        dimension_entrada_lstm = 512 * 4 
        
        self.lstm = nn.LSTM(
            input_size=dimension_entrada_lstm,
            hidden_size= 512,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Linear(512*2, output_classes)
        
        
    def forward(self, x):
        features = self.cnn(x)
        features = features.permute(0,3,1,2)
        batch, width, channels, height = features.size()
        features = features.view(batch, width, channels*height)
        
        out, _ = self.lstm(features)
        logits = self.fc(out)
        
        log_probs = F.log_softmax(logits, dim=2).permute(1,0,2)
        
        return log_probs
        