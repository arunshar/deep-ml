import torch, torch.nn as nn

def build_model() -> nn.Module:
    """
    Return a tiny nn.Module for MNIST classification (10 classes).
    IMPORTANT: If total trainable params > 2048, final accuracy will be set to 0.
    Tip: Consider very small convs, global average pooling, and tiny linear head.
    """
    class TinyNet(nn.Module):
         def __init__(self):
             super().__init__()
             self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
             self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
             self.pool = nn.MaxPool2d(kernel_size=2)        #28x28 -> 14x14 -> 7x7
             self.act = nn.ReLU()
             self.gap = nn.AdaptiveAvgPool2d((1, 1))        # -> (18, 16, 1, 1)
             self.fc = nn.Linear(16, 10)                    # -> (8, 10) 
         def forward(self, x):
             # x: (8, 1, 28, 28)
             x = self.act(self.conv1(x))                    # (8, 8, 28, 28)
             x = self.pool(x)                               # (8, 8, 14, 14)
             x = self.act(self.conv2(x))                    # (8, 16, 14, 14)
             x = self.pool(x)                               # (8, 16, 7, 7)
             x = self.gap(x)                                # (8, 16, 1, 1)
             x = torch.flatten(x, 1)                        # (8, 16)
             x = self.fc(x)                                 # (8, 10)
             return x
    return TinyNet()
