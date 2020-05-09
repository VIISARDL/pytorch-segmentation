import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ['SSegNet', 'ssegnet']

def ssegnet(pretrained=False, **kwargs):
    model = SSegNet(**kwargs)

    return model

class SSegNet(nn.Module):
    """Simpler Segmentation"""
    def __init__(self, in_channels=3, num_classes=3):
        super(SSegNet, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_channels, 18, 3, padding=1),
            nn.BatchNorm2d(18),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(18, 9, 3, padding=1),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(9, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(3, num_classes, 3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.LeakyReLU(inplace=True)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal(m.weight)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        return self.f(x)