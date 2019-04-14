import torch
import torch.nn as nn

__all__ = ['SSeg', 'sseg']

def ssegnet(pretrained=False, **kwargs):
    model = SSeg(**kwargs)

    return model

class SSegNet(nn.Module):
	"""Simpler Segmentation"""
    def __init__(self, in_dim=11, out_dim=3):
        super(CSeg, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_dim, 18, 3, padding=1),
            nn.BatchNorm2d(18),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(18, 9, 3, padding=1),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(9, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(3, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.f(x)