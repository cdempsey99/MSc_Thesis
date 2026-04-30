import torch.nn as nn
import torch.nn.functional as F
from configs.config import *


# Note the choice of decoder embedding dim of 256 is arbitrary
class SegFormerDecoderHead(nn.Module):

    def __init__(self, in_channels=1024, embed_dim=256, num_classes=NUM_CLASSES):
        super().__init__()

        # 1024 -> 256 using a 2dconv as Pointwise MLP
        self.linear_fusion = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.activation1 = nn.GELU()

        # Spatial Context Layer (3x3 conv): 256 -> 256
        self.spatial_refine = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.activation2 = nn.BatchNorm2d(embed_dim)

        # Classifier: 256 -> 24
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # x starts as [1, 1024, 28, 28]
        x = self.linear_fusion(x)
        x = self.activation1(x)
        x = self.spatial_refine(x)
        x = self.activation2(x)
        x = self.classifier(x)

        # Upsampling, we stretch the 24 outputs back to the 224 x 224 of the original image
        # TODO : Change this to learnable weights?
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        return x

class DecoderEnsemble(nn.Module):

    def __init__(self, M=5, in_channels=1024, embed_dim=256, num_classes=NUM_CLASSES):
        super().__init__()

        self.M = M

        # Create M individual SegFormerDecoderHead instances
        self.heads = nn.ModuleList([
            SegFormerDecoderHead(in_channels, embed_dim, num_classes) for _ in range(M)
        ])

    def forward(self, x):
        # Run encoder features through each of the M heads
        # Output is a list of tensors [[1, 24, 224, 224], [1, 24, 224, 224], ...]
        head_outputs = [head(x) for head in self.heads]

        # Stack the outputs along a new 'member' dimension
        # so [Member, Batch, Class, H, W] = [5, ?, 24, 224, 224]
        return torch.stack(head_outputs)
