import torch.nn as nn
import torch.nn.functional as F
from configs.config import *


# Note the choice of decoder embedding dim of 256 is arbitrary
class SegFormerDecoderHeadOld(nn.Module):

    def __init__(self, in_channels=1024, embed_dim=256, num_classes=NUM_CLASSES, p_drop=0.1):
        super().__init__()

        # 1024 -> 256 using a 2dconv as Pointwise MLP
        self.linear_fusion = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.activation1 = nn.GELU()

        # Spatial Context Layer (3x3 conv): 256 -> 256
        self.spatial_refine = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.activation2 = nn.GELU()

        # Dropout layer
        self.dropout = nn.Dropout2d(p=p_drop)

        # Classifier: 256 -> 24
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

        # Learnable upsampling
        # use groups=num_classes so each class channels upscales independently
        #self.learnable_upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=8, groups=num_classes)

    def forward(self, x):
        # x starts as [1, 1024, 28, 28]
        x = self.linear_fusion(x)
        x = self.activation1(x)
        x = self.spatial_refine(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.dropout(x)
        x = self.classifier(x)
        #x = self.learnable_upsample(x)

        # Upsampling, we stretch the 24 outputs back to the 224 x 224 of the original image
        # TODO : Change this to learnable weights?
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return x


class SegFormerDecoderHead(nn.Module):

    def __init__(self, in_channels=1024, embed_dim=512, num_classes=NUM_CLASSES, p_drop=0.1):
        super().__init__()

        # 1024 -> 512
        self.linear_fusion = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.activation1 = nn.GELU()

        # Spatial refinement block 1
        self.spatial_refine1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.activation2 = nn.GELU()

        # Spatial refinement block 2
        self.spatial_refine2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.activation3 = nn.GELU()

        # Bottleneck 512 -> 256
        self.bottleneck = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(embed_dim // 2)
        self.activation4 = nn.GELU()

        self.dropout = nn.Dropout2d(p=p_drop)
        self.classifier = nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.linear_fusion(x)
        x = self.bn1(x)
        x = self.activation1(x)

        x = self.spatial_refine1(x)
        x = self.bn2(x)
        x = self.activation2(x)

        x = self.spatial_refine2(x)
        x = self.bn3(x)
        x = self.activation3(x)

        x = self.bottleneck(x)
        x = self.bn4(x)
        x = self.activation4(x)

        x = self.dropout(x)
        x = self.classifier(x)

        # Progressive upsampling
        x = F.interpolate(x, size=(56, 56), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        return x


class DecoderEnsemble(nn.Module):

    def __init__(self, M=5, in_channels=1024, embed_dim=256, num_classes=NUM_CLASSES):
        super().__init__()

        self.M = M

        # Create a range of dropout values for the various heads
        self.dropout_rates = torch.linspace(0.05, 0.35, M).tolist()

        # Create M individual SegFormerDecoderHead instances
        self.heads = nn.ModuleList([
            SegFormerDecoderHead(in_channels, embed_dim, num_classes, p_drop=self.dropout_rates[i]) for i in range(M)
        ])

    def forward(self, x):
        # Run encoder features through each of the M heads
        # Output is a list of tensors [[1, 24, 224, 224], [1, 24, 224, 224], ...]
        head_outputs = [head(x) for head in self.heads]

        # Stack the outputs along a new 'member' dimension
        # so [Member, Batch, Class, H, W] = [5, ?, 24, 224, 224]
        return torch.stack(head_outputs)
