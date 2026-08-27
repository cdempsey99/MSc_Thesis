import math
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


class VariationalBottleneck(nn.Module):
    """
    Inserted between the encoder and the M decoder heads. Each head has its OWN
    independently-parameterised Gaussian z_m = mu_m + sigma_m * eps_m, giving the
    JSD/Pearson diversity loss a per-head structural parameter to push apart - not
    just a shared noise seed drawn M times from one identical distribution (the
    original design: every head's sample differed only by IID per-pixel noise,
    which a head's own conv layers tend to smooth out rather than turn into
    confidently-differing class boundaries).
    The prior mean for every head is the encoder's own (detached) output rather
    than N(0,1), so the KL term penalises drifting away from what the encoder
    actually produced for this input, not toward a generic prior that would fight
    the pretrained representation. Off by default (use_variational_bottleneck=False
    upstream) - see DecoderEnsemble, which is untouched by this class and keeps
    working exactly as before either way.
    """

    def __init__(self, channels=1024, num_heads=5, sigma_prior=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.sigma_prior = sigma_prior
        self.mu_convs = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1) for _ in range(num_heads)])
        self.logvar_convs = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1) for _ in range(num_heads)])

        # Zero/negative-bias init so training starts close to today's deterministic
        # behaviour (mu_m ~= e, small sigma) rather than a cold, garbled embedding
        for mu_conv, logvar_conv in zip(self.mu_convs, self.logvar_convs):
            nn.init.zeros_(mu_conv.weight)
            nn.init.zeros_(mu_conv.bias)
            nn.init.zeros_(logvar_conv.weight)
            nn.init.constant_(logvar_conv.bias, -4.0)

    def forward(self, e):
        prior_mean = e.detach()
        mus, logvars, kls = [], [], []
        for mu_conv, logvar_conv in zip(self.mu_convs, self.logvar_convs):
            mu = e + mu_conv(e)
            logvar = logvar_conv(e)
            kl = -0.5 * torch.mean(
                1 + logvar - math.log(self.sigma_prior ** 2)
                - (mu - prior_mean).pow(2) / (self.sigma_prior ** 2)
                - logvar.exp() / (self.sigma_prior ** 2)
            )
            mus.append(mu)
            logvars.append(logvar)
            kls.append(kl)
        return mus, logvars, torch.stack(kls).mean()

    def sample(self, mus, logvars):
        z_samples = []
        for mu, logvar in zip(mus, logvars):
            std = torch.exp(0.5 * logvar)
            z_samples.append(mu + std * torch.randn_like(std))
        return z_samples


class VBEvalWrapper(nn.Module):
    """Wraps a DecoderEnsemble + a trained per-head VariationalBottleneck so existing eval
    functions (evaluate_test_set, get_decoder_output_maps, and evaluate_error_localization's
    lambda call sites) can be used unchanged: each head receives its own mu_m(e) - the
    deterministic posterior mean it was actually trained around, no sampling noise at eval -
    instead of either raw encoder features or one shared mu across all heads."""

    def __init__(self, decoder, bottleneck):
        super().__init__()
        self.decoder = decoder
        self.bottleneck = bottleneck
        self.M = decoder.M

    def forward(self, x):
        mus, _, _ = self.bottleneck(x)
        head_outputs = [self.decoder.heads[m](mus[m]) for m in range(self.decoder.M)]
        return torch.stack(head_outputs)


class StudentHead(nn.Module):
    """
    Single decoder head for EnDD (Ensemble Distribution Distillation).
    Outputs Dirichlet concentration parameters α > 0 instead of raw logits.
    Architecture mirrors SegFormerDecoderHead; only the output activation differs.
    At inference: Dirichlet mean = α / Σα_k gives the class probability map.
    """

    def __init__(self, in_channels=1024, embed_dim=512, num_classes=NUM_CLASSES, p_drop=0.1):
        super().__init__()

        self.linear_fusion = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.activation1 = nn.GELU()

        self.spatial_refine1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.activation2 = nn.GELU()

        self.spatial_refine2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.activation3 = nn.GELU()

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

        x = F.interpolate(x, size=(56, 56), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Softplus ensures α > 0; small offset keeps lgamma numerically stable
        return F.softplus(x) + 1e-5
