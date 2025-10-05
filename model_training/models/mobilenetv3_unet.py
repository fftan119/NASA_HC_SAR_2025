import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import mobilenet_v3_small

# Lightweight decoder blocks
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip):
        x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class MNetV3SmallUNet(nn.Module):
    """MobileNetV3-small backbone with a minimal U-Net style decoder.
    Input: 1×H×W grayscale. Output: 1×H×W logits.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = mobilenet_v3_small(weights=None if not pretrained else None)  # offline-safe
        # Adapt first conv for 1-channel input
        first_conv = self.backbone.features[0][0]
        if first_conv.in_channels != 1:
            new_first = nn.Conv2d(1, first_conv.out_channels, kernel_size=first_conv.kernel_size,
                                  stride=first_conv.stride, padding=first_conv.padding, bias=False)
            with torch.no_grad():
                new_first.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
            self.backbone.features[0][0] = new_first

        # Collect feature map indices (MobileNetV3-small has 13 feature blocks)
        # We'll take shallow, mid, deep features for skips.
        self.stage_ids = [2, 6, 12]  # empirically chosen layers to tap
        self.out_channels = []
        dummy = torch.zeros(1,1,256,256)
        feats = []
        x = dummy
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in self.stage_ids:
                feats.append(x)
                self.out_channels.append(x.shape[1])
        enc_out_ch = x.shape[1]
        c1, c2, c3 = self.out_channels  # shallow->deep

        # projection on the deepest encoder output
        self.enc_proj = ConvBNReLU(enc_out_ch, 160)

        # decoder
        self.up3 = UpBlock(160, c3, 128)
        self.up2 = UpBlock(128, c2, 96)
        self.up1 = UpBlock(96,  c1, 64)
        self.head = nn.Sequential(
            ConvBNReLU(64, 32),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        skips = []
        k = 0
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in self.stage_ids:
                skips.append(x)
        x = self.enc_proj(x)
        x = self.up3(x, skips[-1])
        x = self.up2(x, skips[-2])
        x = self.up1(x, skips[-3])
        logits = self.head(x)
        return logits