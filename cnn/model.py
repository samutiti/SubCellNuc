"""Small CNN encoder-decoder + classifier head.

128x128 input -> 768-dim embedding -> two heads:
  (a) decoder      : 768 -> 128x128 reconstruction (MSE loss)
  (b) classifier   : 768 -> num_classes protein logits (CE loss)

Same architecture works for in_channels=2 (protein+nucleus) or 3 (protein+MT+nucleus);
just pass `in_channels` at construction time.
"""

from typing import Dict

import torch
import torch.nn as nn


def conv_block(in_c: int, out_c: int, stride: int = 2) -> nn.Sequential:
    """Conv -> BN -> GELU. Halves spatial dims when stride=2."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.GELU(),
    )


def deconv_block(in_c: int, out_c: int, stride: int = 2) -> nn.Sequential:
    """Transposed conv -> BN -> GELU. Doubles spatial dims when stride=2."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.GELU(),
    )


class Encoder(nn.Module):
    """128x128 -> 768-dim embedding via 5 stride-2 conv blocks + global avg pool."""

    def __init__(self, in_channels: int, embed_dim: int = 768):
        super().__init__()
        self.stem = conv_block(in_channels, 64, stride=2)   # 128 -> 64
        self.b1 = conv_block(64, 128, stride=2)              # 64 -> 32
        self.b2 = conv_block(128, 256, stride=2)             # 32 -> 16
        self.b3 = conv_block(256, 512, stride=2)             # 16 -> 8
        self.b4 = conv_block(512, embed_dim, stride=2)       # 8 -> 4
        self.pool = nn.AdaptiveAvgPool2d(1)                  # -> (B, 768, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.pool(x).flatten(1)                          # (B, 768)
        return x


class Decoder(nn.Module):
    """768-dim embedding -> 128x128 reconstruction. Mirrors the encoder."""

    def __init__(self, out_channels: int, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim * 4 * 4)
        self.b0 = deconv_block(embed_dim, 512, stride=2)     # 4 -> 8
        self.b1 = deconv_block(512, 256, stride=2)           # 8 -> 16
        self.b2 = deconv_block(256, 128, stride=2)           # 16 -> 32
        self.b3 = deconv_block(128, 64, stride=2)            # 32 -> 64
        self.b4 = deconv_block(64, 32, stride=2)             # 64 -> 128
        self.head = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.act = nn.Sigmoid()                              # images are normalized to [0, 1]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z).view(z.size(0), -1, 4, 4)
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        return self.act(self.head(x))


class ProteinCNN(nn.Module):
    """Encoder + reconstruction decoder + protein classifier head.

    Forward returns:
        embedding: (B, 768)
        recon:     (B, in_channels, 128, 128)
        logits:    (B, num_classes)
    """

    def __init__(self, in_channels: int, num_classes: int, embed_dim: int = 768):
        super().__init__()
        self.encoder = Encoder(in_channels, embed_dim)
        self.decoder = Decoder(in_channels, embed_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        logits = self.classifier(z)
        return {"embedding": z, "recon": recon, "logits": logits}

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: just return the 768-dim embedding."""
        return self.encoder(x)
