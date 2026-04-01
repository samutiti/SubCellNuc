# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageProjector(nn.Module):
    """
    Compress 1536-d image embedding to 1280-d.
    """
    def __init__(self, in_dim=1536, out_dim=1280, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ProteinPool(nn.Module):
    """
    Aggregates variable-length protein embeddings [B, N, 1280] into [B, 1280].

    Uses attention pooling with a learnable query.
    """
    def __init__(self, dim=1280):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))
        self.key = nn.Linear(dim, dim, bias=False)
        self.val = nn.Linear(dim, dim, bias=False)

    def forward(self, prot_embs, mask=None):
        """
        prot_embs: FloatTensor [B, N, D]
        mask: BoolTensor [B, N] where True means "valid"
        """
        B, N, D = prot_embs.shape
        q = self.query.view(1, 1, D).expand(B, 1, D)      # [B,1,D]
        k = self.key(prot_embs)                           # [B,N,D]
        v = self.val(prot_embs)                           # [B,N,D]

        # attention logits: [B,1,N]
        logits = torch.matmul(q, k.transpose(-1, -2)) / (D ** 0.5)

        if mask is not None:
            # mask: True=valid; convert to -inf for invalid positions
            logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(logits, dim=-1)              # [B,1,N]
        pooled = torch.matmul(attn, v).squeeze(1)         # [B,D]
        return pooled


def clip_loss(img, prot, temperature=0.07):
    """
    Symmetric CLIP/InfoNCE loss.
    img:  [B, D]
    prot: [B, D]
    """
    img = F.normalize(img, dim=-1)
    prot = F.normalize(prot, dim=-1)

    logits = (img @ prot.t()) / temperature  # [B,B]
    targets = torch.arange(img.size(0), device=img.device) # [0 to B-1]

    loss_i2p = F.cross_entropy(logits, targets)
    loss_p2i = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i2p + loss_p2i)