# mlp_models_v3.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoderV3(nn.Module):
    """
    Encodes a 1536-d image embedding into two representations:
      - h  (hidden): intermediate [B, hidden_dim] used for image-side
                     auxiliary tasks (protein identity, localization).
      - z  (shared): [B, shared_dim] projection into the shared latent
                     space where the CLIP loss is applied against the
                     ESM-branch projection.
    """

    def __init__(self, in_dim=1536, hidden_dim=2048, shared_dim=512, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.projector = nn.Linear(hidden_dim, shared_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


class ProteinEncoderV3(nn.Module):
    """
    Symmetric counterpart to ImageEncoderV3 on the ESM side.

    Takes a pooled 1280-d ESM embedding and maps it into the same shared
    latent space as the image branch.  Returning (h, z) keeps the API
    symmetric with ImageEncoderV3 even though the protein-side hidden
    representation is not consumed downstream by default.
    """

    def __init__(self, in_dim=1280, hidden_dim=2048, shared_dim=512, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.projector = nn.Linear(hidden_dim, shared_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


class ProteinPool(nn.Module):
    """
    Aggregates variable-length protein embeddings [B, N, 1280] -> [B, 1280].
    Identical to the v2 implementation; reproduced here so this module is
    self-contained.
    """

    def __init__(self, dim=1280):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))
        self.key   = nn.Linear(dim, dim, bias=False)
        self.val   = nn.Linear(dim, dim, bias=False)

    def forward(self, prot_embs, mask=None):
        B, N, D = prot_embs.shape
        q = self.query.view(1, 1, D).expand(B, 1, D)
        k = self.key(prot_embs)
        v = self.val(prot_embs)

        logits = torch.matmul(q, k.transpose(-1, -2)) / (D ** 0.5)

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn   = torch.softmax(logits, dim=-1)
        pooled = torch.matmul(attn, v).squeeze(1)
        return pooled


class ProteinIdentityHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class LocalizationHead(nn.Module):
    def __init__(self, in_dim, n_locs):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_locs)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def clip_loss(img, prot, temperature=0.07):
    """Symmetric CLIP / InfoNCE loss between two equally-sized batches."""
    img  = F.normalize(img,  dim=-1)
    prot = F.normalize(prot, dim=-1)

    logits  = (img @ prot.t()) / temperature
    targets = torch.arange(img.size(0), device=img.device)

    loss_i2p = F.cross_entropy(logits,   targets)
    loss_p2i = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i2p + loss_p2i)


def combined_loss(
    img_h, img_z, prot_z,
    gene_ids, loc_labels,
    identity_head, loc_head,
    temperature=0.07,
    lambda_id=1.0,
    lambda_loc=1.0,
):
    """
    Joint loss: contrastive alignment in the shared latent space + image-side
    protein-identity and localization heads.

    Args:
        img_h:         [B, hidden_dim]  image hidden representation
        img_z:         [B, shared_dim]  image projection in shared space
        prot_z:        [B, shared_dim]  ESM projection in shared space
        gene_ids:      [B]  long  (-1 = no label for this sample)
        loc_labels:    [B, n_locs]  float32 multi-hot
        identity_head: ProteinIdentityHead on img_h
        loc_head:      LocalizationHead    on img_h
    """
    loss_clip = clip_loss(img_z, prot_z, temperature)

    valid = gene_ids >= 0
    if valid.any():
        id_logits = identity_head(img_h[valid])
        loss_id   = F.cross_entropy(id_logits, gene_ids[valid])
    else:
        loss_id = torch.zeros(1, device=img_h.device).squeeze()

    loc_logits = loc_head(img_h)
    loss_loc   = F.binary_cross_entropy_with_logits(loc_logits, loc_labels)

    total = loss_clip + lambda_id * loss_id + lambda_loc * loss_loc
    return total, loss_clip, loss_id, loss_loc
