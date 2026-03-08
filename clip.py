import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttnMILPool(nn.Module):
    """Gated attention MIL pooling over instances in a bag."""
    def __init__(self, d_model: int, d_attn: int = 256, dropout: float = 0.1):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(d_model, d_attn), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(d_model, d_attn), nn.Sigmoid())
        self.w = nn.Linear(d_attn, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: torch.Tensor | None = None):
        """
        x:    [B, N, D]
        mask: [B, N] boolean, True for valid instances
        """
        v = self.V(x)
        u = self.U(x)
        h = self.dropout(v * u)
        logits = self.w(h).squeeze(-1)  # [B, N]

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        attn = torch.softmax(logits, dim=1)  # [B, N]
        pooled = torch.einsum("bn,bnd->bd", attn, x)  # [B, D]
        return pooled, attn


class ProteinBagToCLIP(nn.Module):
    """
    Takes a padded bag of per-protein ESM embeddings, returns a single 1280-d bag embedding.
    """
    def __init__(self, d_esm: int, d_clip: int = 1280, d_attn: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_esm),
            nn.Linear(d_esm, d_clip),
        )
        self.pool = GatedAttnMILPool(d_clip, d_attn=d_attn, dropout=dropout)

    def forward(self, prot_esm, mask=None):
        """
        prot_esm: [B, N, D_esm]
        mask:     [B, N]
        """
        x = self.proj(prot_esm)              # [B, N, 1280]
        pooled, attn = self.pool(x, mask)    # [B, 1280]
        return pooled, attn


def clip_loss(gene_emb, prot_emb, temperature: float = 0.07):
    """
    gene_emb: [B, 1280]
    prot_emb: [B, 1280]
    """
    gene_emb = F.normalize(gene_emb, dim=-1)
    prot_emb = F.normalize(prot_emb, dim=-1)

    logits = (gene_emb @ prot_emb.T) / temperature  # [B, B]
    labels = torch.arange(gene_emb.size(0), device=gene_emb.device)

    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def collate_gene_bags(samples):
    """
    samples: list of dicts:
      {
        "gene_emb":  Tensor[1536] or inputs to your gene encoder,
        "prot_list": list[Tensor[D_esm]]  (variable length)
      }
    """
    B = len(samples)

    # If you already have gene embeddings:
    gene_emb = torch.stack([s["gene_emb"] for s in samples], dim=0)  # [B,1536]

    lengths = torch.tensor([len(s["prot_list"]) for s in samples], dtype=torch.long)
    if lengths.max().item() == 0:
        raise ValueError("Batch contains only empty protein bags; drop/handle these earlier.")

    Nmax = int(lengths.max().item())
    D_esm = samples[0]["prot_list"][0].numel()

    prot_bag = torch.zeros(B, Nmax, D_esm, dtype=samples[0]["prot_list"][0].dtype)
    mask = torch.zeros(B, Nmax, dtype=torch.bool)

    for i, s in enumerate(samples):
        n = len(s["prot_list"])
        if n == 0:
            continue
        prot_bag[i, :n] = torch.stack(s["prot_list"], dim=0)
        mask[i, :n] = True

    return {"gene_emb": gene_emb, "prot_bag": prot_bag, "mask": mask, "lengths": lengths}

def train_step(batch, protein_bag_encoder, optimizer, temperature=0.07, device="cuda"):
    protein_bag_encoder.train()

    gene_emb = batch["gene_emb"].to(device)      # [B,1280]
    prot_bag = batch["prot_bag"].to(device)      # [B,N,D_esm]
    mask = batch["mask"].to(device)              # [B,N]

    prot_emb, attn = protein_bag_encoder(prot_bag, mask=mask)  # [B,1280]
    loss = clip_loss(gene_emb, prot_emb, temperature=temperature)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.item(), attn