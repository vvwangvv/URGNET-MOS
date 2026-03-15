import torch
import torch.nn as nn
from typing import Optional, Tuple
from urgent_mos.utils import lengths2padding_mask


class MeanPooling(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_frame_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape
        key_padding_mask = lengths2padding_mask(lengths, max_len=T)
        valid = (~key_padding_mask).float()  # (B,T)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1)
        pooled = (x * valid.unsqueeze(-1)).sum(dim=1) / denom  # (B,D)

        attn = valid / denom
        return pooled, (attn, x) if return_frame_scores else None


class AttentivePooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_frame_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape
        assert lengths.max().item() == T, f"lengths.max()={lengths.max().item()} != T={T}"
        scores = self.score(x).squeeze(-1)
        scores.masked_fill_(lengths2padding_mask(lengths), float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # (B,T)
        pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # (B,D)

        return pooled, (attn, x) if return_frame_scores else None


class MetricTokenPooling(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim

        self.metric_token = nn.Parameter(torch.randn(input_dim) * 0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,  # (B,L,D)
        )

        self.norm_q = nn.LayerNorm(input_dim)
        self.norm_x = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_frame_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape
        assert lengths.max().item() == T, f"lengths.max()={lengths.max().item()} != T={T}"
        q = self.metric_token.view(1, 1, self.input_dim).expand(B, 1, self.input_dim)  # (B,1,D)

        q = self.norm_q(q)
        x = self.norm_x(x)

        # out: (B,1,D)
        # attn_w: (B,1,T) when average_attn_weights=True (default)
        out, attn_w = self.attn(
            query=q,
            key=x,
            value=x,
            key_padding_mask=lengths2padding_mask(lengths),
            need_weights=return_frame_scores,
            average_attn_weights=True,
        )

        pooled = out.squeeze(1)  # (B,D)
        attn = attn_w.squeeze(1) if return_frame_scores and attn_w is not None else None  # (B,T)
        return pooled, (attn, x) if return_frame_scores else None
