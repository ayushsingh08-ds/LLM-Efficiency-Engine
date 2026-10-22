from __future__ import annotations

import torch
from torch import nn


class PromptEncoder(nn.Module):
    def __init__(self, input_dim: int = 5, embedding_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.net(x)
        return nn.functional.normalize(emb, p=2, dim=1)


class ModelEncoder(nn.Module):
    def __init__(self, input_dim: int = 6, embedding_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.net(x)
        return nn.functional.normalize(emb, p=2, dim=1)
