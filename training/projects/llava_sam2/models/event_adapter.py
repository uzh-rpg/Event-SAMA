"""Event-language adapter: the small MLP connector that bridges RGB-aligned
features to the event modality.

This mirrors the "event-language adapter" introduced in EventGPT
(arXiv:2412.00832, Section 4 / Section 5.1). It sits AFTER SAMA's existing
``mlp1`` projector (which is RGB-aligned and frozen) and BEFORE the LLM token
stream. We add it instead of retraining ``mlp1`` so the RGB→language
alignment knowledge baked into the pretrained Sa2VA/SAMA weights is preserved.

Design:
    Input  : (N, ..., D)   where D = LLM hidden size (post mlp1 / post STC).
    Output : (N, ..., D)
    Body   : LayerNorm -> Linear(D, D) -> GELU -> Linear(D, D)
    Output is added residually to the input. The second Linear is zero-initialised
    so the adapter starts as the identity function -- training it does no harm
    to the pretrained model and recovers exactly the RGB pipeline at step 0.

The same adapter instance is intended to be applied to BOTH:
    * ``F_key`` after ``mlp1`` (the keyframe path),
    * ``F_agg`` after the STC context aggregator's projection,
so that we have a single bottleneck where event<->language calibration is learned.
"""

from __future__ import annotations

import torch
from torch import nn


class EventAdapter(nn.Module):
    """LayerNorm + 2-layer residual MLP. Identity at init (zero-init last linear)."""

    def __init__(self, hidden_size: int, expansion: float = 1.0,
                 dropout: float = 0.0):
        super().__init__()
        inner = max(1, int(round(hidden_size * expansion)))
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, inner)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner, hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return residual + h
