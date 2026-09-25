"""
model/temporal_attention.py
----------------------------
Temporal attention mechanism for the TGAT model.

Paper reference: Section III-B (Eq. 5)
    H_{t}^{(l+1)} = σ( Σ_{τ=1}^{T} α_{t,τ} · D^{-½} A_τ D^{-½} · H_τ^{(l)} · W^{(l)} )

    where α_{t,τ} are the temporal attention coefficients that capture
    the relevance of information from different time steps τ when computing
    the representation at time step t.

Upgrade over the paper's TGCN:
    The paper's TGCN uses a simple scalar temporal attention coefficient α_{t,τ}
    learned as a parameter per (t, τ) pair. This doesn't generalise to unseen
    time steps.

    TGAT (Xu et al., 2020 — "Inductive Representation Learning on Temporal
    Graphs") instead computes α_{t,τ} via a MULTI-HEAD ATTENTION mechanism
    over learnable TIME ENCODINGS of the time delta Δt = t - τ:

        Φ(Δt) = [cos(ω₁·Δt), sin(ω₁·Δt), ..., cos(ω_d·Δt), sin(ω_d·Δt)]

    where ω₁...ω_d are learnable frequency parameters. This gives the model
    an expressive, continuous-time representation of "how long ago" each
    historical state occurred, enabling it to generalise to shock periods
    of any length without needing to fix a maximum sequence length.

    The attention weights α_{t,τ} are then computed as scaled dot-product
    attention between the query (current time step's encoding) and the
    key (each past time step's encoding), conditioned on the node embeddings.

Architecture:
    TemporalEncoding    — learnable cosine/sine time encoding Φ(Δt) ∈ ℝ^{time_dim}
    TemporalAttention   — multi-head attention over time-encoded node features
                          Input:  query  (N, hidden_dim)  — current step
                                  keys   (T, N, hidden_dim) — all past steps
                                  values (T, N, hidden_dim) — all past steps
                                  time_deltas (T,)          — Δt for each step
                          Output: (N, hidden_dim) — temporally-attended node embeddings
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings as CFG


# ─────────────────────────────────────────────────────────────────────────────
# TIME ENCODING  (learnable cosine/sine basis)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEncoding(nn.Module):
    """
    Learnable time encoding using cosine/sine basis functions.

    Maps a scalar time delta Δt ∈ ℝ to a vector Φ(Δt) ∈ ℝ^{time_dim}:

        Φ(Δt) = [ cos(ω₁·Δt), sin(ω₁·Δt), ...,
                   cos(ω_{d/2}·Δt), sin(ω_{d/2}·Δt) ]

    The frequencies ω₁...ω_{d/2} are learnable parameters, initialised
    following the geometric progression from the transformer positional
    encoding literature (Vaswani et al., 2017), which gives the model
    both short-range and long-range time sensitivity from the start of
    training.

    Args:
        time_dim : Dimension of the output time encoding vector.
                   Must be even (cos/sin pairs). Default: CFG.TGAT_TIME_DIM.
    """

    def __init__(self, time_dim: int = CFG.TGAT_TIME_DIM):
        super().__init__()
        if time_dim % 2 != 0:
            raise ValueError(
                f"time_dim must be even (cos/sin pairs), got {time_dim}."
            )
        self.time_dim = time_dim
        n_freqs       = time_dim // 2

        # Initialise frequencies on a geometric scale: ω_i = 1 / 10000^{2i/d}
        # This ensures both fine-grained (large ω) and coarse-grained (small ω)
        # temporal patterns are captured from the first forward pass.
        init_freqs = torch.tensor(
            [1.0 / (10000.0 ** (2 * i / time_dim)) for i in range(n_freqs)],
            dtype=torch.float32,
        )
        # Make frequencies learnable so the model can adapt them to the
        # characteristic time scales of financial shock propagation.
        self.freqs = nn.Parameter(init_freqs)                # (n_freqs,)
        self.bias  = nn.Parameter(torch.zeros(time_dim))     # optional learned bias

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of time deltas.

        Args:
            delta_t : FloatTensor of any shape (...,) — time deltas in [0, 1]
                      (normalised: 0 = start of shock, 1 = end of shock).

        Returns:
            FloatTensor of shape (..., time_dim) — time encoding vectors.
        """
        # Expand for broadcasting: (..., 1) × (n_freqs,) → (..., n_freqs)
        dt     = delta_t.unsqueeze(-1)                       # (..., 1)
        angles = dt * self.freqs                             # (..., n_freqs)

        # Interleave cos and sin:  [cos(ω₁t), sin(ω₁t), cos(ω₂t), ...]
        cos_enc = torch.cos(angles)                          # (..., n_freqs)
        sin_enc = torch.sin(angles)                          # (..., n_freqs)

        # Stack along last dim and flatten: (..., n_freqs, 2) → (..., time_dim)
        encoding = torch.stack([cos_enc, sin_enc], dim=-1)   # (..., n_freqs, 2)
        encoding = encoding.flatten(-2)                      # (..., time_dim)
        return encoding + self.bias


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL ATTENTION  (multi-head, over sequence of graph snapshots)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Multi-head temporal attention that aggregates information from a sequence
    of T historical node-embedding snapshots, weighted by their temporal
    relevance to the current time step.

    This module implements the α_{t,τ} computation from paper Eq. 5:

        α_{t,τ} = softmax_τ( (Q_t · K_τᵀ) / √d_k )

    where:
        Q_t  = W_Q · [h_t ‖ Φ(0)]           query: current embeddings + t=0 encoding
        K_τ  = W_K · [h_τ ‖ Φ(t - τ)]       key:   past embeddings + time encoding
        V_τ  = W_V · h_τ                     value: past embeddings

    The time encoding Φ(Δt) is concatenated to the node embedding before
    the query/key projections, giving each attention head access to both
    the content (what state was the market in?) and the timing (how long
    ago was it?) simultaneously.

    Args:
        hidden_dim  : Node embedding dimension. Paper: 64.
        time_dim    : Time encoding dimension. Paper: 16.
        num_heads   : Number of attention heads. Paper: 4.
        dropout     : Attention dropout. Paper: 0.3.
    """

    def __init__(
        self,
        hidden_dim  : int   = CFG.TGAT_HIDDEN_DIM,
        time_dim    : int   = CFG.TGAT_TIME_DIM,
        num_heads   : int   = CFG.TGAT_NUM_HEADS,
        dropout     : float = CFG.TGAT_DROPOUT,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.hidden_dim  = hidden_dim
        self.time_dim    = time_dim
        self.num_heads   = num_heads
        self.head_dim    = hidden_dim // num_heads
        self.dropout_p   = dropout

        # Input to Q/K projections: node embedding + time encoding
        qk_input_dim = hidden_dim + time_dim

        # Linear projections for Q, K, V — one set, all heads concatenated
        self.W_Q = nn.Linear(qk_input_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(qk_input_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim,   hidden_dim, bias=False)

        # Output projection: merge heads back to hidden_dim
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.time_enc = TemporalEncoding(time_dim)
        self.attn_drop = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all linear layers."""
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        h_seq       : torch.Tensor,    # (T, N, hidden_dim) — all time steps
        time_deltas : torch.Tensor,    # (T,)               — Δt for each step
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute temporally-attended node embeddings.

        Args:
            h_seq        : Sequence of node embeddings across T time steps.
                           Shape (T, N, hidden_dim).
            time_deltas  : Time delta for each step (normalised to [0, 1]).
                           Shape (T,). The current step is assumed to be the
                           LAST element (time_deltas[-1]).
            return_weights: If True, also return attention weight matrix.

        Returns:
            attended : FloatTensor (N, hidden_dim) — updated node embeddings.
            weights  : FloatTensor (N, num_heads, T) or None — attention weights.
        """
        T, N, D = h_seq.shape
        assert D == self.hidden_dim, f"Expected hidden_dim={self.hidden_dim}, got {D}"

        # ── Time encodings for every step ──────────────────────────────────
        # Φ(Δt) for each of the T steps:  (T, time_dim)
        time_enc_all = self.time_enc(time_deltas)          # (T, time_dim)

        # Broadcast to (T, N, time_dim) for concatenation with node embeddings
        time_enc_all = time_enc_all.unsqueeze(1).expand(T, N, self.time_dim)

        # ── Query: current step (last in sequence) ──────────────────────────
        h_current    = h_seq[-1]                           # (N, hidden_dim)
        t_current    = time_enc_all[-1]                    # (N, time_dim)
        q_input      = torch.cat([h_current, t_current], dim=-1)   # (N, hidden_dim + time_dim)

        # ── Keys and Values: all steps ──────────────────────────────────────
        kv_input     = torch.cat([h_seq, time_enc_all], dim=-1)     # (T, N, hidden_dim + time_dim)

        # Linear projections
        Q  = self.W_Q(q_input)                             # (N, hidden_dim)
        K  = self.W_K(kv_input)                            # (T, N, hidden_dim)
        V  = self.W_V(h_seq)                               # (T, N, hidden_dim)

        # ── Reshape into multi-head format ─────────────────────────────────
        # Q: (N, num_heads, head_dim)
        Q = Q.view(N, self.num_heads, self.head_dim)

        # K, V: (T, N, num_heads, head_dim)
        K = K.view(T, N, self.num_heads, self.head_dim)
        V = V.view(T, N, self.num_heads, self.head_dim)

        # ── Scaled dot-product attention ────────────────────────────────────
        # Q: (N, num_heads, head_dim)
        # K: (T, N, num_heads, head_dim) → (N, num_heads, head_dim, T)
        K_t  = K.permute(1, 2, 3, 0)                      # (N, num_heads, head_dim, T)
        Q_e  = Q.unsqueeze(-1)                             # (N, num_heads, head_dim, 1)

        # Attention scores: (N, num_heads, 1, T) → (N, num_heads, T)
        scale   = math.sqrt(self.head_dim)
        scores  = torch.matmul(Q_e.transpose(-2, -1), K_t)  # (N, num_heads, 1, T)
        scores  = scores.squeeze(-2) / scale               # (N, num_heads, T)

        # Softmax over T (time dimension)
        weights = F.softmax(scores, dim=-1)                # (N, num_heads, T)
        weights = self.attn_drop(weights)

        # ── Weighted sum of values ──────────────────────────────────────────
        # V: (T, N, num_heads, head_dim) → (N, num_heads, T, head_dim)
        V_t      = V.permute(1, 2, 0, 3)                  # (N, num_heads, T, head_dim)

        # weights: (N, num_heads, T) → (N, num_heads, 1, T)
        w_e      = weights.unsqueeze(-2)                   # (N, num_heads, 1, T)
        attended = torch.matmul(w_e, V_t)                  # (N, num_heads, 1, head_dim)
        attended = attended.squeeze(-2)                    # (N, num_heads, head_dim)

        # ── Merge heads ─────────────────────────────────────────────────────
        attended = attended.contiguous().view(N, self.hidden_dim)  # (N, hidden_dim)
        attended = self.W_O(attended)                      # (N, hidden_dim)

        if return_weights:
            return attended, weights
        return attended, None


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    torch.manual_seed(42)

    T, N   = 10, 50          # 10 time steps, 50 Nifty tickers
    D, Td  = CFG.TGAT_HIDDEN_DIM, CFG.TGAT_TIME_DIM
    heads  = CFG.TGAT_NUM_HEADS

    # Time encoding test
    enc    = TemporalEncoding(Td)
    deltas = torch.linspace(0, 1, T)
    phi    = enc(deltas)
    assert phi.shape == (T, Td), f"Expected ({T},{Td}), got {phi.shape}"
    print(f"TemporalEncoding OK: {phi.shape}  (range [{phi.min():.3f}, {phi.max():.3f}])")

    # Temporal attention test
    attn   = TemporalAttention(
        hidden_dim=D, time_dim=Td, num_heads=heads, dropout=0.0
    )
    h_seq       = torch.randn(T, N, D)
    time_deltas = torch.linspace(0, 1, T)

    out, weights = attn(h_seq, time_deltas, return_weights=True)
    assert out.shape     == (N, D),          f"out shape wrong: {out.shape}"
    assert weights.shape == (N, heads, T),   f"weights shape wrong: {weights.shape}"

    # Weights must sum to 1 along time axis
    w_sum = weights.sum(dim=-1)
    assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-5), \
        f"Attention weights don't sum to 1: {w_sum.mean():.6f}"

    print(f"TemporalAttention OK: h_seq {h_seq.shape} → out {out.shape}")
    print(f"Attention weights  : {weights.shape}  (sum={w_sum.mean():.6f} ≈ 1.0)")

    # Parameter count
    n_params = sum(p.numel() for p in attn.parameters())
    print(f"TemporalAttention params: {n_params:,}")
    print("\nAll temporal_attention tests PASSED.")