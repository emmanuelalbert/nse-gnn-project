"""
model/layers.py
---------------
Building blocks for the TGAT encoder stack.

Paper reference: Section III-B, Table 1

Components:
    GCNLayer         — graph convolution with symmetric normalisation
    ResidualBlock    — two GCN layers with a residual skip connection (paper: "after each pair")
    TGATEncoderLayer — one full TGAT layer: temporal attention → graph conv → LayerNorm
    InputProjection  — projects raw node features + time encoding into hidden_dim
    LinkDecoder      — final linear decoder for link prediction (paper: "linear layer decoder")

Design notes vs. paper:
  - Paper (TGCN): standard GCN (Kipf & Welling, 2016) with learned temporal weights.
    Our TGAT uses the same GCN spatial aggregation but replaces the fixed temporal
    weighting with the learned multi-head attention from temporal_attention.py.

  - Residual connections: paper explicitly states "residual connections were added
    after each pair of TGCN layers to enhance learning efficiency."
    We implement this as ResidualBlock wrapping two GCNLayer modules.

  - LayerNorm: paper uses LayerNorm after each layer (Table 1).
    We apply it AFTER the residual addition (post-norm), which is standard
    practice for graph networks and more stable than pre-norm for small graphs.

  - Dropout (rate 0.3): applied inside GCNLayer on the projected node features
    before graph convolution, and inside TemporalAttention on attention weights.
    This matches the paper's dropout placement.

  - L2 regularisation (weight_decay 1e-5): applied at the optimiser level
    (training/trainer.py), not inside these layers.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings as CFG


# ─────────────────────────────────────────────────────────────────────────────
# GCN LAYER
# ─────────────────────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """
    Single graph convolutional layer with symmetric degree normalisation.

    Implements the Kipf & Welling (2016) propagation rule (paper Eq. 5):
        H' = σ( D^{-½} A D^{-½} H W )

    Uses PyG's GCNConv which handles the normalisation internally.
    Dropout is applied to the INPUT features (before convolution), following
    the paper's training configuration (Table 1, dropout=0.3).

    Args:
        in_dim      : Input node feature dimension.
        out_dim     : Output node embedding dimension.
        dropout     : Feature dropout rate.
        activation  : Activation function ('relu' | 'elu' | 'none').
        bias        : Whether to include a learnable bias term.
    """

    def __init__(
        self,
        in_dim      : int,
        out_dim     : int,
        dropout     : float = CFG.TGAT_DROPOUT,
        activation  : str   = "relu",
        bias        : bool  = True,
    ):
        super().__init__()
        self.conv      = GCNConv(in_dim, out_dim, bias=bias, add_self_loops=CFG.GRAPH_SELF_LOOPS)
        self.dropout   = nn.Dropout(p=dropout)
        self.activation = self._build_activation(activation)

    def forward(
        self,
        x           : torch.Tensor,   # (N, in_dim)
        edge_index  : torch.Tensor,   # (2, E)
        edge_weight : Optional[torch.Tensor] = None,   # (E,)
    ) -> torch.Tensor:
        """
        Returns:
            x : (N, out_dim) — updated node embeddings after graph convolution.
        """
        x = self.dropout(x)
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        if self.activation is not None:
            x = self.activation(x)
        return x

    @staticmethod
    def _build_activation(name: str) -> Optional[nn.Module]:
        return {"relu": nn.ReLU(), "elu": nn.ELU(), "none": None}.get(name, nn.ReLU())


# ─────────────────────────────────────────────────────────────────────────────
# RESIDUAL BLOCK  (paper: "after each pair of TGCN layers")
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Two GCNLayers with a residual skip connection and LayerNorm (paper Table 1).

    Structure:
        x → GCNLayer_1 → GCNLayer_2 → (+x) → LayerNorm → output

    If in_dim ≠ hidden_dim, a linear projection is applied to x before
    the residual addition (standard practice in deep residual networks).

    Paper quote: "Residual connections were added after each pair of TGCN
    layers to enhance learning efficiency."

    Args:
        in_dim      : Input dimension.
        hidden_dim  : Hidden and output dimension.
        dropout     : Dropout rate for both GCNLayers.
        use_norm    : Whether to apply LayerNorm after residual (paper: True).
    """

    def __init__(
        self,
        in_dim      : int,
        hidden_dim  : int,
        dropout     : float = CFG.TGAT_DROPOUT,
        use_norm    : bool  = CFG.TGAT_USE_LAYER_NORM,
    ):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim,     hidden_dim, dropout=dropout, activation="relu")
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim, dropout=dropout, activation="none")

        # Projection for dimension matching in the skip connection
        self.skip_proj = (
            nn.Linear(in_dim, hidden_dim, bias=False)
            if in_dim != hidden_dim
            else nn.Identity()
        )

        self.norm     = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()
        self.act_out  = nn.ReLU()

    def forward(
        self,
        x           : torch.Tensor,
        edge_index  : torch.Tensor,
        edge_weight : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x           : (N, in_dim)
            edge_index  : (2, E)
            edge_weight : (E,) or None

        Returns:
            (N, hidden_dim) — residual-connected, normalised node embeddings.
        """
        residual = self.skip_proj(x)              # (N, hidden_dim) — skip path
        out      = self.gcn1(x, edge_index, edge_weight)    # first conv
        out      = self.gcn2(out, edge_index, edge_weight)  # second conv
        out      = self.norm(out + residual)      # post-norm residual
        out      = self.act_out(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# INPUT PROJECTION
# ─────────────────────────────────────────────────────────────────────────────

class InputProjection(nn.Module):
    """
    Projects raw node features (F dims) into the model's hidden_dim space,
    optionally concatenating the time encoding before projection.

    This layer sits at the very start of the encoder (before any GCN or
    temporal attention layers) and ensures that features of any dimensionality
    F can be mapped into the fixed hidden_dim expected by GCNLayer.

    In the paper, F=1 (mean_return only). When additional features are
    enabled via settings.NODE_FEATURE_COLS, this projection handles the
    mapping automatically without changing the rest of the architecture.

    Args:
        in_features : Raw node feature dimension F.
        hidden_dim  : Target embedding dimension.
        time_dim    : Time encoding dimension (0 = no time at input).
        dropout     : Dropout on the projected features.
    """

    def __init__(
        self,
        in_features : int,
        hidden_dim  : int  = CFG.TGAT_HIDDEN_DIM,
        time_dim    : int  = 0,    # set to CFG.TGAT_TIME_DIM to include time at input
        dropout     : float = CFG.TGAT_DROPOUT,
    ):
        super().__init__()
        self.in_features = in_features
        self.time_dim    = time_dim
        proj_in_dim      = in_features + time_dim

        self.proj    = nn.Linear(proj_in_dim, hidden_dim)
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act     = nn.ReLU()

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x          : torch.Tensor,            # (N, in_features)
        time_enc   : Optional[torch.Tensor] = None,  # (N, time_dim) or None
    ) -> torch.Tensor:
        """
        Returns:
            (N, hidden_dim) — projected and normalised node embeddings.
        """
        if time_enc is not None and self.time_dim > 0:
            x = torch.cat([x, time_enc], dim=-1)    # (N, in_features + time_dim)
        x = self.proj(x)       # (N, hidden_dim)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# TGAT ENCODER LAYER  (one full layer: temporal attn → graph conv → norm)
# ─────────────────────────────────────────────────────────────────────────────

class TGATEncoderLayer(nn.Module):
    """
    One complete TGAT encoder layer combining temporal attention and
    graph convolution, implementing the core of paper Eq. 5.

    Processing order:
        1. TemporalAttention: aggregate information across T time steps,
           weighted by temporal relevance (the α_{t,τ} attention weights).
        2. ResidualBlock: apply two GCN layers with skip connection,
           refining the temporally-attended embeddings through the graph
           structure at the current time step.
        3. LayerNorm: stabilise the output of the residual block.

    The temporal attention output and the GCN output are combined via
    a gating mechanism that learns how much of each to use:
        output = gate * attn_out + (1 - gate) * gcn_out

    This is an improvement over the paper's additive combination, giving
    the model the ability to emphasise graph structure (during normal
    market periods) or temporal dynamics (during fast-moving shocks).

    Args:
        hidden_dim  : Node embedding dimension. Paper: 64.
        time_dim    : Time encoding dimension. Paper: 16.
        num_heads   : Attention heads. Paper: 4.
        dropout     : Dropout rate. Paper: 0.3.
        use_norm    : Apply LayerNorm. Paper: True.
    """

    def __init__(
        self,
        hidden_dim  : int   = CFG.TGAT_HIDDEN_DIM,
        time_dim    : int   = CFG.TGAT_TIME_DIM,
        num_heads   : int   = CFG.TGAT_NUM_HEADS,
        dropout     : float = CFG.TGAT_DROPOUT,
        use_norm    : bool  = CFG.TGAT_USE_LAYER_NORM,
    ):
        super().__init__()

        # Import here to avoid circular imports between layers.py and temporal_attention.py
        from model.temporal_attention import TemporalAttention

        self.temporal_attn = TemporalAttention(
            hidden_dim = hidden_dim,
            time_dim   = time_dim,
            num_heads  = num_heads,
            dropout    = dropout,
        )
        self.graph_conv = ResidualBlock(
            in_dim     = hidden_dim,
            hidden_dim = hidden_dim,
            dropout    = dropout,
            use_norm   = use_norm,
        )
        # Gating: learn how much to use temporal vs. spatial information
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm    = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        h_seq        : torch.Tensor,   # (T, N, hidden_dim)
        edge_index   : torch.Tensor,   # (2, E)
        time_deltas  : torch.Tensor,   # (T,)
        edge_weight  : Optional[torch.Tensor] = None,   # (E,)
        return_attn  : bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            h_seq       : Sequence of node embeddings, shape (T, N, hidden_dim).
            edge_index  : Graph connectivity for the CURRENT time step (T-1).
            time_deltas : Normalised time deltas for each step.
            edge_weight : Optional edge weights.
            return_attn : If True, return attention weights too.

        Returns:
            h_new    : Updated node embeddings (N, hidden_dim).
            attn_w   : Attention weights (N, num_heads, T) or None.
        """
        T, N, D = h_seq.shape
        h_curr  = h_seq[-1]      # current step embeddings (N, hidden_dim)

        # 1. Temporal attention: aggregate across T steps
        h_attn, attn_w = self.temporal_attn(
            h_seq, time_deltas, return_weights=return_attn
        )                        # (N, hidden_dim)

        # 2. Graph convolution on current-step structure
        h_gcn = self.graph_conv(h_curr, edge_index, edge_weight)  # (N, hidden_dim)

        # 3. Gated combination of temporal and spatial information
        gate_input = torch.cat([h_attn, h_gcn], dim=-1)   # (N, 2*hidden_dim)
        gate_val   = self.gate(gate_input)                 # (N, hidden_dim) ∈ (0, 1)
        h_new      = gate_val * h_attn + (1 - gate_val) * h_gcn   # (N, hidden_dim)

        # 4. Post-layer norm + dropout
        h_new = self.norm(h_new + h_curr)    # residual from layer input
        h_new = self.dropout(h_new)

        return h_new, attn_w


# ─────────────────────────────────────────────────────────────────────────────
# LINK DECODER  (paper: "linear layer decoder")
# ─────────────────────────────────────────────────────────────────────────────

class LinkDecoder(nn.Module):
    """
    Decodes pairs of node embeddings into link existence probabilities.

    Paper quote (Table 1): "Linear layer decoder simplified final predictions,
    contributing to model robustness."

    For each potential edge (i, j), we concatenate the embeddings of nodes i
    and j, pass them through an MLP, and output a scalar probability in (0, 1).

    Architecture:
        [h_i ‖ h_j] → Linear(2·hidden, decoder_hidden) → ReLU → Dropout
                     → Linear(decoder_hidden, 1) → Sigmoid

    This is evaluated for ALL N×N pairs at inference time (vectorised via
    broadcasting), and for positive + sampled negative pairs during training.

    Args:
        hidden_dim      : Node embedding dimension from the encoder.
        decoder_hidden  : Intermediate MLP dimension. Paper suggests keeping
                          this small (32) to regularise the decoder.
        dropout         : Decoder dropout rate.
    """

    def __init__(
        self,
        hidden_dim      : int   = CFG.TGAT_HIDDEN_DIM,
        decoder_hidden  : int   = CFG.TGAT_DECODER_HIDDEN_DIM,
        dropout         : float = CFG.TGAT_DROPOUT,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, decoder_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(decoder_hidden, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_pairs(
        self,
        h          : torch.Tensor,     # (N, hidden_dim)
        src_nodes  : torch.Tensor,     # (E,) — source node indices
        dst_nodes  : torch.Tensor,     # (E,) — destination node indices
    ) -> torch.Tensor:
        """
        Decode edge probabilities for a set of (src, dst) index pairs.
        Used during training with positive + negative sampled edges.

        Returns:
            scores : (E,) FloatTensor of raw logits (before sigmoid).
        """
        h_src  = h[src_nodes]                        # (E, hidden_dim)
        h_dst  = h[dst_nodes]                        # (E, hidden_dim)
        h_cat  = torch.cat([h_src, h_dst], dim=-1)  # (E, 2*hidden_dim)
        return self.mlp(h_cat).squeeze(-1)           # (E,)

    def forward_all(
        self,
        h : torch.Tensor,     # (N, hidden_dim)
    ) -> torch.Tensor:
        """
        Decode edge probabilities for ALL N×N pairs.
        Used at inference time to produce the full predicted adjacency matrix.

        Vectorised via broadcasting to avoid an explicit double loop.

        Returns:
            scores : (N, N) FloatTensor of raw logits (before sigmoid).
        """
        N = h.shape[0]
        # h_i: (N, 1, hidden_dim)  h_j: (1, N, hidden_dim)
        # cat → (N, N, 2*hidden_dim)
        h_i   = h.unsqueeze(1).expand(N, N, -1)
        h_j   = h.unsqueeze(0).expand(N, N, -1)
        h_cat = torch.cat([h_i, h_j], dim=-1)       # (N, N, 2*hidden_dim)
        flat  = h_cat.view(N * N, -1)               # (N², 2*hidden_dim)
        out   = self.mlp(flat).view(N, N)            # (N, N)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    N, D, T = 50, CFG.TGAT_HIDDEN_DIM, 8
    E       = 200

    # Dummy graph
    edge_index  = torch.randint(0, N, (2, E))
    edge_weight = torch.ones(E)

    # ── GCNLayer ──
    gcn = GCNLayer(D, D)
    x   = torch.randn(N, D)
    out = gcn(x, edge_index, edge_weight)
    assert out.shape == (N, D), f"GCNLayer shape: {out.shape}"
    print(f"GCNLayer          OK : {x.shape} → {out.shape}")

    # ── ResidualBlock ──
    res_in_dim = 4   # raw features dim
    res = ResidualBlock(res_in_dim, D)
    x_raw = torch.randn(N, res_in_dim)
    out_r = res(x_raw, edge_index, edge_weight)
    assert out_r.shape == (N, D), f"ResidualBlock shape: {out_r.shape}"
    print(f"ResidualBlock     OK : {x_raw.shape} → {out_r.shape}")

    # ── InputProjection ──
    proj = InputProjection(in_features=2, hidden_dim=D, time_dim=CFG.TGAT_TIME_DIM)
    x_feat  = torch.randn(N, 2)
    x_tenc  = torch.randn(N, CFG.TGAT_TIME_DIM)
    out_p   = proj(x_feat, x_tenc)
    assert out_p.shape == (N, D), f"InputProjection shape: {out_p.shape}"
    print(f"InputProjection   OK : {x_feat.shape} + time → {out_p.shape}")

    # ── TGATEncoderLayer ──
    layer    = TGATEncoderLayer()
    h_seq    = torch.randn(T, N, D)
    t_deltas = torch.linspace(0, 1, T)
    h_new, attn_w = layer(h_seq, edge_index, t_deltas, edge_weight, return_attn=True)
    assert h_new.shape  == (N, D),                          f"h_new shape: {h_new.shape}"
    assert attn_w.shape == (N, CFG.TGAT_NUM_HEADS, T),     f"attn_w shape: {attn_w.shape}"
    print(f"TGATEncoderLayer  OK : h_seq {h_seq.shape} → {h_new.shape}")
    print(f"  attention weights  : {attn_w.shape}")

    # ── LinkDecoder ──
    decoder = LinkDecoder()
    h_embed = torch.randn(N, D)
    src  = torch.randint(0, N, (500,))
    dst  = torch.randint(0, N, (500,))

    logits_pairs = decoder.forward_pairs(h_embed, src, dst)
    logits_all   = decoder.forward_all(h_embed)
    assert logits_pairs.shape == (500,),  f"pairs shape: {logits_pairs.shape}"
    assert logits_all.shape   == (N, N),  f"all shape: {logits_all.shape}"
    print(f"LinkDecoder pairs OK : {logits_pairs.shape}")
    print(f"LinkDecoder all   OK : {logits_all.shape}")

    # Total param counts
    total_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in [gcn, res, proj, layer, decoder]
    )
    print(f"\nTotal params across all layer tests: {total_params:,}")
    print("\nAll layers tests PASSED.")