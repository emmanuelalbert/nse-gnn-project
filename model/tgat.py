"""
model/tgat.py
-------------
Temporal Graph Attention Network (TGAT) for Nifty 50 shock propagation.

Paper reference: Section III-B (Eq. 5), Table 1

    Full forward pass:

        x_raw ──► InputProjection ──► h^(0)
                                         │
               ┌─────────────────────────┘
               │    (repeated L times)
               ▼
        TGATEncoderLayer_l:
            TemporalAttention   ──► α_{t,τ}-weighted aggregation across T steps
            ResidualBlock       ──► GCN spatial aggregation on current graph
            Gating + LayerNorm  ──► h^(l+1)
               │
               └────────────────────────►  h_final (N, hidden_dim)
                                                │
                                         LinkDecoder
                                                │
                                     logits (N,N) ──► sigmoid ──► A_predicted

    Upgrade over paper TGCN:
      1. Multi-head temporal attention + learnable time encoding (TGAT) replaces
         the fixed α_{t,τ} scalar weights of the paper's TGCN.
      2. Gated combination of temporal + spatial representations per layer.
      3. 2-layer MLP decoder for better capacity while remaining lightweight.

    Paper hyperparameters (Table 1):
        num_layers:2  hidden_dim:64  num_heads:4  time_dim:16
        dropout:0.3   LayerNorm:True  residual:True  negative_sampling:1.0
        early_stop_patience:20

Usage:
    from model.tgat import TGAT, build_tgat
    model = build_tgat(in_features=1)          # paper default: mean_return only
    model = build_tgat(in_features=2)          # [mean_return, volatility]

    pos_logits, neg_logits, h = model(x_seq, edge_index, time_deltas,
                                      edge_weight, pos_edges, neg_edges)
    adj_pred = model.predict_adjacency(x_seq, edge_index, time_deltas)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from model.layers import InputProjection, LinkDecoder, TGATEncoderLayer
from model.temporal_attention import TemporalEncoding

logger = logging.getLogger(__name__)


class TGAT(nn.Module):
    """
    Temporal Graph Attention Network for directed link prediction on
    shock-period Granger causality graphs.

    Args:
        in_features    : Raw node feature dim F. Paper default: 1 (mean_return).
        hidden_dim     : Node embedding dim. Paper: 64.
        num_layers     : Encoder layers stacked. Paper: 2.
        num_heads      : Temporal attention heads. Paper: 4.
        time_dim       : Time encoding dim. Paper: 16.
        decoder_hidden : Link decoder MLP hidden dim. Paper: 32.
        dropout        : Dropout rate throughout. Paper: 0.3.
        use_layer_norm : LayerNorm after each layer. Paper: True.
        residual       : Residual connections in GCN blocks. Paper: True.
    """

    def __init__(
        self,
        in_features    : int   = len(CFG.NODE_FEATURE_COLS),
        hidden_dim     : int   = CFG.TGAT_HIDDEN_DIM,
        num_layers     : int   = CFG.TGAT_NUM_LAYERS,
        num_heads      : int   = CFG.TGAT_NUM_HEADS,
        time_dim       : int   = CFG.TGAT_TIME_DIM,
        decoder_hidden : int   = CFG.TGAT_DECODER_HIDDEN_DIM,
        dropout        : float = CFG.TGAT_DROPOUT,
        use_layer_norm : bool  = CFG.TGAT_USE_LAYER_NORM,
        residual       : bool  = CFG.TGAT_RESIDUAL,
    ):
        super().__init__()
        self.in_features    = in_features
        self.hidden_dim     = hidden_dim
        self.num_layers     = num_layers
        self.num_heads      = num_heads
        self.time_dim       = time_dim
        self.dropout_rate   = dropout
        self.use_layer_norm = use_layer_norm
        self.residual       = residual

        # 1. Input projection: raw features + time encoding → hidden_dim
        self.input_proj = InputProjection(
            in_features=in_features,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            dropout=dropout,
        )

        # 2. Shared time encoder (used in input projection AND encoder layers)
        self.time_enc = TemporalEncoding(time_dim)

        # 3. TGAT encoder stack
        self.encoder_layers = nn.ModuleList([
            TGATEncoderLayer(
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_norm=use_layer_norm,
            )
            for _ in range(num_layers)
        ])

        # 4. Link decoder: [h_i || h_j] → scalar logit
        self.decoder = LinkDecoder(
            hidden_dim=hidden_dim,
            decoder_hidden=decoder_hidden,
            dropout=dropout,
        )

        # 5. Output LayerNorm to stabilise embedding scale across shock periods
        self.output_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        self._init_weights()
        logger.info(
            f"TGAT | in={in_features} hidden={hidden_dim} "
            f"layers={num_layers} heads={num_heads} "
            f"params={self.count_parameters():,}"
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────
    # FORWARD (training: sampled edges only)
    # ─────────────────────────────────────────

    def forward(
        self,
        x_seq       : torch.Tensor,                   # (T, N, F) or (N, F)
        edge_index  : torch.Tensor,                   # (2, E)
        time_deltas : torch.Tensor,                   # (T,)
        edge_weight : Optional[torch.Tensor] = None,  # (E,)
        pos_edges   : Optional[torch.Tensor] = None,  # (2, E_pos)
        neg_edges   : Optional[torch.Tensor] = None,  # (2, E_neg)
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Encode temporal graph sequence, then decode sampled edge pairs.

        Returns:
            pos_logits : (E_pos,) raw scores for positive edges, or None.
            neg_logits : (E_neg,) raw scores for negative edges, or None.
            h_final    : (N, hidden_dim) node embeddings.
        """
        h = self.encode(x_seq, edge_index, time_deltas, edge_weight)

        pos_logits = (self.decoder.forward_pairs(h, pos_edges[0], pos_edges[1])
                      if pos_edges is not None else None)
        neg_logits = (self.decoder.forward_pairs(h, neg_edges[0], neg_edges[1])
                      if neg_edges is not None else None)

        return pos_logits, neg_logits, h

    # ─────────────────────────────────────────
    # ENCODER
    # ─────────────────────────────────────────

    def encode(
        self,
        x_seq       : torch.Tensor,
        edge_index  : torch.Tensor,
        time_deltas : torch.Tensor,
        edge_weight : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the full T-step encoder stack.

        For each layer:
          - TemporalAttention aggregates across ALL T past states.
          - ResidualBlock (2× GCN) refines the current-step representation.
          - Gating combines temporal + spatial outputs.
          - Only the LAST (current) step's embedding is updated per layer.
            Past steps serve as immutable context for temporal attention.

        Returns:
            h : (N, hidden_dim) — final node embeddings after L layers.
        """
        # Normalise (N, F) → (1, N, F) so all paths are 3-D
        if x_seq.dim() == 2:
            x_seq       = x_seq.unsqueeze(0)
            time_deltas = time_deltas[:1] if len(time_deltas) > 0 else torch.zeros(1)

        T, N, _ = x_seq.shape

        # Time encodings: (T, time_dim) → broadcast to (T, N, time_dim)
        t_enc = self.time_enc(time_deltas).unsqueeze(1).expand(T, N, self.time_dim)

        # Project each step independently: (T, N, hidden_dim)
        h_seq = torch.stack([
            self.input_proj(x_seq[t], t_enc[t]) for t in range(T)
        ])

        # Encoder layers: update only the current (last) step embedding
        for layer in self.encoder_layers:
            h_new, _ = layer(
                h_seq=h_seq,
                edge_index=edge_index,
                time_deltas=time_deltas,
                edge_weight=edge_weight,
                return_attn=False,
            )
            # Replace last step with updated embedding; preserve history
            h_seq = torch.cat([h_seq[:-1], h_new.unsqueeze(0)], dim=0)

        return self.output_norm(h_seq[-1])   # (N, hidden_dim)

    def encode_with_attention(
        self,
        x_seq       : torch.Tensor,
        edge_index  : torch.Tensor,
        time_deltas : torch.Tensor,
        edge_weight : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        encode() but also collects per-layer temporal attention weights.
        Used for visualising 'which time step did each node attend to most?'

        Returns:
            h_final   : (N, hidden_dim)
            attn_list : List[Tensor(N, num_heads, T)], one per encoder layer.
        """
        if x_seq.dim() == 2:
            x_seq       = x_seq.unsqueeze(0)
            time_deltas = time_deltas[:1] if len(time_deltas) > 0 else torch.zeros(1)

        T, N, _ = x_seq.shape
        t_enc   = self.time_enc(time_deltas).unsqueeze(1).expand(T, N, self.time_dim)
        h_seq   = torch.stack([self.input_proj(x_seq[t], t_enc[t]) for t in range(T)])

        attn_list = []
        for layer in self.encoder_layers:
            h_new, attn_w = layer(
                h_seq=h_seq, edge_index=edge_index,
                time_deltas=time_deltas, edge_weight=edge_weight,
                return_attn=True,
            )
            h_seq = torch.cat([h_seq[:-1], h_new.unsqueeze(0)], dim=0)
            if attn_w is not None:
                attn_list.append(attn_w)

        return self.output_norm(h_seq[-1]), attn_list

    # ─────────────────────────────────────────
    # INFERENCE: FULL ADJACENCY PREDICTION
    # ─────────────────────────────────────────

    @torch.no_grad()
    def predict_adjacency(
        self,
        x_seq         : torch.Tensor,
        edge_index    : torch.Tensor,
        time_deltas   : torch.Tensor,
        edge_weight   : Optional[torch.Tensor] = None,
        apply_sigmoid : bool = True,
    ) -> torch.Tensor:
        """
        Predict the full N×N adjacency matrix for a shock period.

        Paper Section III-B:
            "After training, the TGCN model outputs a series of predicted
             adjacency matrices for each time step within the shock period."

        Args:
            apply_sigmoid : True → probabilities ∈ (0,1).
                            False → raw logits.

        Returns:
            (N, N) FloatTensor — predicted adjacency matrix.
        """
        self.eval()
        h      = self.encode(x_seq, edge_index, time_deltas, edge_weight)
        logits = self.decoder.forward_all(h)         # (N, N)
        return torch.sigmoid(logits) if apply_sigmoid else logits

    @torch.no_grad()
    def predict_from_data(
        self,
        data          : Data,
        apply_sigmoid : bool = True,
    ) -> torch.Tensor:
        """
        Predict adjacency directly from a PyG Data snapshot.
        Convenience wrapper used by the inference engine.

        Args:
            data : PyG Data with fields x (N,F), edge_index (2,E), time_delta (E,).
        """
        x   = data.x.unsqueeze(0)          # (1, N, F)
        dt  = data.time_delta[:1]           # (1,)
        ew  = getattr(data, "edge_weight", None)
        return self.predict_adjacency(x, data.edge_index, dt, ew, apply_sigmoid)

    # ─────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters()
                   if not trainable_only or p.requires_grad)

    def parameter_breakdown(self) -> Dict[str, int]:
        return {name: sum(p.numel() for p in m.parameters())
                for name, m in self.named_children()}

    def summary(self) -> str:
        lines = [
            "=" * 56,
            "TGAT — Nifty 50 Shock Propagation",
            "=" * 56,
            f"  in_features    : {self.in_features}",
            f"  hidden_dim     : {self.hidden_dim}",
            f"  num_layers     : {self.num_layers}",
            f"  num_heads      : {self.num_heads}",
            f"  time_dim       : {self.time_dim}",
            f"  dropout        : {self.dropout_rate}",
            f"  use_layer_norm : {self.use_layer_norm}",
            f"  residual       : {self.residual}",
            "-" * 56,
        ]
        total = 0
        for name, n in self.parameter_breakdown().items():
            lines.append(f"  {name:<20} {n:>10,} params")
            total += n
        lines += ["-" * 56, f"  {'TOTAL':<20} {total:>10,} params", "=" * 56]
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save weights + config to a .pt checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "in_features"    : self.in_features,
                "hidden_dim"     : self.hidden_dim,
                "num_layers"     : self.num_layers,
                "num_heads"      : self.num_heads,
                "time_dim"       : self.time_dim,
                "decoder_hidden" : self.decoder.mlp[0].out_features,
                "dropout"        : self.dropout_rate,
                "use_layer_norm" : self.use_layer_norm,
                "residual"       : self.residual,
            },
        }, path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TGAT":
        """Reload a saved TGAT checkpoint."""
        ckpt  = torch.load(path, map_location=device)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        logger.info(f"Model loaded ← {path}  (device={device})")
        return model


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_tgat(in_features: Optional[int] = None) -> TGAT:
    """
    Build a TGAT model from config/settings.py.
    This is the single canonical constructor used by training/ and inference/.
    """
    model = TGAT(
        in_features    = in_features or len(CFG.NODE_FEATURE_COLS),
        hidden_dim     = CFG.TGAT_HIDDEN_DIM,
        num_layers     = CFG.TGAT_NUM_LAYERS,
        num_heads      = CFG.TGAT_NUM_HEADS,
        time_dim       = CFG.TGAT_TIME_DIM,
        decoder_hidden = CFG.TGAT_DECODER_HIDDEN_DIM,
        dropout        = CFG.TGAT_DROPOUT,
        use_layer_norm = CFG.TGAT_USE_LAYER_NORM,
        residual       = CFG.TGAT_RESIDUAL,
    )
    return model.to(CFG.DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level="INFO", format=CFG.LOG_FORMAT)
    torch.manual_seed(0)

    N, T, F = 50, 10, 1
    E       = 420

    x_seq       = torch.randn(T, N, F)
    edge_index  = torch.randint(0, N, (2, E))
    edge_weight = torch.ones(E)
    time_deltas = torch.linspace(0.0, 1.0, T)

    model = build_tgat(in_features=F)
    print(model.summary())

    # Training forward
    pos_edges = torch.randint(0, N, (2, 200))
    neg_edges = torch.randint(0, N, (2, 200))
    pos_logits, neg_logits, h = model(
        x_seq, edge_index, time_deltas, edge_weight, pos_edges, neg_edges
    )
    assert pos_logits.shape == (200,) and neg_logits.shape == (200,)
    assert h.shape == (N, CFG.TGAT_HIDDEN_DIM)
    print(f"\nTraining forward   OK : pos={pos_logits.shape}  h={h.shape}")

    # Loss + backward
    from model.loss import TGATLoss
    loss = TGATLoss()(pos_logits, neg_logits)
    loss.backward()
    print(f"Loss + backward    OK : {loss.item():.4f}")

    # Inference: full N×N adjacency
    adj = model.predict_adjacency(x_seq, edge_index, time_deltas, edge_weight)
    assert adj.shape == (N, N)
    assert 0.0 <= adj.min().item() and adj.max().item() <= 1.0
    print(f"predict_adjacency  OK : {adj.shape}  [{adj.min():.3f}, {adj.max():.3f}]")

    # Attention weights (interpretability)
    model.eval()
    h2, attn = model.encode_with_attention(x_seq, edge_index, time_deltas, edge_weight)
    assert len(attn) == CFG.TGAT_NUM_LAYERS
    assert attn[0].shape == (N, CFG.TGAT_NUM_HEADS, T)
    w_sum = attn[0].sum(dim=-1)
    assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-4)
    print(f"Attention weights  OK : {len(attn)} layers × {attn[0].shape}")

    # Save / load roundtrip
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "tgat_test.pt")
        model.save(ckpt)
        m2   = TGAT.load(ckpt)
        adj2 = m2.predict_adjacency(x_seq, edge_index, time_deltas, edge_weight)
        assert torch.allclose(adj, adj2, atol=1e-5)
    print(f"Save / Load        OK : weights match")

    # PyG Data object
    from torch_geometric.data import Data as PyGData
    data    = PyGData(x=x_seq[-1], edge_index=edge_index,
                      edge_weight=edge_weight, time_delta=time_deltas)
    adj_d   = model.predict_from_data(data)
    assert adj_d.shape == (N, N)
    print(f"predict_from_data  OK : {adj_d.shape}")

    # Multi-feature input
    model2 = build_tgat(in_features=2)
    x2     = torch.randn(T, N, 2)
    adj3   = model2.predict_adjacency(x2, edge_index, time_deltas, edge_weight)
    assert adj3.shape == (N, N)
    print(f"Multi-feature (F=2) OK : {adj3.shape}")

    print(f"\nTotal params: {model.count_parameters():,}")
    print("\nAll tgat.py tests PASSED.")