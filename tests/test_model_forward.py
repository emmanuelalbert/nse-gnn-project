"""
tests/test_model_forward.py
-----------------------------
Unit tests for model/tgat.py — TGAT forward pass (paper Section III-B, Eq. 5).

Covers:
    - build_tgat() factory produces a model matching config/settings.py
    - Training forward: pos/neg logits shapes match sampled edge counts
    - encode(): output shape (N, hidden_dim), handles both (T,N,F) and
      (N,F) input
    - predict_adjacency(): full (N,N) matrix, values in [0,1] after sigmoid,
      raw logits unconstrained when apply_sigmoid=False
    - encode_with_attention(): per-layer attention weights shape
      (N, num_heads, T) and sum to 1 across the T axis (paper Eq. 5's
      alpha_{t,tau} is a proper attention distribution)
    - Multi-feature input (in_features > 1)
    - Save / load checkpoint roundtrip preserves predictions
    - predict_from_data() PyG Data convenience wrapper

Note: no dependency on model/loss.py — loss/backward is exercised with a
plain BCEWithLogitsLoss so this test suite stays scoped to the model's
forward-pass shape contract regardless of the loss module's specifics.

Run:
    pytest tests/test_model_forward.py -v
    python tests/test_model_forward.py
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch.nn as nn
from torch_geometric.data import Data as PyGData

from config import settings as CFG
from model.tgat import TGAT, build_tgat


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

N, T, F = 12, 6, 1     # small dims for fast tests
E = 40                 # graph edges
E_SAMPLE = 20           # sampled pos/neg edges for training forward


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def synthetic_graph():
    x_seq = torch.randn(T, N, F)
    edge_index = torch.randint(0, N, (2, E))
    edge_weight = torch.ones(E)
    time_deltas = torch.linspace(0.0, 1.0, T)
    return x_seq, edge_index, edge_weight, time_deltas


@pytest.fixture
def small_model():
    return TGAT(
        in_features=F,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        time_dim=8,
        decoder_hidden=8,
        dropout=0.0,
        use_layer_norm=True,
        residual=True,
    )


# ─────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────

class TestBuildTgat:

    def test_build_tgat_matches_settings(self):
        model = build_tgat(in_features=1)
        assert model.hidden_dim == CFG.TGAT_HIDDEN_DIM
        assert model.num_layers == CFG.TGAT_NUM_LAYERS
        assert model.num_heads == CFG.TGAT_NUM_HEADS
        assert model.time_dim == CFG.TGAT_TIME_DIM

    def test_build_tgat_default_in_features_from_config(self):
        model = build_tgat()
        assert model.in_features == len(CFG.NODE_FEATURE_COLS)

    def test_build_tgat_device_matches_config(self):
        model = build_tgat(in_features=1)
        param_device = next(model.parameters()).device.type
        assert param_device == CFG.DEVICE


# ─────────────────────────────────────────────
# TRAINING FORWARD  (pos/neg logits)
# ─────────────────────────────────────────────

class TestTrainingForward:

    def test_pos_neg_logits_shape(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        pos_edges = torch.randint(0, N, (2, E_SAMPLE))
        neg_edges = torch.randint(0, N, (2, E_SAMPLE))

        pos_logits, neg_logits, h = small_model(
            x_seq, edge_index, time_deltas, edge_weight, pos_edges, neg_edges
        )
        assert pos_logits.shape == (E_SAMPLE,)
        assert neg_logits.shape == (E_SAMPLE,)
        assert h.shape == (N, small_model.hidden_dim)

    def test_logits_are_finite(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        pos_edges = torch.randint(0, N, (2, E_SAMPLE))
        neg_edges = torch.randint(0, N, (2, E_SAMPLE))
        pos_logits, neg_logits, _ = small_model(
            x_seq, edge_index, time_deltas, edge_weight, pos_edges, neg_edges
        )
        assert torch.isfinite(pos_logits).all()
        assert torch.isfinite(neg_logits).all()

    def test_none_edges_give_none_logits(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        pos_logits, neg_logits, h = small_model(
            x_seq, edge_index, time_deltas, edge_weight, pos_edges=None, neg_edges=None
        )
        assert pos_logits is None
        assert neg_logits is None
        assert h.shape == (N, small_model.hidden_dim)

    def test_backward_pass_produces_gradients(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        pos_edges = torch.randint(0, N, (2, E_SAMPLE))
        neg_edges = torch.randint(0, N, (2, E_SAMPLE))
        pos_logits, neg_logits, _ = small_model(
            x_seq, edge_index, time_deltas, edge_weight, pos_edges, neg_edges
        )
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones(E_SAMPLE), torch.zeros(E_SAMPLE)])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        grad_norms = [p.grad.norm().item() for p in small_model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0
        assert any(g > 0 for g in grad_norms)


# ─────────────────────────────────────────────
# ENCODE()
# ─────────────────────────────────────────────

class TestEncode:

    def test_encode_output_shape(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        h = small_model.encode(x_seq, edge_index, time_deltas, edge_weight)
        assert h.shape == (N, small_model.hidden_dim)

    def test_encode_accepts_2d_input(self, small_model, synthetic_graph):
        """encode() must normalise (N,F) -> (1,N,F) internally."""
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        x_single = x_seq[-1]                 # (N, F)
        td_single = time_deltas[-1:].clone()  # (1,)
        h = small_model.encode(x_single, edge_index, td_single, edge_weight)
        assert h.shape == (N, small_model.hidden_dim)

    def test_encode_deterministic_in_eval_mode(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        small_model.eval()
        with torch.no_grad():
            h1 = small_model.encode(x_seq, edge_index, time_deltas, edge_weight)
            h2 = small_model.encode(x_seq, edge_index, time_deltas, edge_weight)
        assert torch.allclose(h1, h2)


# ─────────────────────────────────────────────
# predict_adjacency()
# ─────────────────────────────────────────────

class TestPredictAdjacency:

    def test_shape_is_n_by_n(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        adj = small_model.predict_adjacency(x_seq, edge_index, time_deltas, edge_weight)
        assert adj.shape == (N, N)

    def test_sigmoid_output_in_unit_interval(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        adj = small_model.predict_adjacency(
            x_seq, edge_index, time_deltas, edge_weight, apply_sigmoid=True
        )
        assert adj.min().item() >= 0.0
        assert adj.max().item() <= 1.0

    def test_raw_logits_not_bounded_to_unit_interval(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        logits = small_model.predict_adjacency(
            x_seq, edge_index, time_deltas, edge_weight, apply_sigmoid=False
        )
        probs = small_model.predict_adjacency(
            x_seq, edge_index, time_deltas, edge_weight, apply_sigmoid=True
        )
        assert torch.allclose(torch.sigmoid(logits), probs, atol=1e-5)

    def test_predict_adjacency_does_not_require_grad(self, small_model, synthetic_graph):
        """@torch.no_grad() decorator — output must not carry a grad_fn."""
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        adj = small_model.predict_adjacency(x_seq, edge_index, time_deltas, edge_weight)
        assert not adj.requires_grad

    def test_predict_adjacency_puts_model_in_eval_mode(self, small_model, synthetic_graph):
        small_model.train()
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        small_model.predict_adjacency(x_seq, edge_index, time_deltas, edge_weight)
        assert small_model.training is False


# ─────────────────────────────────────────────
# encode_with_attention()  — attention output range
# ─────────────────────────────────────────────

class TestAttentionWeights:

    def test_attention_list_length_matches_num_layers(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        small_model.eval()
        _, attn_list = small_model.encode_with_attention(x_seq, edge_index, time_deltas, edge_weight)
        assert len(attn_list) == small_model.num_layers

    def test_attention_shape_is_n_heads_t(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        small_model.eval()
        _, attn_list = small_model.encode_with_attention(x_seq, edge_index, time_deltas, edge_weight)
        assert attn_list[0].shape == (N, small_model.num_heads, T)

    def test_attention_sums_to_one_across_time(self, small_model, synthetic_graph):
        """
        alpha_{t,tau} in paper Eq. 5 is a proper attention distribution over
        past time steps — each (node, head) row must sum to ~1 across T.
        """
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        small_model.eval()
        _, attn_list = small_model.encode_with_attention(x_seq, edge_index, time_deltas, edge_weight)
        for attn in attn_list:
            row_sums = attn.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    def test_attention_weights_non_negative(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        small_model.eval()
        _, attn_list = small_model.encode_with_attention(x_seq, edge_index, time_deltas, edge_weight)
        for attn in attn_list:
            assert (attn >= 0.0).all()

    def test_attention_weights_bounded_by_one(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        small_model.eval()
        _, attn_list = small_model.encode_with_attention(x_seq, edge_index, time_deltas, edge_weight)
        for attn in attn_list:
            assert (attn <= 1.0 + 1e-6).all()


# ─────────────────────────────────────────────
# MULTI-FEATURE INPUT
# ─────────────────────────────────────────────

class TestMultiFeatureInput:

    def test_two_feature_input_works(self, synthetic_graph):
        _, edge_index, edge_weight, time_deltas = synthetic_graph
        model2 = TGAT(in_features=2, hidden_dim=16, num_layers=1,
                      num_heads=2, time_dim=8, decoder_hidden=8, dropout=0.0)
        x2 = torch.randn(T, N, 2)
        adj = model2.predict_adjacency(x2, edge_index, time_deltas, edge_weight)
        assert adj.shape == (N, N)

    def test_wrong_feature_dim_raises(self, small_model, synthetic_graph):
        _, edge_index, edge_weight, time_deltas = synthetic_graph
        x_wrong = torch.randn(T, N, 3)   # model built for F=1
        with pytest.raises(RuntimeError):
            small_model.encode(x_wrong, edge_index, time_deltas, edge_weight)


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────

class TestSaveLoad:

    def test_roundtrip_preserves_predictions(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        small_model.eval()
        adj_before = small_model.predict_adjacency(x_seq, edge_index, time_deltas, edge_weight)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "model.pt"
            small_model.save(str(ckpt_path))
            reloaded = TGAT.load(str(ckpt_path))
            reloaded.eval()
            adj_after = reloaded.predict_adjacency(x_seq, edge_index, time_deltas, edge_weight)

        assert torch.allclose(adj_before, adj_after, atol=1e-5)

    def test_loaded_config_matches_original(self, small_model):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "model.pt"
            small_model.save(str(ckpt_path))
            reloaded = TGAT.load(str(ckpt_path))

        assert reloaded.hidden_dim == small_model.hidden_dim
        assert reloaded.num_layers == small_model.num_layers
        assert reloaded.num_heads == small_model.num_heads


# ─────────────────────────────────────────────
# predict_from_data()  — PyG Data convenience wrapper
# ─────────────────────────────────────────────

class TestPredictFromData:

    def test_predict_from_data_shape(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        data = PyGData(
            x=x_seq[-1], edge_index=edge_index,
            edge_weight=edge_weight, time_delta=time_deltas,
        )
        adj = small_model.predict_from_data(data)
        assert adj.shape == (N, N)

    def test_predict_from_data_matches_direct_call(self, small_model, synthetic_graph):
        x_seq, edge_index, edge_weight, time_deltas = synthetic_graph
        small_model.eval()
        data = PyGData(
            x=x_seq[-1], edge_index=edge_index,
            edge_weight=edge_weight, time_delta=time_deltas,
        )
        adj_from_data = small_model.predict_from_data(data)
        adj_direct = small_model.predict_adjacency(
            x_seq[-1], edge_index, time_deltas[:1], edge_weight
        )
        assert torch.allclose(adj_from_data, adj_direct, atol=1e-5)


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

class TestUtilities:

    def test_count_parameters_positive(self, small_model):
        assert small_model.count_parameters() > 0

    def test_parameter_breakdown_sums_to_total(self, small_model):
        breakdown = small_model.parameter_breakdown()
        assert sum(breakdown.values()) == small_model.count_parameters(trainable_only=False)

    def test_summary_is_nonempty_string(self, small_model):
        summary = small_model.summary()
        assert isinstance(summary, str)
        assert "TGAT" in summary


# ─────────────────────────────────────────────
# MANUAL RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"])