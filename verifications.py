import torch
import numpy as np
from graph.dataset import ShockGraphDataset
from shock_detection.shock_registry import ShockRegistry
from data.storage import load_returns
from graph.adjacency import AdjacencyStore
from model.tgat import build_tgat
from training.checkpointing import CheckpointManager
from config import settings as CFG

returns = load_returns()
shocks = ShockRegistry().load()
target_names = {t['name'] for t in CFG.TARGET_SHOCK_PERIODS}
scoped = [s for s in shocks if s.name in target_names]

store = AdjacencyStore()
gc_matrices = {}
for s in scoped:
    mat, _ = store.load(s.name, stage='pre_training')
    gc_matrices[s.name] = mat

dataset = ShockGraphDataset(
    returns=returns, gc_matrices=gc_matrices, shocks=scoped,
    tickers=list(returns.columns), feature_cols=CFG.NODE_FEATURE_COLS,
)
dataset.build()

model = build_tgat(in_features=len(CFG.NODE_FEATURE_COLS))
ckpt = CheckpointManager(fold=1, monitor=CFG.CHECKPOINT_MONITOR, mode=CFG.CHECKPOINT_MODE)
model = ckpt.load_best(model, device=CFG.DEVICE)
model.eval()

device = torch.device(CFG.DEVICE)
shock = scoped[0]  # Global Financial Crisis
snaps = dataset.get_period_snapshots(shock.name)

x_seq = torch.stack([s.x for s in snaps], dim=0).to(device)
time_deltas = torch.cat([s.time_delta[:1] for s in snaps]).to(device)
edge_index = snaps[-1].edge_index.to(device)
edge_weight = getattr(snaps[-1], "edge_weight", None)
if edge_weight is not None:
    edge_weight = edge_weight.to(device)

with torch.no_grad():
    probs = model.predict_adjacency(x_seq, edge_index, time_deltas, edge_weight, apply_sigmoid=True)

probs_np = probs.cpu().numpy()
print(f"Shape: {probs_np.shape}")
print(f"Min: {probs_np.min():.6f}")
print(f"Max: {probs_np.max():.6f}")
print(f"Mean: {probs_np.mean():.6f}")
print(f"Median: {np.median(probs_np):.6f}")
print(f"Std: {probs_np.std():.6f}")
print(f"% above 0.5: {(probs_np >= 0.5).mean()*100:.2f}%")
print(f"\nHistogram (10 bins):")
hist, edges = np.histogram(probs_np, bins=10, range=(0,1))
for i in range(10):
    print(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}): {hist[i]}")