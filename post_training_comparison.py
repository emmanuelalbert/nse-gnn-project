import numpy as np
from graph.adjacency import AdjacencyStore
from shock_detection.shock_registry import ShockRegistry
from config import settings as CFG

store = AdjacencyStore()
shocks = ShockRegistry().load()
target_names = {t['name'] for t in CFG.TARGET_SHOCK_PERIODS}
scoped = [s for s in shocks if s.name in target_names]

print(f'{"Shock":40s} {"Pre density":>12s} {"Post density":>13s} {"Change":>8s}')
for shock in scoped:
    try:
        pre_mat, tickers = store.load(shock.name, stage='pre_training')
        post_mat, _ = store.load(shock.name, stage='post_training')
        N = pre_mat.shape[0]
        pre_density = (pre_mat.sum() - np.trace(pre_mat)) / (N * (N - 1))
        post_density = (post_mat.sum() - np.trace(post_mat)) / (N * (N - 1))
        print(f'{shock.name:40s} {pre_density:12.4f} {post_density:13.4f} {post_density-pre_density:+8.4f}')
    except FileNotFoundError as e:
        print(f'{shock.name}: {e}')