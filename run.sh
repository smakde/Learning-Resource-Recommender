#!/usr/bin/env bash
set -e

# Avoid OpenBLAS oversubscription warnings + slowdowns
export OPENBLAS_NUM_THREADS=1

python - <<'PY'
import numpy as np

from src.data import prepare
from src.als_train import train_als

# Prefer LightFM if available, else LogisticMF
try:
    from src.lightfm_train import train_lightfm
    HAVE_LIGHTFM = True
except Exception:
    HAVE_LIGHTFM = False
    from src.lmf_train import train_lmf

from src.hybrid import score_als, score_lightfm, score_content_knn
from src.metrics import precision_at_k, recall_at_k, mapk, ndcg_at_k

def _as_len(a, n):
    a = np.asarray(a).ravel()
    if a.shape[0] == n:
        return a
    out = np.full(n, -np.inf, dtype=float)
    out[: min(n, a.shape[0])] = a[: min(n, a.shape[0])]
    return out

def blend_three(s1, s2, s3, w, n_items):
    s1 = _as_len(s1, n_items)
    s2 = _as_len(s2, n_items)
    s3 = _as_len(s3, n_items)
    w1, w2, w3 = w
    return w1 * s1 + w2 * s2 + w3 * s3

def safe_topk(scores, k):
    scores = np.asarray(scores).ravel()
    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return np.array([], dtype=int)
    idx = np.where(finite_mask)[0]
    k = max(1, min(k, idx.size))
    part = np.argpartition(scores[idx], -k)[-k:]
    top = idx[part][np.argsort(scores[idx][part])[::-1]]
    return top

# Use the included sample dataset for a quick sanity run
R, Xc, meta = prepare("data/sample", dataset="sample")
n_items = Xc.shape[0]

als = train_als(R)
aux = train_lightfm(R, Xc) if HAVE_LIGHTFM else train_lmf(R)

test_gt, recommended = [], []

# Naive per-user holdout: remove last interaction (if any)
R_lil = R.tolil()
for u in range(R_lil.shape[0]):
    if R_lil.rows[u]:
        hold = R_lil.rows[u][-1]
        R_lil.rows[u] = R_lil.rows[u][:-1]
        R_lil.data[u] = R_lil.data[u][:-1]
        test_gt.append([hold])
    else:
        test_gt.append([])

R_train = R_lil.tocsr()

for u in range(R_train.shape[0]):
    s_als = score_als(als, u, als.item_factors)
    s_aux = score_lightfm(aux, u, item_features=Xc) if HAVE_LIGHTFM else score_als(aux, u, aux.item_factors)
    s_cnt = score_content_knn(Xc, R_train[u].indices)

    s = blend_three(s_als, s_aux, s_cnt, w=(0.5, 0.4, 0.1), n_items=n_items)
    s[R_train[u].indices] = -np.inf

    top = safe_topk(s, 10)
    recommended.append(top.tolist())

print("P@10", round(precision_at_k(recommended, test_gt, 10), 4))
print("R@10", round(recall_at_k(recommended, test_gt, 10), 4))
print("MAP@10", round(mapk(test_gt, recommended, 10), 4))
print("NDCG@10", round(ndcg_at_k(recommended, test_gt, 10), 4))
PY
