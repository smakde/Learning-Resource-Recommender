import streamlit as st
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

from src.hybrid import score_als, score_lightfm, score_content_knn, explain_item

st.set_page_config(page_title="Learning-Resource Recommender", layout="wide")
st.title("📚 Learning-Resource Recommender")

# ---- robust blender that forces equal length at n_items ----
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

with st.sidebar:
    st.header("Config")
    dataset_choice = st.selectbox("Dataset", ["sample", "movielens"])
    dataset_path = st.text_input(
        "Dataset folder",
        "data/sample" if dataset_choice == "sample" else "data/movielens_100k",
    )
    factors = st.slider("Latent factors", 8, 128, 32, 8)
    w_als = st.slider("Weight ALS", 0.0, 1.0, 0.5, 0.05)
    w_aux = st.slider("Weight LightFM/LMF", 0.0, 1.0, 0.4, 0.05)
    w_cnt = st.slider("Weight Content", 0.0, 1.0, 0.1, 0.05)
    k = st.slider("Top-N", 3, 30, 10, 1)

R, Xc, meta = prepare(dataset_path, dataset=dataset_choice)
st.success(f"Loaded {R.shape[0]} users × {R.shape[1]} items")

als = train_als(R, factors=factors)

aux = train_lightfm(R, Xc) if HAVE_LIGHTFM else train_lmf(R, factors=factors)

user = st.number_input("User index (0-based)", min_value=0, max_value=R.shape[0] - 1, value=0, step=1)

hist = R[user].indices
st.write(f"History count: {len(hist)}")

n_items = Xc.shape[0]
s_als = score_als(als, user, als.item_factors)
s_cnt = score_content_knn(Xc, hist)
s_aux = score_lightfm(aux, user, item_features=Xc) if HAVE_LIGHTFM else score_als(aux, user, aux.item_factors)

scores = blend_three(s_als, s_aux, s_cnt, w=(w_als, w_aux, w_cnt), n_items=n_items)
scores[hist] = -np.inf  # filter seen

top_idx = safe_topk(scores, k)
items = meta["items"].iloc[top_idx] if top_idx.size else meta["items"].iloc[[]]
tfidf_vocab = meta.get("tfidf_vocab")

st.subheader("Top-N Recommendations")
if top_idx.size == 0:
    st.info("No recommendable items (all items seen or scores unavailable). Try another user.")
else:
    for rank, (i_idx, row) in enumerate(zip(top_idx, items.itertuples(index=False)), 1):
        expl = ", ".join(explain_item(i_idx, Xc, hist, tfidf_vocab=tfidf_vocab, top_terms=5))
        st.markdown(
            f"**{rank}. {row.title}** — _score: {scores[i_idx]:.3f}_  "
            + (f" · <span style='opacity:0.7'>keywords: {expl}</span>" if expl else ""),
            unsafe_allow_html=True,
        )
