# src/hybrid.py
import numpy as np

def score_als(model, user_idx, item_factors):
    """
    Returns a score vector of length n_items using MF factors.
    Works for implicit.ALS and implicit.LogisticMF (both expose .user_factors / .item_factors).
    """
    u = model.user_factors[user_idx]
    return item_factors @ u

def score_lightfm(model, user_idx, item_features=None):
    """
    Returns a score vector of length n_items for a LightFM model.
    LightFM exposes .predict(user_ids, item_ids, item_features=?).
    """
    # If the model has item_embeddings, we can infer n_items.
    if hasattr(model, "item_embeddings"):
        n_items = model.item_embeddings.shape[0]
    else:
        # Fall back: try to infer from features
        n_items = item_features.shape[0] if item_features is not None else None
    if n_items is None:
        raise ValueError("Unable to infer number of items for LightFM scoring.")

    items = np.arange(n_items)
    return model.predict(user_ids=user_idx, item_ids=items, item_features=item_features)

def score_content_knn(Xc, user_profile_indices):
    """
    Content-only fallback: average of the user's consumed item vectors,
    then cosine-like dot-product similarity.
    """
    if len(user_profile_indices) == 0:
        return np.zeros(Xc.shape[0])
    prof = Xc[user_profile_indices].mean(axis=0)
    sims = Xc @ prof.T
    return np.asarray(sims).ravel()

def explain_item(item_idx, Xc, user_profile_indices, tfidf_vocab=None, top_terms=5):
    """
    Very lightweight explanation: show the TF-IDF terms that contribute most when
    projected on the user's average profile vector.
    """
    if len(user_profile_indices) == 0 or tfidf_vocab is None:
        return []
    prof = Xc[user_profile_indices].mean(axis=0)
    item_vec = Xc[item_idx]
    contrib = np.asarray(item_vec.multiply(prof)).ravel()
    take = min(len(tfidf_vocab), contrib.size)
    top_idx = np.argsort(contrib[:take])[-top_terms:][::-1]
    return [tfidf_vocab[i] for i in top_idx]
