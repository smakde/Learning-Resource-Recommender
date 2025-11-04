import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from .ids import build_mappings
import os

def load_sample(path: str):
    ratings = pd.read_csv(os.path.join(path, "ratings.csv"))
    items = pd.read_csv(os.path.join(path, "items.csv"))
    return ratings, items

def build_interactions(ratings, user2idx, item2idx, implicit_weight=True):
    rows = ratings.user_id.map(user2idx)
    cols = ratings.item_id.map(item2idx)
    data = np.ones(len(ratings)) if implicit_weight else ratings.rating.values.astype(float)
    return sparse.coo_matrix((data, (rows, cols)),
                             shape=(len(user2idx), len(item2idx))).tocsr()

def build_content_matrix(items, item2idx, tfidf_min_df=1):
    tfidf = TfidfVectorizer(min_df=tfidf_min_df, stop_words="english")
    title_tfidf = tfidf.fit_transform(items["title"].fillna(""))
    genre_cols = [c for c in ["Comedy","Drama","Action","Romance","Sci-Fi","Thriller"] if c in items.columns]
    if genre_cols:
        genres = sparse.csr_matrix(items[genre_cols].fillna(0).values, dtype=float)
        Xc = sparse.hstack([title_tfidf, genres], format="csr")
    else:
        Xc = title_tfidf
    order = items.item_id.map(item2idx).values
    Xc = Xc[order]
    vocab = sorted(tfidf.vocabulary_.items(), key=lambda kv: kv[1])
    vocab = [w for w,_ in vocab]
    return Xc, vocab

def prepare(path, dataset="sample"):
    ratings, items = load_sample(path)
    user2idx, idx2user = build_mappings(ratings.user_id.tolist())
    item2idx, idx2item = build_mappings(ratings.item_id.tolist())
    R = build_interactions(ratings, user2idx, item2idx, implicit_weight=True)
    Xc, vocab = build_content_matrix(items, item2idx, tfidf_min_df=1)
    items_indexed = items.set_index("item_id").loc[idx2item].reset_index()
    meta = {
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": idx2user,
        "idx2item": idx2item,
        "items": items_indexed,
        "tfidf_vocab": vocab
    }
    return R, Xc, meta
