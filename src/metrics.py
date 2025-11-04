import numpy as np
def precision_at_k(recommended, ground_truth, k=10):
    return np.mean([len(set(r[:k]) & set(gt)) / max(k,1) for r,gt in zip(recommended, ground_truth)])
def recall_at_k(recommended, ground_truth, k=10):
    return np.mean([len(set(r[:k]) & set(gt)) / max(1,len(gt)) for r,gt in zip(recommended, ground_truth)])
def apk(actual, predicted, k=10):
    pred = predicted[:k]; score=0.0; hits=0.0
    for i,p in enumerate(pred, 1):
        if p in actual and p not in pred[:i-1]:
            hits += 1; score += hits / i
    return score / min(len(actual), k) if actual else 0.0
def mapk(actuals, predicteds, k=10):
    return np.mean([apk(a, p, k) for a,p in zip(actuals, predicteds)])
def ndcg_at_k(recommended, ground_truth, k=10):
    def dcg(rel): return sum((2**r - 1) / np.log2(i+2) for i,r in enumerate(rel))
    scores = []
    for rec, gt in zip(recommended, ground_truth):
        rel = [1 if i in gt else 0 for i in rec[:k]]
        ideal = sorted(rel, reverse=True)
        denom = dcg(ideal) or 1.0
        scores.append(dcg(rel) / denom)
    return np.mean(scores)
