from implicit.lmf import LogisticMatrixFactorization
def train_lmf(R, factors=32, reg=1e-2, iters=30, neg_samples=5):
    model = LogisticMatrixFactorization(
        factors=factors,
        regularization=reg,
        iterations=iters,
        neg_prop=neg_samples
    )
    model.fit(R.T)  # item-user
    return model
