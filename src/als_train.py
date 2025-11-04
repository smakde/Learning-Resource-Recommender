from implicit.als import AlternatingLeastSquares

def train_als(R, factors=32, reg=1e-2, iters=15, alpha=40):
    Cui = (R * alpha).tocsr()
    model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters)
    model.fit(Cui.T.tocsr())  # keep input CSR
    return model
