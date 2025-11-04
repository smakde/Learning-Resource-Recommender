import numpy as np
def build_mappings(vals):
    uniques = sorted(set(vals))
    to_idx = {v:i for i,v in enumerate(uniques)}
    to_val = np.array(uniques)
    return to_idx, to_val
