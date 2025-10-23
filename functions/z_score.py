import numpy as np

def zscore(x):
    mean, std = np.mean(x), np.std(x)
    if std == 0:
        return np.zeros_like(x, dtype=np.float32), mean, std
    return ((x - mean) / std).astype(np.float32), mean, std


def inverse_zscore(z, mean, std):
    if std == 0:
        return np.full_like(z, fill_value=mean, dtype=np.float32)
    return (z * std + mean).astype(np.float32)
