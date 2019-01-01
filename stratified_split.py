from sklearn.model_selection import train_test_split
import numpy as np


def stratified_split(x, y, split_bins=3, test_frac=0.3):
    counts_1, bins_1 = np.histogram(y, split_bins)
    y_binned_1 = np.digitize(y, bins_1[0:split_bins])
    train_x, tmp_x, train_y, tmp_y = train_test_split(x, y, test_size=test_frac, stratify=y_binned_1)
    counts_2, bins_2 = np.histogram(tmp_y, 3)
    y_binned_2 = np.digitize(tmp_y, bins_2[0:split_bins])
    test_x, val_x, test_y, val_y = train_test_split(tmp_x, tmp_y, test_size=0.5, stratify=y_binned_2)
    return train_x, val_x, test_x, train_y, val_y, test_y
