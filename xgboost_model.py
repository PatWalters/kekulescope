#!/usr/bin/env python

import sys

import xgboost as xgb
from scipy.stats import spearmanr, pearsonr, kendalltau
from stratified_split import stratified_split
import pandas as pd

from pandas_desc import PandasDescriptors


def build_model(input_file_name):
    output_csv = input_file_name.replace(".sdf", "_xgboost.csv")
    pandas_descriptors = PandasDescriptors(['morgan2', 'descriptors'])
    desc_df = pandas_descriptors.from_molecule_file(input_file_name, name_field='ChEMBL_ID', activity_field="pIC50")
    descriptor_cols = [x for x in desc_df.columns if (x.startswith("B_") or x.startswith("D_"))]
    desc_vals = desc_df[descriptor_cols].values

    out_list = []
    for cycle in range(0, 10):
        train_X, val_X, test_X, train_Y, val_Y, test_Y = stratified_split(desc_vals, [float(x) for x in desc_df.pIC50])
        estimator = xgb.XGBRegressor()
        estimator.fit(train_X, train_Y)
        pred_Y = estimator.predict(test_X)
        for o, p in zip(test_Y, pred_Y):
            out_list.append([cycle, o, p])
        spearman_val = spearmanr(test_Y, pred_Y)[0] ** 2
        pearson_val = pearsonr(test_Y, pred_Y)[0]
        kendall_val = kendalltau(test_Y, pred_Y)[0]
        print(f"{input_file_name:10s} {spearman_val:.2f} {pearson_val:.2f} {kendall_val:.2f}")
    df = pd.DataFrame(out_list, columns=["cycle", "obs", "pred"])
    df.to_csv(output_csv, index=False)


for file_name in sys.argv[1:]:
    build_model(file_name)
