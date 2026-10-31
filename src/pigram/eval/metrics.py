# src/pigram/eval/metrics.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_pearson(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) < 2:
        return 0.0

    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0

    value = pearsonr(y_true, y_pred)[0]

    if value is None or np.isnan(value):
        return 0.0

    return float(value)


def safe_spearman(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) < 2:
        return 0.0

    value = spearmanr(y_true, y_pred).correlation

    if value is None or np.isnan(value):
        return 0.0

    return float(value)


def safe_auroc(y_bin, y_score) -> float:
    y_bin = np.asarray(y_bin).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    if len(np.unique(y_bin)) < 2:
        return 0.5

    return float(roc_auc_score(y_bin, y_score))


def safe_auprc(y_bin, y_score) -> float:
    y_bin = np.asarray(y_bin).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    if len(np.unique(y_bin)) < 2:
        return 0.0

    return float(average_precision_score(y_bin, y_score))