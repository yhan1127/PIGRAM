# src/pigram/eval/downstream.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from pigram.data.loader import make_dummy_annotation, move_sample_to_device_with_ann
from pigram.train.pigram import load_pigram_from_checkpoint
from pigram.utils.io import load_pickle, save_json


DEFAULT_TARGETS = ["GFP", "NELF", "HNRNPC", "CHK2", "UBLCP1"]

DEFAULT_EXPECTED_FULL_COUNTS = {
    "GFP": {"n_candidates": 1875, "n_positive": 520},
    "NELF": {"n_candidates": 9833, "n_positive": 797},
    "HNRNPC": {"n_candidates": 3328, "n_positive": 233},
    "CHK2": {"n_candidates": 10000, "n_positive": 1255},
    "UBLCP1": {"n_candidates": 10000, "n_positive": 892},
}


def cfg_to_args(cfg: Dict[str, Any]) -> SimpleNamespace:
    flat: Dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return SimpleNamespace(**flat)


def parse_targets(x: Any) -> List[str]:
    if x is None:
        return DEFAULT_TARGETS
    if isinstance(x, list):
        return [str(item).upper() for item in x]
    raw = str(x).strip()
    if raw.lower() == "all":
        return DEFAULT_TARGETS
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def parse_splits(x: Any) -> List[str]:
    if x is None:
        return ["val", "test"]
    if isinstance(x, list):
        return [str(item) for item in x]
    return [item.strip() for item in str(x).split(",") if item.strip()]


def parse_int_list(x: Any) -> List[int]:
    if x is None:
        return [1, 3, 5, 10, 20, 50]
    if isinstance(x, list):
        return [int(item) for item in x]
    return [int(item.strip()) for item in str(x).split(",") if item.strip()]


def find_processed_pkl(processed_dir: Path, target: str, split: str) -> Path:
    candidates = [
        processed_dir / f"{target}_downstream_{split}_processed.pkl",
        processed_dir / f"{target}_downstream_{split}_processed_20260430_v6.pkl",
    ]
    for path in candidates:
        if path.exists():
            return path

    globbed = sorted(processed_dir.glob(f"{target}*{split}*processed*.pkl"))
    if globbed:
        return globbed[0]

    raise FileNotFoundError(
        f"No processed pkl found for {target}/{split} under {processed_dir}"
    )


def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def prepare_batch_for_pigram(
    sample: Dict[str, Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    ann = make_dummy_annotation(sample)
    return move_sample_to_device_with_ann(sample, ann, device)


def sigmoid_scaled(x: np.ndarray, scale: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x / float(scale)))


def flat_finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return arr[np.isfinite(arr)]


def safe_mean(x: np.ndarray) -> float:
    arr = flat_finite(x)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_max(x: np.ndarray) -> float:
    arr = flat_finite(x)
    if arr.size == 0:
        return float("nan")
    return float(np.max(arr))


def safe_min(x: np.ndarray) -> float:
    arr = flat_finite(x)
    if arr.size == 0:
        return float("nan")
    return float(np.min(arr))


def topk_mean(x: np.ndarray, k: int) -> float:
    arr = flat_finite(x)
    if arr.size == 0:
        return float("nan")
    k_eff = min(int(k), arr.size)
    if k_eff <= 0:
        return float("nan")
    idx = np.argpartition(arr, -k_eff)[-k_eff:]
    return float(np.mean(arr[idx]))


def topk_abs_mean(x: np.ndarray, k: int) -> float:
    arr = np.abs(flat_finite(x))
    if arr.size == 0:
        return float("nan")
    k_eff = min(int(k), arr.size)
    if k_eff <= 0:
        return float("nan")
    idx = np.argpartition(arr, -k_eff)[-k_eff:]
    return float(np.mean(arr[idx]))


def route_weighted_mean(score: np.ndarray, route: np.ndarray) -> float:
    score = np.asarray(score, dtype=np.float32)
    route = np.asarray(route, dtype=np.float32)
    if score.shape != route.shape:
        return float("nan")
    valid = np.isfinite(score) & np.isfinite(route)
    if valid.sum() == 0:
        return float("nan")
    denom = float(np.sum(route[valid])) + 1e-8
    return float(np.sum(score[valid] * route[valid]) / denom)


def sequence_gc_fraction(seq: str) -> float:
    seq = str(seq).upper()
    valid = [char for char in seq if char in {"A", "C", "G", "U", "T"}]
    if not valid:
        return float("nan")
    return float(sum(char in {"G", "C"} for char in valid) / len(valid))


def sequence_purine_fraction(seq: str) -> float:
    seq = str(seq).upper()
    valid = [char for char in seq if char in {"A", "C", "G", "U", "T"}]
    if not valid:
        return float("nan")
    return float(sum(char in {"A", "G"} for char in valid) / len(valid))


def score_candidate_readouts(
    logits: np.ndarray,
    sigmoid_scale: float,
    k_values: List[int],
) -> Dict[str, float]:
    prob = sigmoid_scaled(logits, sigmoid_scale)
    out = {
        "mean_prob": safe_mean(prob),
        "max_prob": safe_max(prob),
        "mean_logit": safe_mean(logits),
        "max_logit": safe_max(logits),
    }
    for k in k_values:
        out[f"top{k}_prob"] = topk_mean(prob, k)
        out[f"top{k}_logit"] = topk_mean(logits, k)
    return out


def score_one_sample(
    model,
    sample: Dict[str, Any],
    device: torch.device,
    sigmoid_scale: float,
    k_values: List[int],
) -> Dict[str, Any]:
    batch = prepare_batch_for_pigram(sample, device)

    start = time.perf_counter()
    with torch.no_grad():
        out = model(batch, sigmoid_scale=sigmoid_scale)
    end = time.perf_counter()

    final_logits = to_numpy(out["patch_motif_logits"]).astype(np.float32)
    grammar_ha_logits = to_numpy(out["head_a_logits"]).astype(np.float32)
    coarse_g_logits = to_numpy(out["grammar_logits"]).astype(np.float32)
    route_r = to_numpy(out["route_r"]).astype(np.float32)
    delta_logits = to_numpy(out["delta_logits"]).astype(np.float32)
    correction_logits = to_numpy(out["correction_logits"]).astype(np.float32)

    final_prob = sigmoid_scaled(final_logits, sigmoid_scale)
    grammar_ha_prob = sigmoid_scaled(grammar_ha_logits, sigmoid_scale)
    coarse_g_prob = sigmoid_scaled(coarse_g_logits, sigmoid_scale)

    seq = sample.get("partner_seq", "")

    scores = {
        "final_Sfinal_mean_prob": safe_mean(final_prob),
        "final_Sfinal_mean_logit": safe_mean(final_logits),
        "final_Sfinal_max_prob": safe_max(final_prob),
        "final_Sfinal_max_logit": safe_max(final_logits),
        "grammar_HA_mean_prob": safe_mean(grammar_ha_prob),
        "grammar_HA_mean_logit": safe_mean(grammar_ha_logits),
        "grammar_HA_max_prob": safe_max(grammar_ha_prob),
        "grammar_HA_max_logit": safe_max(grammar_ha_logits),
        "coarse_G_mean_prob": safe_mean(coarse_g_prob),
        "coarse_G_mean_logit": safe_mean(coarse_g_logits),
        "coarse_G_max_prob": safe_max(coarse_g_prob),
        "coarse_G_max_logit": safe_max(coarse_g_logits),
        "route_weighted_final_prob": route_weighted_mean(final_prob, route_r),
        "route_weighted_final_logit": route_weighted_mean(final_logits, route_r),
        "route_weighted_grammar_HA_prob": route_weighted_mean(grammar_ha_prob, route_r),
        "route_weighted_grammar_HA_logit": route_weighted_mean(grammar_ha_logits, route_r),
        "baseline_length": float(len(seq)),
        "baseline_gc_fraction": sequence_gc_fraction(seq),
        "baseline_purine_fraction": sequence_purine_fraction(seq),
        "route_mean": safe_mean(route_r),
        "route_max": safe_max(route_r),
        "delta_abs_mean": safe_mean(np.abs(delta_logits)),
        "correction_abs_mean": safe_mean(np.abs(correction_logits)),
        "correction_max": safe_max(correction_logits),
        "correction_min": safe_min(correction_logits),
        "n_patches": int(final_logits.shape[0]),
        "n_motifs": int(final_logits.shape[1]),
        "n_pairs": int(final_logits.size),
        "forward_seconds": float(end - start),
    }

    for k in k_values:
        scores[f"final_Sfinal_top{k}_prob"] = topk_mean(final_prob, k)
        scores[f"final_Sfinal_top{k}_logit"] = topk_mean(final_logits, k)
        scores[f"grammar_HA_top{k}_prob"] = topk_mean(grammar_ha_prob, k)
        scores[f"grammar_HA_top{k}_logit"] = topk_mean(grammar_ha_logits, k)
        scores[f"coarse_G_top{k}_prob"] = topk_mean(coarse_g_prob, k)
        scores[f"coarse_G_top{k}_logit"] = topk_mean(coarse_g_logits, k)
        scores[f"residual_correction_top{k}_logit"] = topk_mean(correction_logits, k)
        scores[f"residual_correction_abs_top{k}_logit"] = topk_abs_mean(correction_logits, k)
        scores[f"residual_delta_abs_top{k}_logit"] = topk_abs_mean(delta_logits, k)

    return scores


def assign_topn_labels_for_target(
    df: pd.DataFrame,
    direction: str,
    n_positive: int,
) -> np.ndarray:
    affinity = pd.to_numeric(df["affinity"], errors="coerce").to_numpy(dtype=float)
    y = np.full(len(df), np.nan, dtype=float)

    valid_idx = np.where(np.isfinite(affinity))[0]
    if valid_idx.size == 0:
        return y

    n_positive = min(int(n_positive), valid_idx.size)
    if n_positive <= 0:
        return y

    if direction == "low":
        ordered = valid_idx[np.argsort(affinity[valid_idx])]
    elif direction == "high":
        ordered = valid_idx[np.argsort(-affinity[valid_idx])]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    y[valid_idx] = 0.0
    y[ordered[:n_positive]] = 1.0
    return y


def add_faformer_labels(
    scores_df: pd.DataFrame,
    expected_counts: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    df = scores_df.copy()
    df["label_low_topN"] = np.nan
    df["label_high_topN"] = np.nan

    for target in sorted(df["target"].unique()):
        mask = df["target"] == target
        target_df = df[mask].copy()
        if target not in expected_counts:
            continue

        n_positive = expected_counts[target]["n_positive"]

        df.loc[mask, "label_low_topN"] = assign_topn_labels_for_target(
            target_df,
            direction="low",
            n_positive=n_positive,
        )
        df.loc[mask, "label_high_topN"] = assign_topn_labels_for_target(
            target_df,
            direction="high",
            n_positive=n_positive,
        )

    return df


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if y_true.size == 0:
        return float("nan")
    k_eff = min(int(k), y_true.size)
    order = np.argsort(-y_score)
    return float(np.mean(y_true[order[:k_eff]]))


def enrichment_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    precision = precision_at_k(y_true, y_score, k)
    base_rate = float(np.nanmean(y_true))
    if not np.isfinite(precision) or base_rate <= 0:
        return float("nan")
    return float(precision / base_rate)


def average_precision_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if y_true.size == 0 or np.sum(y_true) <= 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1)
    precision = cum_pos / ranks
    return float(np.sum(precision * y_sorted) / np.sum(y_sorted))


def auroc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)

    pos = y_true == 1
    n_pos = float(np.sum(pos))
    n_neg = float(np.sum(~pos))
    sum_ranks_pos = float(np.sum(ranks[pos]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_metrics_for_group(
    df: pd.DataFrame,
    target: str,
    split: str,
    label_mode: str,
    score_name: str,
) -> Dict[str, Any]:
    y_true = pd.to_numeric(df[label_mode], errors="coerce").to_numpy(dtype=float)
    y_score = pd.to_numeric(df[score_name], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    y_valid = y_true[valid]

    return {
        "target": target,
        "split": split,
        "label_mode": label_mode,
        "score_name": score_name,
        "n_candidates": int(valid.sum()),
        "n_positive": int(np.nansum(y_valid)) if valid.sum() else 0,
        "positive_rate": float(np.nanmean(y_valid)) if valid.sum() else float("nan"),
        "top10_precision": precision_at_k(y_true, y_score, 10),
        "top50_precision": precision_at_k(y_true, y_score, 50),
        "ef10": enrichment_at_k(y_true, y_score, 10),
        "ef50": enrichment_at_k(y_true, y_score, 50),
        "prauc": average_precision_score_np(y_true, y_score),
        "auroc": auroc_rank(y_true, y_score),
    }


def compute_full_metrics(scores_df: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    rows = []
    label_modes = ["label_low_topN", "label_high_topN"]

    for target in sorted(scores_df["target"].unique()):
        target_df = scores_df[scores_df["target"] == target].copy()
        groups = [("all", target_df)]

        for split in sorted(target_df["split"].unique()):
            groups.append((split, target_df[target_df["split"] == split].copy()))

        for split_name, group_df in groups:
            for label_mode in label_modes:
                if group_df[label_mode].notna().sum() == 0:
                    continue

                for score_name in score_cols:
                    if score_name not in group_df.columns:
                        continue

                    rows.append(
                        compute_metrics_for_group(
                            df=group_df,
                            target=target,
                            split=split_name,
                            label_mode=label_mode,
                            score_name=score_name,
                        )
                    )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics

    macro_rows = []
    metric_cols = [
        "top10_precision",
        "top50_precision",
        "ef10",
        "ef50",
        "prauc",
        "auroc",
    ]

    for (split, label_mode, score_name), sub in metrics.groupby(
        ["split", "label_mode", "score_name"]
    ):
        row = {
            "target": "MACRO",
            "split": split,
            "label_mode": label_mode,
            "score_name": score_name,
            "n_candidates": int(sub["n_candidates"].sum()),
            "n_positive": int(sub["n_positive"].sum()),
            "positive_rate": float(sub["positive_rate"].mean()),
        }
        for metric in metric_cols:
            row[metric] = float(sub[metric].mean())
        macro_rows.append(row)

    return pd.concat([metrics, pd.DataFrame(macro_rows)], ignore_index=True)


def make_score_cols(k_values: List[int]) -> List[str]:
    cols = [
        "final_Sfinal_mean_prob",
        "final_Sfinal_mean_logit",
        "final_Sfinal_max_prob",
        "final_Sfinal_max_logit",
        "grammar_HA_mean_prob",
        "grammar_HA_mean_logit",
        "grammar_HA_max_prob",
        "grammar_HA_max_logit",
        "coarse_G_mean_prob",
        "coarse_G_mean_logit",
        "route_weighted_final_prob",
        "route_weighted_final_logit",
        "route_weighted_grammar_HA_prob",
        "route_weighted_grammar_HA_logit",
    ]

    for k in k_values:
        cols.extend(
            [
                f"final_Sfinal_top{k}_prob",
                f"final_Sfinal_top{k}_logit",
                f"grammar_HA_top{k}_prob",
                f"grammar_HA_top{k}_logit",
                f"coarse_G_top{k}_prob",
                f"coarse_G_top{k}_logit",
                f"residual_correction_top{k}_logit",
                f"residual_correction_abs_top{k}_logit",
                f"residual_delta_abs_top{k}_logit",
            ]
        )

    return cols


def selection_score_value(
    row: pd.Series,
    selection_metric: str,
    weights: Dict[str, float],
) -> float:
    if selection_metric != "composite":
        return float(row[selection_metric])
    return float(
        weights["prauc_weight"] * row["prauc"]
        + weights["top10_weight"] * row["top10_precision"]
        + weights["top50_weight"] * row["top50_precision"]
    )


def choose_target_readout(
    metrics: pd.DataFrame,
    target: str,
    selection_scores: List[str],
    selection_metric: str,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    val = metrics[
        (metrics["target"] == target)
        & (metrics["split"] == "val")
        & (metrics["label_mode"].isin(["label_low_topN", "label_high_topN"]))
        & (metrics["score_name"].isin(selection_scores))
    ].copy()

    if val.empty:
        raise ValueError(f"No validation metrics for target={target}")

    val["selection_value"] = val.apply(
        lambda row: selection_score_value(row, selection_metric, weights),
        axis=1,
    )
    val = val.sort_values(
        ["selection_value", "top50_precision", "top10_precision", "auroc"],
        ascending=[False, False, False, False],
    )

    best = val.iloc[0]
    other_label = "label_high_topN" if best["label_mode"] == "label_low_topN" else "label_low_topN"
    other = val[val["label_mode"] == other_label].copy()
    other_best = float("nan")
    if not other.empty:
        other_best = float(other["selection_value"].max())

    return {
        "target": target,
        "direction": "low" if best["label_mode"] == "label_low_topN" else "high",
        "label_mode": best["label_mode"],
        "selected_score_name": best["score_name"],
        "selection_metric": selection_metric,
        "val_selection_value": float(best["selection_value"]),
        "val_top10_precision": float(best["top10_precision"]),
        "val_top50_precision": float(best["top50_precision"]),
        "val_prauc": float(best["prauc"]),
        "val_auroc": float(best["auroc"]),
        "best_opposite_direction_value": other_best,
        "direction_margin": (
            float(best["selection_value"] - other_best)
            if np.isfinite(other_best)
            else float("nan")
        ),
    }


def choose_all_readouts(
    metrics: pd.DataFrame,
    targets: List[str],
    selection_scores: List[str],
    selection_metric: str,
    weights: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    for target in targets:
        rows.append(
            choose_target_readout(
                metrics=metrics,
                target=target,
                selection_scores=selection_scores,
                selection_metric=selection_metric,
                weights=weights,
            )
        )
    return pd.DataFrame(rows)


def extract_metric_row(
    metrics: pd.DataFrame,
    target: str,
    split: str,
    label_mode: str,
    score_name: str,
) -> pd.Series | None:
    sub = metrics[
        (metrics["target"] == target)
        & (metrics["split"] == split)
        & (metrics["label_mode"] == label_mode)
        & (metrics["score_name"] == score_name)
    ].copy()
    if sub.empty:
        return None
    return sub.iloc[0]


def make_selected_test_table(
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, selection in selected.iterrows():
        metric_row = extract_metric_row(
            metrics=metrics,
            target=selection["target"],
            split="test",
            label_mode=selection["label_mode"],
            score_name=selection["selected_score_name"],
        )
        if metric_row is None:
            continue

        rows.append(
            {
                "target": selection["target"],
                "direction": selection["direction"],
                "selected_score_name": selection["selected_score_name"],
                "val_selection_value": selection["val_selection_value"],
                "test_top10_precision": float(metric_row["top10_precision"]),
                "test_top50_precision": float(metric_row["top50_precision"]),
                "test_prauc": float(metric_row["prauc"]),
                "test_auroc": float(metric_row["auroc"]),
                "test_ef10": float(metric_row["ef10"]),
                "test_ef50": float(metric_row["ef50"]),
                "n_candidates": int(metric_row["n_candidates"]),
                "n_positive": int(metric_row["n_positive"]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    metric_cols = [
        "test_top10_precision",
        "test_top50_precision",
        "test_prauc",
        "test_auroc",
        "test_ef10",
        "test_ef50",
    ]

    macro = {
        "target": "MACRO",
        "direction": "",
        "selected_score_name": "",
        "val_selection_value": float(out["val_selection_value"].mean()),
        "n_candidates": int(out["n_candidates"].sum()),
        "n_positive": int(out["n_positive"].sum()),
    }
    for col in metric_cols:
        macro[col] = float(out[col].mean())

    return pd.concat([out, pd.DataFrame([macro])], ignore_index=True)


def score_downstream_samples(
    model,
    sigmoid_scale: float,
    processed_dir: Path,
    targets: List[str],
    splits: List[str],
    k_values: List[int],
    device: torch.device,
) -> pd.DataFrame:
    rows = []

    for target in targets:
        for split in splits:
            pkl_path = find_processed_pkl(processed_dir, target, split)
            samples = load_pickle(pkl_path)
            print(f"[Load] {target}/{split}: {pkl_path} | n={len(samples)}")

            for sample_index, sample in enumerate(samples):
                scores = score_one_sample(
                    model=model,
                    sample=sample,
                    device=device,
                    sigmoid_scale=sigmoid_scale,
                    k_values=k_values,
                )

                row = {
                    "target": target,
                    "split": split,
                    "sample_index": sample_index,
                    "processed_pkl": str(pkl_path),
                    "aptamer_id": sample.get("aptamer_id", sample.get("pdb", "")),
                    "pdb": sample.get("pdb", ""),
                    "partner_seq": sample.get("partner_seq", ""),
                    "length": len(str(sample.get("partner_seq", ""))),
                    "affinity": sample.get("affinity", np.nan),
                    "label_high_topN_existing": sample.get("label_high_topN", np.nan),
                    "label_low_topN_existing": sample.get("label_low_topN", np.nan),
                }
                row.update(scores)
                rows.append(row)

                if (sample_index + 1) % 100 == 0:
                    print(f"[Score] {target}/{split}: {sample_index + 1}/{len(samples)}")

    return pd.DataFrame(rows)


def expected_counts_from_config(cfg: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    expected_positives = cfg.get("preprocess", {}).get("expected_positives", {})
    if not expected_positives:
        return DEFAULT_EXPECTED_FULL_COUNTS

    out = {}
    for target, n_positive in expected_positives.items():
        target = str(target).upper()
        out[target] = {
            "n_candidates": DEFAULT_EXPECTED_FULL_COUNTS.get(target, {}).get("n_candidates", 0),
            "n_positive": int(n_positive),
        }

    return out


def run_downstream_eval(cfg: Dict[str, Any]) -> None:
    args = cfg_to_args(cfg)
    device = torch.device(
        args.device
        if torch.cuda.is_available() and str(args.device).lower() == "cuda"
        else "cpu"
    )
    print(f"Using device: {device}")

    model, model_args, ckpt, fine_rule_map = load_pigram_from_checkpoint(
        checkpoint_path=args.pigram_ckpt,
        eval_args=args,
        device=device,
    )

    sigmoid_scale = float(model_args.sigmoid_scale)

    processed_dir = Path(args.processed_downstream_dir)
    out_dir = Path(args.pigram_eval_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = parse_targets(cfg.get("eval", {}).get("targets", DEFAULT_TARGETS))
    splits = parse_splits(cfg.get("eval", {}).get("splits", ["val", "test"]))
    k_values = parse_int_list(cfg.get("eval", {}).get("k_values", [1, 3, 5, 10, 20, 50]))

    print("[Targets]", targets)
    print("[Splits]", splits)
    print("[K values]", k_values)

    scores_df = score_downstream_samples(
        model=model,
        sigmoid_scale=sigmoid_scale,
        processed_dir=processed_dir,
        targets=targets,
        splits=splits,
        k_values=k_values,
        device=device,
    )

    expected_counts = expected_counts_from_config(cfg)
    scores_df = add_faformer_labels(scores_df, expected_counts)

    score_cols = make_score_cols(k_values)
    metrics_df = compute_full_metrics(scores_df, score_cols)

    selection_scores = [
        col
        for col in score_cols
        if col.startswith("final_Sfinal") or col.startswith("grammar_HA")
    ]

    selection_cfg = cfg.get("selection", {})
    weights = {
        "prauc_weight": float(selection_cfg.get("prauc_weight", 0.4)),
        "top10_weight": float(selection_cfg.get("top10_weight", 0.3)),
        "top50_weight": float(selection_cfg.get("top50_weight", 0.3)),
    }
    selection_metric = str(cfg.get("eval", {}).get("selection_metric", "composite"))

    selected_df = choose_all_readouts(
        metrics=metrics_df,
        targets=targets,
        selection_scores=selection_scores,
        selection_metric=selection_metric,
        weights=weights,
    )

    selected_test_df = make_selected_test_table(
        metrics=metrics_df,
        selected=selected_df,
    )

    scores_csv = out_dir / "scores.csv"
    metrics_csv = out_dir / "metrics.csv"
    selected_csv = out_dir / "selected_readouts.csv"
    selected_test_csv = out_dir / "selected_test.csv"
    summary_json = out_dir / "summary.json"

    scores_df.to_csv(scores_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    selected_df.to_csv(selected_csv, index=False)
    selected_test_df.to_csv(selected_test_csv, index=False)

    summary = {
        "checkpoint": str(args.pigram_ckpt),
        "foundation_checkpoint": str(args.foundation_ckpt),
        "sigmoid_scale": sigmoid_scale,
        "targets": targets,
        "splits": splits,
        "k_values": k_values,
        "selection_metric": selection_metric,
        "selection_weights": weights,
        "fine_rule_map": {str(k): int(v) for k, v in fine_rule_map.items()},
        "outputs": {
            "scores_csv": str(scores_csv),
            "metrics_csv": str(metrics_csv),
            "selected_csv": str(selected_csv),
            "selected_test_csv": str(selected_test_csv),
        },
        "selected_test_macro": (
            selected_test_df[selected_test_df["target"] == "MACRO"].to_dict("records")
            if not selected_test_df.empty
            else []
        ),
    }
    save_json(summary, summary_json)

    print("\nSaved downstream PIGRAM evaluation:")
    print(scores_csv)
    print(metrics_csv)
    print(selected_csv)
    print(selected_test_csv)
    print(summary_json)

    print("\n=== Selected test results ===")
    if not selected_test_df.empty:
        print(selected_test_df.to_string(index=False))
    else:
        print("No selected test results.")