# src/pigram/eval/baseline.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from pigram.data.loader import make_dummy_annotation, move_sample_to_device, move_sample_to_device_with_ann
from pigram.eval.downstream import (
    add_faformer_labels,
    compute_full_metrics,
    expected_counts_from_config,
    find_processed_pkl,
    parse_int_list,
    parse_splits,
    parse_targets,
    score_candidate_readouts,
)
from pigram.train.baseline import build_model as build_single_stage_model
from pigram.train.proto_res import build_model as build_proto_res_model
from pigram.train.pigram import load_pigram_from_checkpoint
from pigram.utils.io import load_pickle, save_json


def cfg_to_args(cfg: Dict[str, Any]) -> SimpleNamespace:
    flat: Dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return SimpleNamespace(**flat)


def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_mean(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def safe_max(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.max())


def load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def namespace_from_checkpoint_config(ckpt: Dict[str, Any]) -> SimpleNamespace:
    cfg = ckpt.get("config", {})
    if not isinstance(cfg, dict):
        raise ValueError("Checkpoint config must be a dictionary.")
    return SimpleNamespace(**cfg)


def load_single_stage_baseline(
    model_name: str,
    ckpt_path: str | Path,
    device: torch.device,
):
    ckpt = load_checkpoint(ckpt_path, device)
    cfg = namespace_from_checkpoint_config(ckpt)
    checkpoint_model_name = ckpt.get("model_name", model_name)

    model = build_single_stage_model(
        model_name=checkpoint_model_name,
        cfg=cfg,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print(
        f"[Loaded] {model_name}: {ckpt_path} | "
        f"checkpoint_model_name={checkpoint_model_name} | epoch={ckpt.get('epoch')}"
    )

    return model, cfg


def load_proto_res(
    ckpt_path: str | Path,
    device: torch.device,
):
    ckpt = load_checkpoint(ckpt_path, device)
    cfg = namespace_from_checkpoint_config(ckpt)

    model = build_proto_res_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print(f"[Loaded] proto_res: {ckpt_path} | epoch={ckpt.get('epoch')}")
    return model, cfg


def prepare_batch_for_pigram(sample: Dict[str, Any], device: torch.device):
    ann = make_dummy_annotation(sample)
    return move_sample_to_device_with_ann(sample, ann, device)


def score_matrix_single_stage(
    model,
    sample: Dict[str, Any],
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    batch = move_sample_to_device(sample, device)
    with torch.no_grad():
        out = model(batch)

    logits = to_numpy(out["patch_motif_logits"]).astype(np.float32)
    return logits, {
        "route_mean": float("nan"),
        "route_max": float("nan"),
    }


def score_matrix_proto_res(
    model,
    sample: Dict[str, Any],
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    batch = move_sample_to_device(sample, device)
    with torch.no_grad():
        out = model.forward_stage2(batch)

    logits = to_numpy(out["patch_motif_logits"]).astype(np.float32)
    route = to_numpy(out["route_r"]).astype(np.float32)

    return logits, {
        "route_mean": safe_mean(route),
        "route_max": safe_max(route),
    }


def score_matrix_pigram(
    model,
    sample: Dict[str, Any],
    device: torch.device,
    sigmoid_scale: float,
    component: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    batch = prepare_batch_for_pigram(sample, device)

    with torch.no_grad():
        out = model(batch, sigmoid_scale=sigmoid_scale)

    if component == "grammar":
        logits = to_numpy(out["head_a_logits"]).astype(np.float32)
    elif component == "final":
        logits = to_numpy(out["patch_motif_logits"]).astype(np.float32)
    else:
        raise ValueError(f"Unsupported PIGRAM component: {component}")

    route = to_numpy(out["route_r"]).astype(np.float32)

    return logits, {
        "route_mean": safe_mean(route),
        "route_max": safe_max(route),
    }


def load_all_models(
    args: SimpleNamespace,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    models: Dict[str, Dict[str, Any]] = {}

    concat_mlp, concat_cfg = load_single_stage_baseline(
        model_name="concat_mlp",
        ckpt_path=args.concat_mlp_ckpt,
        device=device,
    )
    models["Concat-MLP"] = {
        "model": concat_mlp,
        "type": "single_stage",
        "sigmoid_scale": float(getattr(concat_cfg, "sigmoid_scale", 2.0)),
    }

    bilinear, bilinear_cfg = load_single_stage_baseline(
        model_name="bilinear",
        ckpt_path=args.bilinear_ckpt,
        device=device,
    )
    models["Bilinear"] = {
        "model": bilinear,
        "type": "single_stage",
        "sigmoid_scale": float(getattr(bilinear_cfg, "sigmoid_scale", 2.0)),
    }

    transformer, transformer_cfg = load_single_stage_baseline(
        model_name="transformer",
        ckpt_path=args.transformer_ckpt,
        device=device,
    )
    models["Transformer"] = {
        "model": transformer,
        "type": "single_stage",
        "sigmoid_scale": float(getattr(transformer_cfg, "sigmoid_scale", 2.0)),
    }

    prototype, prototype_cfg = load_single_stage_baseline(
        model_name="prototype",
        ckpt_path=args.prototype_ckpt,
        device=device,
    )
    models["Prototype"] = {
        "model": prototype,
        "type": "single_stage",
        "sigmoid_scale": float(getattr(prototype_cfg, "sigmoid_scale", 2.0)),
    }

    proto_res, proto_res_cfg = load_proto_res(
        ckpt_path=args.proto_res_ckpt,
        device=device,
    )
    models["Prototype + residual"] = {
        "model": proto_res,
        "type": "proto_res",
        "sigmoid_scale": float(getattr(proto_res_cfg, "sigmoid_scale", 2.0)),
    }

    pigram_model, pigram_args, pigram_ckpt, fine_rule_map = load_pigram_from_checkpoint(
        checkpoint_path=args.pigram_ckpt,
        eval_args=args,
        device=device,
    )

    models["PIGRAM grammar H_A"] = {
        "model": pigram_model,
        "type": "pigram",
        "component": "grammar",
        "sigmoid_scale": float(pigram_args.sigmoid_scale),
        "fine_rule_map": fine_rule_map,
    }
    models["PIGRAM final S_final"] = {
        "model": pigram_model,
        "type": "pigram",
        "component": "final",
        "sigmoid_scale": float(pigram_args.sigmoid_scale),
        "fine_rule_map": fine_rule_map,
    }

    return models


def score_model_on_sample(
    model_info: Dict[str, Any],
    sample: Dict[str, Any],
    device: torch.device,
    k_values: List[int],
) -> Dict[str, Any]:
    model_type = model_info["type"]

    if model_type == "single_stage":
        logits, aux = score_matrix_single_stage(
            model=model_info["model"],
            sample=sample,
            device=device,
        )
    elif model_type == "proto_res":
        logits, aux = score_matrix_proto_res(
            model=model_info["model"],
            sample=sample,
            device=device,
        )
    elif model_type == "pigram":
        logits, aux = score_matrix_pigram(
            model=model_info["model"],
            sample=sample,
            device=device,
            sigmoid_scale=model_info["sigmoid_scale"],
            component=model_info["component"],
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    readouts = score_candidate_readouts(
        logits=logits,
        sigmoid_scale=float(model_info["sigmoid_scale"]),
        k_values=k_values,
    )
    readouts.update(aux)
    return readouts


def load_downstream_samples(
    processed_dir: Path,
    targets: List[str],
    splits: List[str],
) -> List[Dict[str, Any]]:
    rows = []

    for target in targets:
        for split in splits:
            pkl_path = find_processed_pkl(processed_dir, target, split)
            samples = load_pickle(pkl_path)
            print(f"[Load] {target}/{split}: {pkl_path} | n={len(samples)}")

            for sample_index, sample in enumerate(samples):
                rows.append(
                    {
                        "target": target,
                        "split": split,
                        "sample_index": sample_index,
                        "processed_pkl": str(pkl_path),
                        "sample": sample,
                    }
                )

    return rows


def score_all_models(
    models: Dict[str, Dict[str, Any]],
    sample_rows: List[Dict[str, Any]],
    device: torch.device,
    k_values: List[int],
) -> pd.DataFrame:
    rows = []

    for model_name, model_info in models.items():
        print("\n" + "=" * 80)
        print(f"Scoring model: {model_name}")
        print("=" * 80)

        for idx, item in enumerate(sample_rows):
            sample = item["sample"]

            scores = score_model_on_sample(
                model_info=model_info,
                sample=sample,
                device=device,
                k_values=k_values,
            )

            row = {
                "model": model_name,
                "target": item["target"],
                "split": item["split"],
                "sample_index": item["sample_index"],
                "processed_pkl": item["processed_pkl"],
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

            if (idx + 1) % 100 == 0:
                print(f"[Score] {model_name}: {idx + 1}/{len(sample_rows)}")

    return pd.DataFrame(rows)


def make_selected_score_cols(k_values: List[int]) -> List[str]:
    cols = ["mean_prob", "mean_logit", "max_prob", "max_logit"]
    for k in k_values:
        cols.extend([f"top{k}_prob", f"top{k}_logit"])
    return cols


def select_best_per_model(
    metrics: pd.DataFrame,
    selection_metric: str,
    weights: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()

    for (model, target), sub in metrics.groupby(["model", "target"]):
        val = sub[
            (sub["split"] == "val")
            & (sub["label_mode"].isin(["label_low_topN", "label_high_topN"]))
        ].copy()

        if val.empty:
            continue

        def score(row):
            if selection_metric == "composite":
                return (
                    weights["prauc_weight"] * row["prauc"]
                    + weights["top10_weight"] * row["top10_precision"]
                    + weights["top50_weight"] * row["top50_precision"]
                )
            return row[selection_metric]

        val["selection_value"] = val.apply(score, axis=1)
        val = val.sort_values(
            ["selection_value", "top50_precision", "top10_precision", "auroc"],
            ascending=[False, False, False, False],
        )
        best = val.iloc[0]

        rows.append(
            {
                "model": model,
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
            }
        )

    return pd.DataFrame(rows)


def extract_metric_row(
    metrics: pd.DataFrame,
    model: str,
    target: str,
    split: str,
    label_mode: str,
    score_name: str,
) -> pd.Series | None:
    sub = metrics[
        (metrics["model"] == model)
        & (metrics["target"] == target)
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
            model=selection["model"],
            target=selection["target"],
            split="test",
            label_mode=selection["label_mode"],
            score_name=selection["selected_score_name"],
        )

        if metric_row is None:
            continue

        rows.append(
            {
                "model": selection["model"],
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

    macro_rows = []
    for model, sub in out.groupby("model"):
        macro = {
            "model": model,
            "target": "MACRO",
            "direction": "",
            "selected_score_name": "",
            "val_selection_value": float(sub["val_selection_value"].mean()),
            "n_candidates": int(sub["n_candidates"].sum()),
            "n_positive": int(sub["n_positive"].sum()),
        }
        for col in metric_cols:
            macro[col] = float(sub[col].mean())
        macro_rows.append(macro)

    if macro_rows:
        out = pd.concat([out, pd.DataFrame(macro_rows)], ignore_index=True)

    return out


def run_downstream_baseline_eval(cfg: Dict[str, Any]) -> None:
    args = cfg_to_args(cfg)
    device = torch.device(
        args.device
        if torch.cuda.is_available() and str(args.device).lower() == "cuda"
        else "cpu"
    )
    print(f"Using device: {device}")

    targets = parse_targets(cfg.get("eval", {}).get("targets", None))
    splits = parse_splits(cfg.get("eval", {}).get("splits", ["val", "test"]))
    k_values = parse_int_list(cfg.get("eval", {}).get("k_values", [1, 3, 5, 10, 20, 50]))

    print("[Targets]", targets)
    print("[Splits]", splits)
    print("[K values]", k_values)

    processed_dir = Path(args.processed_downstream_dir)
    out_dir = Path(args.baseline_eval_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = load_all_models(args=args, device=device)

    sample_rows = load_downstream_samples(
        processed_dir=processed_dir,
        targets=targets,
        splits=splits,
    )

    scores_df = score_all_models(
        models=models,
        sample_rows=sample_rows,
        device=device,
        k_values=k_values,
    )

    expected_counts = expected_counts_from_config(cfg)
    scores_df = add_faformer_labels(scores_df, expected_counts)

    score_cols = make_selected_score_cols(k_values)
    metrics_df = compute_full_metrics(scores_df, score_cols)

    selection_cfg = cfg.get("selection", {})
    weights = {
        "prauc_weight": float(selection_cfg.get("prauc_weight", 0.4)),
        "top10_weight": float(selection_cfg.get("top10_weight", 0.3)),
        "top50_weight": float(selection_cfg.get("top50_weight", 0.3)),
    }
    selection_metric = str(cfg.get("eval", {}).get("selection_metric", "composite"))

    selected_df = select_best_per_model(
        metrics=metrics_df,
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
        "targets": targets,
        "splits": splits,
        "k_values": k_values,
        "selection_metric": selection_metric,
        "selection_weights": weights,
        "models": list(models.keys()),
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

    print("\nSaved downstream baseline comparison:")
    print(scores_csv)
    print(metrics_csv)
    print(selected_csv)
    print(selected_test_csv)
    print(summary_json)

    print("\n=== Selected test macro results ===")
    if not selected_test_df.empty:
        macro = selected_test_df[selected_test_df["target"] == "MACRO"]
        print(macro.to_string(index=False))
    else:
        print("No selected test results.")