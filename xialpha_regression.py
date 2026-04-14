from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from patsy import dmatrix


INPUT_ROOT = Path(r"E:\b2f10k\xialpha_parcellate")
OUTPUT_ROOT = Path(r"E:\b2f10k\xialpha_regression")

FEATURE_SPECS = [
    ("xi", "Power", "Xi_estimate_Power"),
    ("xi", "Width", "Xi_estimate_Width"),
    ("xi", "Exponent", "Xi_estimate_Exponent"),
    ("alpha", "Power", "Alpha_estimate_Power"),
    ("alpha", "Width", "Alpha_estimate_Width"),
    ("alpha", "Exponent", "Alpha_estimate_Exponent"),
    ("alpha", "PAF", "Alpha_estimate_PAF"),
]

NONLINEAR_METHODS = ("quadratic", "cubic", "spline_df4")


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(pvals)
    ranked = pvals[order]
    n_tests = len(pvals)
    adjusted = ranked * n_tests / np.arange(1, n_tests + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def load_feature_csv(csv_path: Path) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing feature csv: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f_obj:
        reader = csv.reader(f_obj)
        header = next(reader)
        rows = list(reader)

    if len(header) != 102:
        raise ValueError(f"Unexpected header length in {csv_path}: {len(header)}")
    if not rows:
        raise ValueError(f"No data rows in {csv_path}")

    subject_names = [row[0] for row in rows]
    ages = np.asarray([float(row[1]) for row in rows], dtype=np.float64)
    region_names = header[2:]
    values = np.asarray([[float(v) for v in row[2:]] for row in rows], dtype=np.float64)
    return subject_names, ages, region_names, values


def write_map_csv(path: Path, region_names: list[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f_obj:
        writer = csv.writer(f_obj)
        writer.writerow(region_names)
        writer.writerow([float(v) for v in values])


def write_predictions_csv(path: Path, age_grid: np.ndarray, region_names: list[str], predictions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f_obj:
        writer = csv.writer(f_obj)
        writer.writerow(["age"] + region_names)
        for idx, age in enumerate(age_grid):
            writer.writerow([float(age)] + [float(v) for v in predictions[idx, :]])


def write_region_stats_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f_obj:
        writer = csv.DictWriter(f_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def linear_regression(ages: np.ndarray, values: np.ndarray) -> dict[str, np.ndarray | list[dict[str, object]]]:
    n_regions = values.shape[1]
    age_grid = np.linspace(float(np.min(ages)), float(np.max(ages)), 101)
    predictions = np.zeros((age_grid.shape[0], n_regions), dtype=np.float64)

    beta_age = np.zeros((n_regions,), dtype=np.float64)
    intercept = np.zeros((n_regions,), dtype=np.float64)
    t_age = np.zeros((n_regions,), dtype=np.float64)
    p_age = np.zeros((n_regions,), dtype=np.float64)
    r2 = np.zeros((n_regions,), dtype=np.float64)
    adj_r2 = np.zeros((n_regions,), dtype=np.float64)
    stats_rows: list[dict[str, object]] = []

    X = sm.add_constant(ages)
    X_pred = sm.add_constant(age_grid)

    for region_idx in range(n_regions):
        y = values[:, region_idx]
        fit = sm.OLS(y, X).fit()
        intercept[region_idx] = float(fit.params[0])
        beta_age[region_idx] = float(fit.params[1])
        t_age[region_idx] = float(fit.tvalues[1])
        p_age[region_idx] = float(fit.pvalues[1])
        r2[region_idx] = float(fit.rsquared)
        adj_r2[region_idx] = float(fit.rsquared_adj)
        predictions[:, region_idx] = fit.predict(X_pred)

    q_age = fdr_bh(p_age)

    for region_idx in range(n_regions):
        stats_rows.append(
            {
                "region": region_idx + 1,
                "intercept": intercept[region_idx],
                "beta_age": beta_age[region_idx],
                "t_age": t_age[region_idx],
                "p_age": p_age[region_idx],
                "q_age": q_age[region_idx],
                "r2": r2[region_idx],
                "adj_r2": adj_r2[region_idx],
            }
        )

    return {
        "age_grid": age_grid,
        "predictions": predictions,
        "intercept": intercept,
        "beta_age": beta_age,
        "t_age": t_age,
        "p_age": p_age,
        "q_age": q_age,
        "r2": r2,
        "adj_r2": adj_r2,
        "stats_rows": stats_rows,
    }


def build_nonlinear_design(method: str, ages: np.ndarray) -> np.ndarray:
    if method == "quadratic":
        return np.column_stack([np.ones_like(ages), ages, ages**2])
    if method == "cubic":
        return np.column_stack([np.ones_like(ages), ages, ages**2, ages**3])
    if method == "spline_df4":
        spline = dmatrix(
            "bs(age, df=4, degree=3, include_intercept=False)",
            {"age": ages},
            return_type="dataframe",
        )
        return np.column_stack([np.ones_like(ages), np.asarray(spline, dtype=np.float64)])
    raise ValueError(f"Unsupported nonlinear method: {method}")


def nonlinear_regression(method: str, ages: np.ndarray, values: np.ndarray) -> dict[str, np.ndarray | list[dict[str, object]]]:
    n_regions = values.shape[1]
    age_grid = np.linspace(float(np.min(ages)), float(np.max(ages)), 101)
    predictions = np.zeros((age_grid.shape[0], n_regions), dtype=np.float64)

    model_p = np.zeros((n_regions,), dtype=np.float64)
    r2 = np.zeros((n_regions,), dtype=np.float64)
    adj_r2 = np.zeros((n_regions,), dtype=np.float64)
    delta_pred = np.zeros((n_regions,), dtype=np.float64)
    min_pred = np.zeros((n_regions,), dtype=np.float64)
    max_pred = np.zeros((n_regions,), dtype=np.float64)
    age_at_min = np.zeros((n_regions,), dtype=np.float64)
    age_at_max = np.zeros((n_regions,), dtype=np.float64)
    stats_rows: list[dict[str, object]] = []

    X = build_nonlinear_design(method, ages)
    X_pred = build_nonlinear_design(method, age_grid)

    for region_idx in range(n_regions):
        y = values[:, region_idx]
        fit = sm.OLS(y, X).fit()
        pred = fit.predict(X_pred)
        predictions[:, region_idx] = pred
        model_p[region_idx] = float(fit.f_pvalue)
        r2[region_idx] = float(fit.rsquared)
        adj_r2[region_idx] = float(fit.rsquared_adj)
        delta_pred[region_idx] = float(pred[-1] - pred[0])

        min_idx = int(np.argmin(pred))
        max_idx = int(np.argmax(pred))
        min_pred[region_idx] = float(pred[min_idx])
        max_pred[region_idx] = float(pred[max_idx])
        age_at_min[region_idx] = float(age_grid[min_idx])
        age_at_max[region_idx] = float(age_grid[max_idx])

    q_model = fdr_bh(model_p)

    for region_idx in range(n_regions):
        stats_rows.append(
            {
                "region": region_idx + 1,
                "model_p": model_p[region_idx],
                "model_q": q_model[region_idx],
                "r2": r2[region_idx],
                "adj_r2": adj_r2[region_idx],
                "delta_pred_max_age_minus_min_age": delta_pred[region_idx],
                "min_pred": min_pred[region_idx],
                "max_pred": max_pred[region_idx],
                "age_at_min_pred": age_at_min[region_idx],
                "age_at_max_pred": age_at_max[region_idx],
            }
        )

    return {
        "age_grid": age_grid,
        "predictions": predictions,
        "model_p": model_p,
        "model_q": q_model,
        "r2": r2,
        "adj_r2": adj_r2,
        "delta_pred": delta_pred,
        "min_pred": min_pred,
        "max_pred": max_pred,
        "age_at_min": age_at_min,
        "age_at_max": age_at_max,
        "stats_rows": stats_rows,
    }


def save_linear_outputs(
    out_dir: Path,
    region_names: list[str],
    result: dict[str, np.ndarray | list[dict[str, object]]],
    feature_key: str,
    source_kind: str,
    feature_name: str,
    subject_count: int,
    ages: np.ndarray,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    write_map_csv(out_dir / "beta_age_map.csv", region_names, np.asarray(result["beta_age"]))
    write_map_csv(out_dir / "t_age_map.csv", region_names, np.asarray(result["t_age"]))
    write_map_csv(out_dir / "p_age_map.csv", region_names, np.asarray(result["p_age"]))
    write_map_csv(out_dir / "q_age_map.csv", region_names, np.asarray(result["q_age"]))
    write_map_csv(out_dir / "r2_map.csv", region_names, np.asarray(result["r2"]))
    write_map_csv(out_dir / "adj_r2_map.csv", region_names, np.asarray(result["adj_r2"]))
    write_predictions_csv(
        out_dir / "predictions_age_grid.csv",
        np.asarray(result["age_grid"]),
        region_names,
        np.asarray(result["predictions"]),
    )
    write_region_stats_csv(
        out_dir / "region_stats.csv",
        ["region", "intercept", "beta_age", "t_age", "p_age", "q_age", "r2", "adj_r2"],
        list(result["stats_rows"]),
    )

    summary = {
        "model_type": "linear",
        "implementation": "Gaussian linear model via OLS",
        "feature_key": feature_key,
        "source_kind": source_kind,
        "feature_name": feature_name,
        "subject_count": subject_count,
        "age_min": float(np.min(ages)),
        "age_max": float(np.max(ages)),
        "age_effect_map_definition": "beta_age_map.csv stores the linear age coefficient for each region",
        "outputs": {
            "beta_age_map": str(out_dir / "beta_age_map.csv"),
            "t_age_map": str(out_dir / "t_age_map.csv"),
            "p_age_map": str(out_dir / "p_age_map.csv"),
            "q_age_map": str(out_dir / "q_age_map.csv"),
            "r2_map": str(out_dir / "r2_map.csv"),
            "adj_r2_map": str(out_dir / "adj_r2_map.csv"),
            "predictions_age_grid": str(out_dir / "predictions_age_grid.csv"),
            "region_stats": str(out_dir / "region_stats.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def save_nonlinear_outputs(
    out_dir: Path,
    method: str,
    region_names: list[str],
    result: dict[str, np.ndarray | list[dict[str, object]]],
    feature_key: str,
    source_kind: str,
    feature_name: str,
    subject_count: int,
    ages: np.ndarray,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    write_map_csv(out_dir / "delta_pred_map.csv", region_names, np.asarray(result["delta_pred"]))
    write_map_csv(out_dir / "model_p_map.csv", region_names, np.asarray(result["model_p"]))
    write_map_csv(out_dir / "model_q_map.csv", region_names, np.asarray(result["model_q"]))
    write_map_csv(out_dir / "r2_map.csv", region_names, np.asarray(result["r2"]))
    write_map_csv(out_dir / "adj_r2_map.csv", region_names, np.asarray(result["adj_r2"]))
    write_map_csv(out_dir / "age_at_min_pred_map.csv", region_names, np.asarray(result["age_at_min"]))
    write_map_csv(out_dir / "age_at_max_pred_map.csv", region_names, np.asarray(result["age_at_max"]))
    write_predictions_csv(
        out_dir / "predictions_age_grid.csv",
        np.asarray(result["age_grid"]),
        region_names,
        np.asarray(result["predictions"]),
    )
    write_region_stats_csv(
        out_dir / "region_stats.csv",
        [
            "region",
            "model_p",
            "model_q",
            "r2",
            "adj_r2",
            "delta_pred_max_age_minus_min_age",
            "min_pred",
            "max_pred",
            "age_at_min_pred",
            "age_at_max_pred",
        ],
        list(result["stats_rows"]),
    )

    if method == "quadratic":
        model_description = "y ~ age + age^2"
    elif method == "cubic":
        model_description = "y ~ age + age^2 + age^3"
    else:
        model_description = "y ~ cubic spline(age, df=4)"

    summary = {
        "model_type": "nonlinear",
        "method": method,
        "model_description": model_description,
        "feature_key": feature_key,
        "source_kind": source_kind,
        "feature_name": feature_name,
        "subject_count": subject_count,
        "age_min": float(np.min(ages)),
        "age_max": float(np.max(ages)),
        "age_effect_map_definition": "delta_pred_map.csv stores predicted value at max(age) minus predicted value at min(age) for each region",
        "outputs": {
            "delta_pred_map": str(out_dir / "delta_pred_map.csv"),
            "model_p_map": str(out_dir / "model_p_map.csv"),
            "model_q_map": str(out_dir / "model_q_map.csv"),
            "r2_map": str(out_dir / "r2_map.csv"),
            "adj_r2_map": str(out_dir / "adj_r2_map.csv"),
            "age_at_min_pred_map": str(out_dir / "age_at_min_pred_map.csv"),
            "age_at_max_pred_map": str(out_dir / "age_at_max_pred_map.csv"),
            "predictions_age_grid": str(out_dir / "predictions_age_grid.csv"),
            "region_stats": str(out_dir / "region_stats.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def feature_csv_path(source_kind: str, feature_name: str, feature_key: str) -> Path:
    return INPUT_ROOT / source_kind / feature_name / f"{feature_key}_schaefer100.csv"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_summary: dict[str, object] = {
        "input_root": str(INPUT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "linear_model": "Gaussian linear model via OLS",
        "nonlinear_models": {
            "quadratic": "y ~ age + age^2",
            "cubic": "y ~ age + age^2 + age^3",
            "spline_df4": "y ~ cubic spline(age, df=4)",
        },
        "features": {},
        "errors": [],
    }

    for source_kind, feature_name, feature_key in FEATURE_SPECS:
        csv_path = feature_csv_path(source_kind, feature_name, feature_key)
        print(f"[PROCESSING] {feature_key}", flush=True)
        try:
            subject_names, ages, region_names, values = load_feature_csv(csv_path)
            linear_result = linear_regression(ages, values)
            linear_dir = OUTPUT_ROOT / "linear" / source_kind / feature_name
            nonlinear_root = OUTPUT_ROOT / "nonlinear"

            feature_summary = {
                "input_csv": str(csv_path),
                "subject_count": len(subject_names),
                "region_count": len(region_names),
                "age_min": float(np.min(ages)),
                "age_max": float(np.max(ages)),
                "linear": save_linear_outputs(
                    linear_dir,
                    region_names,
                    linear_result,
                    feature_key,
                    source_kind,
                    feature_name,
                    len(subject_names),
                    ages,
                ),
                "nonlinear": {},
            }

            for method in NONLINEAR_METHODS:
                print(f"[NONLINEAR] {feature_key} {method}", flush=True)
                nonlinear_result = nonlinear_regression(method, ages, values)
                nonlinear_dir = nonlinear_root / method / source_kind / feature_name
                feature_summary["nonlinear"][method] = save_nonlinear_outputs(
                    nonlinear_dir,
                    method,
                    region_names,
                    nonlinear_result,
                    feature_key,
                    source_kind,
                    feature_name,
                    len(subject_names),
                    ages,
                )

            (OUTPUT_ROOT / "feature_summaries").mkdir(parents=True, exist_ok=True)
            (OUTPUT_ROOT / "feature_summaries" / f"{feature_key}.json").write_text(
                json.dumps(feature_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            run_summary["features"][feature_key] = feature_summary
            print(f"[DONE] {feature_key}", flush=True)
        except Exception as exc:
            err = {"feature_key": feature_key, "error": str(exc), "input_csv": str(csv_path)}
            run_summary["errors"].append(err)
            print(f"[ERROR] {feature_key}: {exc}", flush=True)

    (OUTPUT_ROOT / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Finished: {OUTPUT_ROOT / 'run_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
