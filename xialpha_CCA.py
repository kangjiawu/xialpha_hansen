from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from sklearn.cross_decomposition import CCA


REGRESSION_ROOT = Path(r"E:\b2f10k\xialpha_regression")
REFERENCE_DIR = Path(r"E:\b2f10k\hansen_reference")
OUTPUT_ROOT = Path(r"E:\b2f10k\xialpha_CCA")
FIGURE_ROOT = Path(r"E:\b2f10k\figure\CCA")

FEATURE_ORDER = [
    ("xi", "Power", "Xi_estimate_Power"),
    ("xi", "Width", "Xi_estimate_Width"),
    ("xi", "Exponent", "Xi_estimate_Exponent"),
    ("alpha", "Power", "Alpha_estimate_Power"),
    ("alpha", "Width", "Alpha_estimate_Width"),
    ("alpha", "Exponent", "Alpha_estimate_Exponent"),
    ("alpha", "PAF", "Alpha_estimate_PAF"),
]

ANALYSIS_SPECS = [
    {
        "analysis_type": "linear",
        "method": "GLM",
        "input_root": REGRESSION_ROOT / "linear",
        "map_name": "beta_age_map.csv",
        "effect_definition": "linear age coefficient per region",
    },
    {
        "analysis_type": "nonlinear",
        "method": "quadratic",
        "input_root": REGRESSION_ROOT / "nonlinear" / "quadratic",
        "map_name": "delta_pred_map.csv",
        "effect_definition": "predicted value at max(age) minus predicted value at min(age) per region",
    },
    {
        "analysis_type": "nonlinear",
        "method": "cubic",
        "input_root": REGRESSION_ROOT / "nonlinear" / "cubic",
        "map_name": "delta_pred_map.csv",
        "effect_definition": "predicted value at max(age) minus predicted value at min(age) per region",
    },
    {
        "analysis_type": "nonlinear",
        "method": "spline_df4",
        "input_root": REGRESSION_ROOT / "nonlinear" / "spline_df4",
        "map_name": "delta_pred_map.csv",
        "effect_definition": "predicted value at max(age) minus predicted value at min(age) per region",
    },
]


def zscore_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = np.nanmean(matrix, axis=0, keepdims=True)
    std = np.nanstd(matrix, axis=0, ddof=0, keepdims=True)
    std = np.where(np.logical_and(np.isfinite(std), std > 0.0), std, 1.0)
    return (matrix - mean) / std


def corr_columns(a_mat: np.ndarray, b_mat: np.ndarray) -> np.ndarray:
    out = np.zeros((a_mat.shape[1], b_mat.shape[1]), dtype=np.float64)
    for i in range(a_mat.shape[1]):
        a_col = a_mat[:, i]
        for j in range(b_mat.shape[1]):
            b_col = b_mat[:, j]
            a_std = float(np.std(a_col))
            b_std = float(np.std(b_col))
            if a_std <= 0.0 or b_std <= 0.0:
                out[i, j] = 0.0
            else:
                out[i, j] = float(np.corrcoef(a_col, b_col)[0, 1])
    return out


def write_matrix_csv(path: Path, row_names: list[str], col_names: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f_obj:
        writer = csv.writer(f_obj)
        writer.writerow(["name"] + col_names)
        for row_name, row in zip(row_names, matrix, strict=True):
            writer.writerow([row_name] + [float(v) for v in row])


def write_vector_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f_obj:
        writer = csv.DictWriter(f_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_receptor_reference(reference_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    receptor_names = np.load(reference_dir / "receptor_names_pet.npy", allow_pickle=False).tolist()
    receptor_data = np.loadtxt(reference_dir / "receptor_data_scale100.csv", delimiter=",", dtype=np.float64)
    colourmap = np.loadtxt(reference_dir / "colourmap.csv", delimiter=",", dtype=np.float64)
    return receptor_names, receptor_data, colourmap


def load_single_map(csv_path: Path) -> np.ndarray:
    with csv_path.open("r", encoding="utf-8", newline="") as f_obj:
        reader = csv.reader(f_obj)
        header = next(reader)
        rows = list(reader)

    if len(rows) != 1:
        raise ValueError(f"Expected one row in {csv_path}, got {len(rows)}")
    values = np.asarray([float(v) for v in rows[0]], dtype=np.float64)
    if len(header) != 100 or values.shape[0] != 100:
        raise ValueError(f"Expected 100 regions in {csv_path}")
    return values


def build_xialpha_matrix(input_root: Path, map_name: str) -> tuple[list[str], np.ndarray, list[str]]:
    feature_names: list[str] = []
    display_rows: list[str] = []
    vectors: list[np.ndarray] = []

    for source_kind, feature_name, feature_key in FEATURE_ORDER:
        csv_path = input_root / source_kind / feature_name / map_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing regression map: {csv_path}")
        vectors.append(load_single_map(csv_path))
        feature_names.append(feature_key)
        display_rows.append(f"{source_kind}:{feature_name}")

    return feature_names, np.column_stack(vectors), display_rows


def fit_cca(x_mat: np.ndarray, y_mat: np.ndarray) -> dict[str, np.ndarray]:
    n_components = min(x_mat.shape[1], y_mat.shape[1], x_mat.shape[0])
    x_z = zscore_columns(x_mat)
    y_z = zscore_columns(y_mat)

    cca = CCA(n_components=n_components, max_iter=2000)
    x_scores, y_scores = cca.fit_transform(x_z, y_z)

    canonical_corrs = np.zeros((n_components,), dtype=np.float64)
    for idx in range(n_components):
        canonical_corrs[idx] = float(np.corrcoef(x_scores[:, idx], y_scores[:, idx])[0, 1])

    x_loadings = corr_columns(x_z, x_scores)
    y_loadings = corr_columns(y_z, y_scores)

    return {
        "x_z": x_z,
        "y_z": y_z,
        "x_scores": x_scores,
        "y_scores": y_scores,
        "x_weights": np.asarray(cca.x_weights_, dtype=np.float64),
        "y_weights": np.asarray(cca.y_weights_, dtype=np.float64),
        "x_loadings": x_loadings,
        "y_loadings": y_loadings,
        "canonical_correlations": canonical_corrs,
    }


def plot_heatmap(
    ax,
    data: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    *,
    title: str,
    cmap,
    symmetric: bool,
) -> None:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    elif symmetric:
        vmax = float(np.nanpercentile(np.abs(finite), 99.0))
        vmax = max(vmax, 1e-12)
        vmin = -vmax
    else:
        vmin = float(np.nanpercentile(finite, 1.0))
        vmax = float(np.nanpercentile(finite, 99.0))
        if vmax <= vmin:
            vmax = vmin + 1e-12

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def plot_cca_figure(
    *,
    figure_path: Path,
    analysis_label: str,
    canonical_corrs: np.ndarray,
    x_loadings: np.ndarray,
    y_loadings: np.ndarray,
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    feature_labels: list[str],
    receptor_labels: list[str],
    colourmap: np.ndarray,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    comp_labels = [f"C{i}" for i in range(1, canonical_corrs.shape[0] + 1)]
    receptor_cmap = colors.ListedColormap(colourmap[128:, :])
    diverging_cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(14.5, 7.0))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[0.95, 1.1, 1.9],
        height_ratios=[1.0, 1.0],
        wspace=0.45,
        hspace=0.45,
    )

    ax_bar = fig.add_subplot(grid[0, 0])
    bar_colors = receptor_cmap(np.linspace(0.15, 0.95, canonical_corrs.shape[0]))
    ax_bar.bar(np.arange(canonical_corrs.shape[0]), canonical_corrs, color=bar_colors, width=0.72)
    ax_bar.set_title("Canonical Correlations", fontsize=10)
    ax_bar.set_xticks(np.arange(canonical_corrs.shape[0]))
    ax_bar.set_xticklabels(comp_labels, fontsize=8)
    ax_bar.set_ylabel("r", fontsize=9)
    ax_bar.set_ylim(0.0, max(1.0, float(np.nanmax(canonical_corrs) * 1.10)))
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    ax_scatter = fig.add_subplot(grid[1, 0])
    ax_scatter.scatter(
        x_scores[:, 0],
        y_scores[:, 0],
        s=24,
        color=bar_colors[0],
        alpha=0.75,
        edgecolors="none",
    )
    line_coef = np.polyfit(x_scores[:, 0], y_scores[:, 0], deg=1)
    x_line = np.linspace(float(np.min(x_scores[:, 0])), float(np.max(x_scores[:, 0])), 100)
    y_line = np.polyval(line_coef, x_line)
    ax_scatter.plot(x_line, y_line, color="black", linewidth=1.2)
    ax_scatter.set_title("CCA Component 1 Scores", fontsize=10)
    ax_scatter.set_xlabel("Xi-Alpha score", fontsize=9)
    ax_scatter.set_ylabel("Receptor score", fontsize=9)
    ax_scatter.text(
        0.03,
        0.95,
        f"r = {canonical_corrs[0]:.3f}",
        transform=ax_scatter.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)

    ax_x = fig.add_subplot(grid[:, 1])
    im_x = plot_heatmap(
        ax_x,
        x_loadings,
        comp_labels,
        feature_labels,
        title="Xi-Alpha Loadings",
        cmap=diverging_cmap,
        symmetric=True,
    )
    cbar_x = fig.colorbar(im_x, ax=ax_x, fraction=0.046, pad=0.03)
    cbar_x.ax.tick_params(labelsize=7)

    ax_y = fig.add_subplot(grid[:, 2])
    im_y = plot_heatmap(
        ax_y,
        y_loadings,
        comp_labels,
        receptor_labels,
        title="Receptor Loadings",
        cmap=diverging_cmap,
        symmetric=True,
    )
    cbar_y = fig.colorbar(im_y, ax=ax_y, fraction=0.022, pad=0.02)
    cbar_y.ax.tick_params(labelsize=7)

    fig.suptitle(analysis_label, fontsize=12, y=0.98)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_component_scores(path: Path, x_scores: np.ndarray, y_scores: np.ndarray) -> None:
    fieldnames = ["region_index"] + [f"x_score_C{i}" for i in range(1, x_scores.shape[1] + 1)] + [
        f"y_score_C{i}" for i in range(1, y_scores.shape[1] + 1)
    ]
    rows: list[dict[str, object]] = []
    for idx in range(x_scores.shape[0]):
        row: dict[str, object] = {"region_index": idx + 1}
        for comp_idx in range(x_scores.shape[1]):
            row[f"x_score_C{comp_idx + 1}"] = float(x_scores[idx, comp_idx])
            row[f"y_score_C{comp_idx + 1}"] = float(y_scores[idx, comp_idx])
        rows.append(row)
    write_vector_csv(path, fieldnames, rows)


def run_one_analysis(
    *,
    analysis_type: str,
    method: str,
    input_root: Path,
    map_name: str,
    effect_definition: str,
    receptor_names: list[str],
    receptor_data: np.ndarray,
    colourmap: np.ndarray,
) -> dict[str, object]:
    print(f"[PROCESSING] {analysis_type} | {method}", flush=True)
    feature_names, x_matrix, feature_labels = build_xialpha_matrix(input_root, map_name)
    results = fit_cca(x_matrix, receptor_data)

    out_dir = OUTPUT_ROOT / analysis_type / method
    fig_dir = FIGURE_ROOT / analysis_type / method
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    component_labels = [f"C{i}" for i in range(1, results["canonical_correlations"].shape[0] + 1)]
    region_labels = [f"region{i}" for i in range(1, x_matrix.shape[0] + 1)]

    np.savetxt(out_dir / "xialpha_age_effect_matrix.csv", x_matrix, delimiter=",", header=",".join(feature_names), comments="")
    np.savetxt(out_dir / "receptor_matrix.csv", receptor_data, delimiter=",", header=",".join(receptor_names), comments="")

    canonical_rows = [
        {
            "component": component_labels[idx],
            "canonical_correlation": float(results["canonical_correlations"][idx]),
            "shared_variance_r2": float(results["canonical_correlations"][idx] ** 2),
        }
        for idx in range(results["canonical_correlations"].shape[0])
    ]
    write_vector_csv(
        out_dir / "canonical_correlations.csv",
        ["component", "canonical_correlation", "shared_variance_r2"],
        canonical_rows,
    )
    write_matrix_csv(out_dir / "x_weights.csv", feature_names, component_labels, results["x_weights"])
    write_matrix_csv(out_dir / "y_weights.csv", receptor_names, component_labels, results["y_weights"])
    write_matrix_csv(out_dir / "x_loadings.csv", feature_names, component_labels, results["x_loadings"])
    write_matrix_csv(out_dir / "y_loadings.csv", receptor_names, component_labels, results["y_loadings"])
    write_matrix_csv(out_dir / "x_scores.csv", region_labels, component_labels, results["x_scores"])
    write_matrix_csv(out_dir / "y_scores.csv", region_labels, component_labels, results["y_scores"])
    save_component_scores(out_dir / "component_scores_by_region.csv", results["x_scores"], results["y_scores"])

    overview_png = fig_dir / "cca_overview.png"
    plot_cca_figure(
        figure_path=overview_png,
        analysis_label=f"{analysis_type.upper()} | {method}",
        canonical_corrs=results["canonical_correlations"],
        x_loadings=results["x_loadings"],
        y_loadings=results["y_loadings"],
        x_scores=results["x_scores"],
        y_scores=results["y_scores"],
        feature_labels=feature_names,
        receptor_labels=receptor_names,
        colourmap=colourmap,
    )

    summary = {
        "analysis_type": analysis_type,
        "method": method,
        "input_root": str(input_root),
        "input_map_name": map_name,
        "effect_definition": effect_definition,
        "xialpha_feature_order": feature_names,
        "receptor_names": receptor_names,
        "region_count": int(x_matrix.shape[0]),
        "xialpha_feature_count": int(x_matrix.shape[1]),
        "receptor_count": int(receptor_data.shape[1]),
        "canonical_correlations": [float(v) for v in results["canonical_correlations"]],
        "outputs": {
            "xialpha_age_effect_matrix": str(out_dir / "xialpha_age_effect_matrix.csv"),
            "receptor_matrix": str(out_dir / "receptor_matrix.csv"),
            "canonical_correlations": str(out_dir / "canonical_correlations.csv"),
            "x_weights": str(out_dir / "x_weights.csv"),
            "y_weights": str(out_dir / "y_weights.csv"),
            "x_loadings": str(out_dir / "x_loadings.csv"),
            "y_loadings": str(out_dir / "y_loadings.csv"),
            "x_scores": str(out_dir / "x_scores.csv"),
            "y_scores": str(out_dir / "y_scores.csv"),
            "component_scores_by_region": str(out_dir / "component_scores_by_region.csv"),
            "figure_overview": str(overview_png),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] {analysis_type} | {method}", flush=True)
    return summary


def main() -> None:
    if not REGRESSION_ROOT.exists():
        raise FileNotFoundError(f"Regression root not found: {REGRESSION_ROOT}")
    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(f"Reference dir not found: {REFERENCE_DIR}")

    receptor_names, receptor_data, colourmap = load_receptor_reference(REFERENCE_DIR)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for spec in ANALYSIS_SPECS:
        try:
            summary = run_one_analysis(
                analysis_type=spec["analysis_type"],
                method=spec["method"],
                input_root=spec["input_root"],
                map_name=spec["map_name"],
                effect_definition=spec["effect_definition"],
                receptor_names=receptor_names,
                receptor_data=receptor_data,
                colourmap=colourmap,
            )
            run_summaries.append(summary)
        except Exception as exc:
            errors.append(
                {
                    "analysis_type": str(spec["analysis_type"]),
                    "method": str(spec["method"]),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[ERROR] {spec['analysis_type']} | {spec['method']}: {type(exc).__name__}: {exc}", flush=True)

    run_summary = {
        "regression_root": str(REGRESSION_ROOT),
        "reference_dir": str(REFERENCE_DIR),
        "output_root": str(OUTPUT_ROOT),
        "figure_root": str(FIGURE_ROOT),
        "analysis_count": len(run_summaries),
        "error_count": len(errors),
        "summaries": run_summaries,
        "errors": errors,
    }
    (OUTPUT_ROOT / "run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Finished. Success={len(run_summaries)} Error={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
