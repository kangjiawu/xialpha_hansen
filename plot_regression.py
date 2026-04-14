from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import cm, colors
from neuromaps.datasets import fetch_atlas
from scipy.spatial import cKDTree

from brainstorm_fsaverage_toolkit.atlas import load_fsaverage10k_atlas
from brainstorm_fsaverage_toolkit.config import DEFAULT_NEUROMAPS_DATA_DIR
from brainstorm_fsaverage_toolkit.style import harmonize_backgrounds, plot_panel


REGRESSION_ROOT = Path(r"E:\b2f10k\xialpha_regression")
FIGURE_ROOT = Path(r"E:\b2f10k\figure")

DEFAULT_EFFECT_MAPS = (
    "beta_age_map.csv",
    "delta_pred_map.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot regression age-effect maps on fsaverage10k and save mirrored figures."
    )
    parser.add_argument(
        "--regression-root",
        type=Path,
        default=REGRESSION_ROOT,
        help="Root folder that stores xialpha_regression outputs.",
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=FIGURE_ROOT,
        help="Root folder used to save generated PNG figures.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional single *_map.csv to plot. If omitted, the script scans the regression root.",
    )
    parser.add_argument(
        "--all-maps",
        action="store_true",
        help="Plot every *_map.csv under the regression root. Default only plots age-effect maps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files.",
    )
    return parser.parse_args()


def find_schaefer_annot_paths() -> tuple[Path, Path]:
    candidates = [
        Path(r"C:\Users\Administrator\nnt-data\atl-schaefer2018\fsaverage"),
        Path(r"C:\Users\Administrator\nilearn_data\schafer_surface"),
    ]
    left_name = "atl-Schaefer2018_space-fsaverage_hemi-L_desc-100Parcels7Networks_deterministic.annot"
    right_name = "atl-Schaefer2018_space-fsaverage_hemi-R_desc-100Parcels7Networks_deterministic.annot"

    for root in candidates:
        left_path = root / left_name
        right_path = root / right_name
        if left_path.exists() and right_path.exists():
            return left_path, right_path

    raise FileNotFoundError(f"Cannot find Schaefer100 annot files. Tried: {candidates}")


def load_surf_coords(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.darrays[0].data, dtype=np.float64)


def build_fsaverage10k_schaefer100_labels() -> tuple[np.ndarray, np.ndarray]:
    left_annot, right_annot = find_schaefer_annot_paths()
    left_labels_164k, _, _ = nib.freesurfer.read_annot(str(left_annot))
    right_labels_164k, _, _ = nib.freesurfer.read_annot(str(right_annot))

    atlas_data_candidates = [
        Path(r"C:\Users\Administrator\neuromaps-data"),
        Path(r"E:\matlab_to_py_maps\matlab_to_py_maps\.cache\neuromaps-data"),
    ]
    atlas_data_dir = next((p for p in atlas_data_candidates if p.exists()), None)
    if atlas_data_dir is None:
        raise FileNotFoundError(f"Cannot find neuromaps atlas dir. Tried: {atlas_data_candidates}")

    fsavg_164k = fetch_atlas("fsaverage", density="164k", data_dir=atlas_data_dir)
    fsavg_10k = fetch_atlas("fsaverage", density="10k", data_dir=atlas_data_dir)

    sphere_left_164k = load_surf_coords(Path(fsavg_164k["sphere"].L))
    sphere_right_164k = load_surf_coords(Path(fsavg_164k["sphere"].R))
    sphere_left_10k = load_surf_coords(Path(fsavg_10k["sphere"].L))
    sphere_right_10k = load_surf_coords(Path(fsavg_10k["sphere"].R))

    left_tree = cKDTree(sphere_left_164k)
    right_tree = cKDTree(sphere_right_164k)
    _, left_idx = left_tree.query(sphere_left_10k, k=1)
    _, right_idx = right_tree.query(sphere_right_10k, k=1)

    return left_labels_164k[left_idx].astype(np.int32), right_labels_164k[right_idx].astype(np.int32)


def load_region_map(csv_path: Path) -> tuple[list[str], np.ndarray]:
    with csv_path.open("r", encoding="utf-8", newline="") as f_obj:
        reader = csv.reader(f_obj)
        header = next(reader)
        rows = list(reader)

    if not header:
        raise ValueError(f"Empty header: {csv_path}")
    if len(rows) != 1:
        raise ValueError(f"Expected one data row in {csv_path}, got {len(rows)}")

    values = np.asarray([float(v) for v in rows[0]], dtype=np.float64)
    if values.shape[0] != 100:
        raise ValueError(f"Expected 100 parcel values in {csv_path}, got {values.shape[0]}")
    return header, values


def parcels_to_surface(
    values: np.ndarray,
    left_labels_10k: np.ndarray,
    right_labels_10k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_values = np.full(left_labels_10k.shape, np.nan, dtype=np.float64)
    right_values = np.full(right_labels_10k.shape, np.nan, dtype=np.float64)

    for pid in range(1, 51):
        left_values[left_labels_10k == pid] = float(values[pid - 1])
        right_values[right_labels_10k == pid] = float(values[50 + pid - 1])

    return left_values, right_values


def compute_plot_limits(all_values: np.ndarray, signed: bool) -> tuple[float, float]:
    finite = np.asarray(all_values[np.isfinite(all_values)], dtype=np.float64)
    if finite.size == 0:
        return 0.0, 1.0

    if signed:
        vmax = float(np.nanpercentile(np.abs(finite), 99.0))
        vmax = max(vmax, 1e-12)
        return -vmax, vmax

    vmin = float(np.nanpercentile(finite, 1.0))
    vmax = float(np.nanpercentile(finite, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return vmin, vmax


def infer_signed_map(csv_path: Path, values: np.ndarray) -> bool:
    name = csv_path.name.lower()
    if "beta" in name or "delta" in name or "t_" in name:
        return True
    finite = values[np.isfinite(values)]
    return bool(finite.size and np.nanmin(finite) < 0.0 and np.nanmax(finite) > 0.0)


def build_output_path(csv_path: Path, regression_root: Path, figure_root: Path) -> Path:
    relative = csv_path.relative_to(regression_root)
    return figure_root / relative.with_suffix(".png")


def build_title(csv_path: Path, regression_root: Path) -> str:
    relative = csv_path.relative_to(regression_root)
    parts = relative.parts[:-1]
    map_name = relative.stem
    return " | ".join(list(parts) + [map_name])


def plot_surface_map(
    *,
    left_surface: np.ndarray,
    right_surface: np.ndarray,
    out_png: Path,
    title: str,
    signed: bool,
    atlas,
) -> dict[str, float | bool]:
    all_values = np.concatenate([left_surface, right_surface])
    vmin, vmax = compute_plot_limits(all_values, signed=signed)
    cmap = plt.get_cmap("RdBu_r" if signed else "inferno")

    bg = harmonize_backgrounds(
        atlas.hemispheres["L"].background,
        atlas.hemispheres["R"].background,
        atlas.hemispheres["L"].background,
        atlas.hemispheres["R"].background,
        human_lighten=0.0,
        macaque_lighten=0.0,
        bg_mid_gray=0.70,
        bg_contrast=0.38,
        bg_gamma=0.42,
    )
    bg_l, bg_r = bg[0], bg[1]

    fig = plt.figure(figsize=(11.0, 3.3))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.0)
    axes = [fig.add_subplot(grid[0, i], projection="3d") for i in range(4)]
    colorbar_ax = fig.add_subplot(grid[0, 4])

    plot_panel(
        axes[0],
        (atlas.hemispheres["L"].pial_vertices, atlas.hemispheres["L"].pial_faces),
        left_surface,
        bg_l,
        "left",
        "left",
        cmap,
        vmin,
        vmax,
    )
    plot_panel(
        axes[1],
        (atlas.hemispheres["L"].pial_vertices, atlas.hemispheres["L"].pial_faces),
        left_surface,
        bg_l,
        "medial_left",
        "left",
        cmap,
        vmin,
        vmax,
    )
    plot_panel(
        axes[2],
        (atlas.hemispheres["R"].pial_vertices, atlas.hemispheres["R"].pial_faces),
        right_surface,
        bg_r,
        "medial_right",
        "right",
        cmap,
        vmin,
        vmax,
    )
    plot_panel(
        axes[3],
        (atlas.hemispheres["R"].pial_vertices, atlas.hemispheres["R"].pial_faces),
        right_surface,
        bg_r,
        "right",
        "right",
        cmap,
        vmin,
        vmax,
    )

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(mappable, cax=colorbar_ax)
    colorbar.ax.tick_params(labelsize=7)

    fig.text(0.5, 0.98, title, ha="center", va="top", fontsize=9)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "signed_map": signed,
        "vmin": float(vmin),
        "vmax": float(vmax),
    }


def collect_csv_paths(regression_root: Path, input_csv: Path | None, all_maps: bool) -> list[Path]:
    if input_csv is not None:
        return [input_csv]

    if all_maps:
        return sorted(regression_root.rglob("*_map.csv"))

    matches: list[Path] = []
    for name in DEFAULT_EFFECT_MAPS:
        matches.extend(sorted(regression_root.rglob(name)))
    return matches


def main() -> None:
    args = parse_args()
    regression_root = args.regression_root
    figure_root = args.figure_root

    if not regression_root.exists():
        raise FileNotFoundError(f"Regression root not found: {regression_root}")

    csv_paths = collect_csv_paths(regression_root, args.input_csv, args.all_maps)
    if not csv_paths:
        print(f"No matching regression map csv files found under: {regression_root}", flush=True)
        return

    print(f"[INFO] Load atlas and Schaefer100 labels", flush=True)
    atlas = load_fsaverage10k_atlas(DEFAULT_NEUROMAPS_DATA_DIR)
    left_labels_10k, right_labels_10k = build_fsaverage10k_schaefer100_labels()

    figure_root.mkdir(parents=True, exist_ok=True)
    run_records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    total = len(csv_paths)
    for idx, csv_path in enumerate(csv_paths, start=1):
        out_png = build_output_path(csv_path, regression_root, figure_root)
        if out_png.exists() and not args.overwrite:
            print(f"[SKIP] ({idx}/{total}) {out_png}", flush=True)
            continue

        print(f"[PROCESSING] ({idx}/{total}) {csv_path}", flush=True)
        try:
            _, parcel_values = load_region_map(csv_path)
            left_surface, right_surface = parcels_to_surface(parcel_values, left_labels_10k, right_labels_10k)
            signed = infer_signed_map(csv_path, parcel_values)
            out_png.parent.mkdir(parents=True, exist_ok=True)
            plot_info = plot_surface_map(
                left_surface=left_surface,
                right_surface=right_surface,
                out_png=out_png,
                title=build_title(csv_path, regression_root),
                signed=signed,
                atlas=atlas,
            )
            run_records.append(
                {
                    "input_csv": str(csv_path),
                    "output_png": str(out_png),
                    **plot_info,
                }
            )
            print(f"[DONE] ({idx}/{total}) {out_png}", flush=True)
        except Exception as exc:
            errors.append({"input_csv": str(csv_path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[ERROR] ({idx}/{total}) {csv_path}: {type(exc).__name__}: {exc}", flush=True)

    summary = {
        "regression_root": str(regression_root),
        "figure_root": str(figure_root),
        "input_mode": "single_csv" if args.input_csv is not None else ("all_maps" if args.all_maps else "age_effect_maps"),
        "default_effect_maps": list(DEFAULT_EFFECT_MAPS),
        "plotted_count": len(run_records),
        "error_count": len(errors),
        "records": run_records,
        "errors": errors,
    }
    (figure_root / "plot_regression_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Finished. Plotted={len(run_records)} Errors={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
