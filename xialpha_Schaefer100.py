from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from neuromaps.datasets import fetch_atlas
from scipy.spatial import cKDTree


RESULT_ROOT = Path(r"E:\b2f10k\result")
META_JSON_PATH = Path(r"E:\xialphanet_newresults22\XIALPHANET.json")
OUTPUT_ROOT = Path(r"E:\b2f10k\xialpha_parcellate")

FEATURE_SPECS = [
    ("xi", "Power", "Xi_estimate_Power"),
    ("xi", "Width", "Xi_estimate_Width"),
    ("xi", "Exponent", "Xi_estimate_Exponent"),
    ("alpha", "Power", "Alpha_estimate_Power"),
    ("alpha", "Width", "Alpha_estimate_Width"),
    ("alpha", "Exponent", "Alpha_estimate_Exponent"),
    ("alpha", "PAF", "Alpha_estimate_PAF"),
]
ANCHOR_THRESHOLD = 0.05


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

    if sphere_left_164k.shape[0] != left_labels_164k.shape[0]:
        raise ValueError("Left fsaverage annot length does not match 164k sphere.")
    if sphere_right_164k.shape[0] != right_labels_164k.shape[0]:
        raise ValueError("Right fsaverage annot length does not match 164k sphere.")

    left_tree = cKDTree(sphere_left_164k)
    right_tree = cKDTree(sphere_right_164k)
    _, left_idx = left_tree.query(sphere_left_10k, k=1)
    _, right_idx = right_tree.query(sphere_right_10k, k=1)
    return left_labels_164k[left_idx].astype(np.int32), right_labels_164k[right_idx].astype(np.int32)


def load_age_map(meta_json_path: Path) -> dict[str, float]:
    if not meta_json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_json_path}")
    data = json.loads(meta_json_path.read_text(encoding="utf-8"))
    participants = data.get("Participants")
    if not isinstance(participants, list):
        raise ValueError("Invalid metadata JSON: missing Participants list")

    age_map: dict[str, float] = {}
    for item in participants:
        subid = str(item.get("SubID", "")).strip()
        if not subid:
            continue
        try:
            age = float(item.get("Age"))
        except (TypeError, ValueError):
            continue
        age_map[subid] = age
    return age_map


def load_gii_values(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.darrays[0].data, dtype=np.float64).ravel()


def normalize_unit_interval(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Cannot normalize non-finite values")
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def signed_log1p(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values))


def parcellate_schaefer100(
    left_values: np.ndarray,
    right_values: np.ndarray,
    left_labels_10k: np.ndarray,
    right_labels_10k: np.ndarray,
) -> np.ndarray:
    if left_values.shape[0] != left_labels_10k.shape[0]:
        raise ValueError(
            f"Left size mismatch: values={left_values.shape[0]}, labels={left_labels_10k.shape[0]}"
        )
    if right_values.shape[0] != right_labels_10k.shape[0]:
        raise ValueError(
            f"Right size mismatch: values={right_values.shape[0]}, labels={right_labels_10k.shape[0]}"
        )

    out = np.zeros((100,), dtype=np.float64)
    for pid in range(1, 51):
        mask = left_labels_10k == pid
        out[pid - 1] = float(np.nanmean(left_values[mask])) if np.any(mask) else np.nan
    for pid in range(1, 51):
        mask = right_labels_10k == pid
        out[50 + pid - 1] = float(np.nanmean(right_values[mask])) if np.any(mask) else np.nan
    return out


def feature_file_paths(subject_dir: Path, source_kind: str, feature_name: str) -> tuple[Path, Path]:
    source_name = "Xi_estimate" if source_kind == "xi" else "Alpha_estimate"
    maps_dir = subject_dir / source_name / feature_name / "maps"
    left_path = maps_dir / f"{feature_name}_space-fsaverage10k_hemi-L.shape.gii"
    right_path = maps_dir / f"{feature_name}_space-fsaverage10k_hemi-R.shape.gii"
    if not left_path.exists() or not right_path.exists():
        raise FileNotFoundError(f"Missing hemi files under {maps_dir}")
    return left_path, right_path


def load_subject_surface_maps(subject_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    surface_maps: dict[str, dict[str, np.ndarray]] = {}
    for source_kind, feature_name, feature_key in FEATURE_SPECS:
        left_path, right_path = feature_file_paths(subject_dir, source_kind, feature_name)
        surface_maps[feature_key] = {
            "L": load_gii_values(left_path),
            "R": load_gii_values(right_path),
        }
    return surface_maps


def apply_xialpha_surface_rules(
    surface_maps: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    transformed = {
        feature_name: {"L": hemi_maps["L"].copy(), "R": hemi_maps["R"].copy()}
        for feature_name, hemi_maps in surface_maps.items()
    }
    aa_name = "Alpha_estimate_Power"
    xa_name = "Xi_estimate_Power"

    aa_raw = np.concatenate([surface_maps[aa_name]["L"], surface_maps[aa_name]["R"]])
    xa_raw = np.concatenate([surface_maps[xa_name]["L"], surface_maps[xa_name]["R"]])
    aa_anchor = normalize_unit_interval(aa_raw)
    aa_anchor[aa_anchor < ANCHOR_THRESHOLD] = 0.0
    xa_mask = np.where(xa_raw > 0.0, 1.0, 0.0)

    for feature_name in transformed:
        joined = np.concatenate([transformed[feature_name]["L"], transformed[feature_name]["R"]])
        joined = signed_log1p(joined)
        split = transformed[feature_name]["L"].shape[0]
        transformed[feature_name]["L"] = joined[:split]
        transformed[feature_name]["R"] = joined[split:]

    alpha_family = [
        "Alpha_estimate_Power",
        "Alpha_estimate_Width",
        "Alpha_estimate_Exponent",
        "Alpha_estimate_PAF",
    ]
    xi_family = [
        "Xi_estimate_Power",
        "Xi_estimate_Width",
        "Xi_estimate_Exponent",
    ]

    for feature_name in alpha_family:
        joined = np.concatenate([transformed[feature_name]["L"], transformed[feature_name]["R"]])
        joined = joined * aa_anchor
        split = transformed[feature_name]["L"].shape[0]
        transformed[feature_name]["L"] = joined[:split]
        transformed[feature_name]["R"] = joined[split:]

    for feature_name in xi_family:
        joined = np.concatenate([transformed[feature_name]["L"], transformed[feature_name]["R"]])
        joined = joined * xa_mask
        split = transformed[feature_name]["L"].shape[0]
        transformed[feature_name]["L"] = joined[:split]
        transformed[feature_name]["R"] = joined[split:]

    return transformed


def write_feature_csv(
    csv_path: Path,
    rows: list[list[object]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["subject_name", "age"] + [f"region{i}" for i in range(1, 101)]
    with csv_path.open("w", encoding="utf-8", newline="") as f_obj:
        writer = csv.writer(f_obj)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    if not RESULT_ROOT.exists():
        raise FileNotFoundError(f"Result root not found: {RESULT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    age_map = load_age_map(META_JSON_PATH)
    left_labels_10k, right_labels_10k = build_fsaverage10k_schaefer100_labels()

    feature_rows: dict[str, list[list[object]]] = {feature_key: [] for _, _, feature_key in FEATURE_SPECS}
    feature_subjects: dict[str, list[str]] = {feature_key: [] for _, _, feature_key in FEATURE_SPECS}
    errors: list[dict[str, str]] = []

    subject_dirs = [p for p in sorted(RESULT_ROOT.iterdir(), key=lambda p: p.name) if p.is_dir()]
    total = len(subject_dirs)
    print(f"[INFO] Subjects to process: {total}", flush=True)

    skipped_no_age = 0
    processed_subjects = 0

    for idx, subject_dir in enumerate(subject_dirs, start=1):
        subject = subject_dir.name
        age = age_map.get(subject)
        if age is None:
            skipped_no_age += 1
            errors.append({"subject": subject, "error": "missing age metadata"})
            continue

        print(f"[PROCESSING] ({idx}/{total}) {subject}", flush=True)
        try:
            raw_surface_maps = load_subject_surface_maps(subject_dir)
            transformed_maps = apply_xialpha_surface_rules(raw_surface_maps)

            for _, _, feature_key in FEATURE_SPECS:
                left_values = transformed_maps[feature_key]["L"]
                right_values = transformed_maps[feature_key]["R"]
                parcel_values = parcellate_schaefer100(
                    left_values,
                    right_values,
                    left_labels_10k,
                    right_labels_10k,
                )
                row = [subject, age] + parcel_values.astype(float).tolist()
                feature_rows[feature_key].append(row)
                feature_subjects[feature_key].append(subject)

            processed_subjects += 1
            print(f"[DONE] ({idx}/{total}) {subject}", flush=True)
        except Exception as exc:
            errors.append({"subject": subject, "error": str(exc)})
            print(f"[ERROR] ({idx}/{total}) {subject}: {exc}", flush=True)

    run_summary: dict[str, object] = {
        "result_root": str(RESULT_ROOT),
        "meta_json_path": str(META_JSON_PATH),
        "output_root": str(OUTPUT_ROOT),
        "anchor_threshold": ANCHOR_THRESHOLD,
        "applied_processing": [
            "load raw fsaverage10k left/right .shape.gii maps for 7 Xi/Alpha parameters",
            "signed_log1p transform on all parameters",
            "alpha-family weighting by thresholded normalized raw Alpha_estimate_Power anchor",
            "xi-family masking by raw Xi_estimate_Power > 0",
            "Schaefer100 parcellation on fsaverage10k",
        ],
        "processed_subject_count": processed_subjects,
        "skipped_no_age": skipped_no_age,
        "error_count": len(errors),
        "feature_outputs": {},
    }

    for source_kind, feature_name, feature_key in FEATURE_SPECS:
        feature_dir = OUTPUT_ROOT / source_kind / feature_name
        csv_path = feature_dir / f"{feature_key}_schaefer100.csv"
        rows = feature_rows[feature_key]
        write_feature_csv(csv_path, rows)

        feature_summary = {
            "feature_key": feature_key,
            "source_kind": source_kind,
            "feature_name": feature_name,
            "csv_path": str(csv_path),
            "subject_count": len(rows),
            "subjects": feature_subjects[feature_key],
            "columns": ["subject_name", "age"] + [f"region{i}" for i in range(1, 101)],
            "processing": {
                "signed_log1p": True,
                "alpha_anchor_weighting": source_kind == "alpha",
                "xi_positive_mask": source_kind == "xi",
                "schaefer100_parcellation": True,
            },
        }
        (feature_dir / "summary.json").write_text(
            json.dumps(feature_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        run_summary["feature_outputs"][feature_key] = feature_summary

    if errors:
        (OUTPUT_ROOT / "errors.json").write_text(
            json.dumps(errors, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (OUTPUT_ROOT / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Finished: {OUTPUT_ROOT / 'run_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
