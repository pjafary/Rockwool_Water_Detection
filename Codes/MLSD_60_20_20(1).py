#!/usr/bin/env python3
"""Leakage-safe Multi-Learner Spectral Drought (MLSD) stacking model.

Development mode performs nested, physical-sample-grouped base-learner tuning,
generates exactly one out-of-fold (OOF) probability vector for every training
observation, tunes a multinomial logistic-regression meta-learner using those
OOF features, and evaluates the frozen stack on the validation partition.

Final-test mode reloads the frozen choices, creates development-set OOF features
without retuning, refits the base learners on train+validation, and evaluates
the locked test once. The test set is never used for wavelength, learner,
hyperparameter, or meta-learner selection.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CSV = SCRIPT_DIR / "All_classes_smoothed.csv"
SPLIT_MANIFEST_CSV = SCRIPT_DIR / "physical_sample_split_60_20_20_seed42.csv"
OUTPUT_ROOT = SCRIPT_DIR / "results_MLSD_60_20_20_3nm_bands"
FROZEN_CONFIG_JSON = OUTPUT_ROOT / "frozen_MLSD_config.json"

# Run development first. Use final_test only after every RWI, base-learner,
# ablation, stacking, and meta-learner decision is frozen.
RUN_MODE = "final_test"  # "development" or "final_test"
ALLOW_COMPLETED_TEST_RERUN = False

RANDOM_SEED = 42
OUTER_OOF_FOLDS = 5
INNER_TUNING_FOLDS = 4
META_TUNING_FOLDS = 5
N_JOBS = -1
SCORING = "f1_macro"
BOOTSTRAP_ITERATIONS = 2000

# Final MLSD configuration described by the manuscript's ablation result.
# PCA_LDA is used instead of the technically incorrect SIMCA label.
BASE_LEARNERS = ("RF", "PLS_DA", "PCA_LDA")
META_C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)

RWI_WAVELENGTHS_NM = (560.0, 1600.0, 1920.0)
WINDOW_HALF_WIDTH_NM = 1
WAVELENGTH_TOLERANCE_NM = 0.51
DENOMINATOR_EPSILON = 1e-12

CLASS_TO_VWC = {
    "dry": 0,
    "sub9": 5,
    "sub8": 10,
    "sub7": 15,
    "sub6": 20,
    "sub5": 25,
    "sub4": 30,
    "sub3": 35,
    "sub2": 40,
    "sub1": 45,
    "sub0": 50,
}
CLASS_ORDER = list(CLASS_TO_VWC)
CLASS_TO_INTEGER = {name: i for i, name in enumerate(CLASS_ORDER)}
INTEGER_TO_CLASS = {i: name for name, i in CLASS_TO_INTEGER.items()}
N_CLASSES = len(CLASS_ORDER)
EXPECTED_SPLIT_COUNTS = {"train": 60, "validation": 20, "test": 20}

MODULE_NAME = "MLSD_60_20_20"
if __name__ == "__main__":
    sys.modules[MODULE_NAME] = sys.modules[__name__]


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """PLS-DA with one-hot targets and softmax-normalized response scores."""

    def __init__(self, n_components: int = 2, max_iter: int = 1000, tol: float = 1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSDAClassifier":
        self.classes_ = unique_labels(y)
        target = np.zeros((len(y), len(self.classes_)), dtype=float)
        lookup = {label: i for i, label in enumerate(self.classes_)}
        for row, label in enumerate(y):
            target[row, lookup[label]] = 1.0
        self.model_ = PLSRegression(
            n_components=self.n_components,
            scale=False,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.model_.fit(X, target)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict(X), dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


PLSDAClassifier.__module__ = MODULE_NAME


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_spectral_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    raw = pd.read_csv(path)
    wavelength_columns = [c for c in raw.columns if str(c).lower() == "wavelength"]
    if len(wavelength_columns) != 1:
        raise ValueError("The CSV must contain exactly one Wavelength column.")
    wavelength_column = wavelength_columns[0]
    spectrum_columns = [c for c in raw.columns if c != wavelength_column]
    pattern = re.compile(r"^(dry|sub[0-9])-(\d+)$")
    parsed: list[tuple[str, int]] = []
    invalid: list[str] = []
    for column in spectrum_columns:
        match = pattern.fullmatch(str(column))
        if match is None:
            invalid.append(str(column))
        else:
            parsed.append((match.group(1), int(match.group(2))))
    if invalid:
        raise ValueError(f"Unrecognized spectrum headers: {invalid[:10]}")

    wavelengths = pd.to_numeric(raw[wavelength_column], errors="coerce").to_numpy(float)
    matrix = raw[spectrum_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(wavelengths).all() or not np.isfinite(matrix).all():
        raise ValueError("All wavelengths and reflectance values must be finite.")
    if len(np.unique(wavelengths)) != len(wavelengths):
        raise ValueError("Wavelength values must be unique.")
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    spectra = matrix[order, :].T

    metadata = pd.DataFrame(parsed, columns=["class_name", "physical_sample_id"])
    metadata.insert(0, "spectrum_column", spectrum_columns)
    metadata["class_integer"] = metadata["class_name"].map(CLASS_TO_INTEGER)
    metadata["vwc_percent"] = metadata["class_name"].map(CLASS_TO_VWC)
    if metadata["class_integer"].isna().any():
        raise ValueError("Unexpected class label.")
    if metadata.duplicated(["physical_sample_id", "class_name"]).any():
        raise ValueError("Duplicate physical-sample/class observations found.")
    counts = pd.crosstab(metadata["physical_sample_id"], metadata["class_name"])
    counts = counts.reindex(columns=CLASS_ORDER, fill_value=0)
    if len(counts) != 100 or not (counts.to_numpy() == 1).all():
        raise ValueError(
            "Expected 100 physical samples, each containing all 11 VWC classes."
        )
    return {
        "spectra": spectra,
        "wavelengths": wavelengths,
        "metadata": metadata,
        "y": metadata["class_integer"].to_numpy(int),
        "groups": metadata["physical_sample_id"].to_numpy(int),
    }


def load_split_manifest(path: Path, physical_ids: Iterable[int]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    manifest = pd.read_csv(path)
    required = {"physical_sample_id", "split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest requires columns {sorted(required)}")
    manifest = manifest.copy()
    manifest["physical_sample_id"] = pd.to_numeric(
        manifest["physical_sample_id"], errors="raise"
    ).astype(int)
    manifest["split"] = manifest["split"].astype(str).str.lower().str.strip()
    if manifest["physical_sample_id"].duplicated().any():
        raise ValueError("Each physical sample ID must occur once in the manifest.")
    expected_ids = set(int(x) for x in physical_ids)
    observed_ids = set(manifest["physical_sample_id"])
    if expected_ids != observed_ids:
        raise ValueError("Manifest physical sample IDs do not match the data.")
    if manifest["split"].value_counts().to_dict() != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Manifest must contain {EXPECTED_SPLIT_COUNTS}.")
    return manifest.sort_values("physical_sample_id").reset_index(drop=True)


def create_masks(groups: np.ndarray, manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    split_by_id = manifest.set_index("physical_sample_id")["split"].to_dict()
    split = np.asarray([split_by_id[int(group)] for group in groups])
    masks = {name: split == name for name in EXPECTED_SPLIT_COUNTS}
    names = list(EXPECTED_SPLIT_COUNTS)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if set(groups[masks[left]]) & set(groups[masks[right]]):
                raise AssertionError(f"Physical-sample overlap: {left}/{right}")
    for name, expected in EXPECTED_SPLIT_COUNTS.items():
        if len(np.unique(groups[masks[name]])) != expected:
            raise AssertionError(f"Incorrect group count in {name}.")
    return masks


def nearest_index(wavelengths: np.ndarray, target: float) -> tuple[int, float]:
    index = int(np.argmin(np.abs(wavelengths - target)))
    actual = float(wavelengths[index])
    if abs(actual - target) > WAVELENGTH_TOLERANCE_NM:
        raise ValueError(f"Requested {target:g} nm; nearest is {actual:g} nm.")
    return index, actual


def build_features(
    spectra: np.ndarray, wavelengths: np.ndarray
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    center_matches = [nearest_index(wavelengths, value) for value in RWI_WAVELENGTHS_NM]
    indices = [item[0] for item in center_matches]
    actual_centers = [item[1] for item in center_matches]
    a, b, c = (spectra[:, index] for index in indices)
    numerator = a - (b - c)
    denominator = a + (b - c)
    denominator = np.where(
        np.abs(denominator) < DENOMINATOR_EPSILON,
        np.where(denominator < 0, -DENOMINATOR_EPSILON, DENOMINATOR_EPSILON),
        denominator,
    )
    rwi = numerator / denominator

    requested: list[float] = []
    for center in RWI_WAVELENGTHS_NM:
        requested.extend(
            float(value)
            for value in np.arange(
                center - WINDOW_HALF_WIDTH_NM,
                center + WINDOW_HALF_WIDTH_NM + 1,
                1.0,
            )
        )
    unique_indices = sorted({nearest_index(wavelengths, value)[0] for value in requested})
    window_wavelengths = [float(wavelengths[index]) for index in unique_indices]
    X = np.column_stack([rwi, spectra[:, unique_indices]])
    names = ["RWI"] + [f"R_{value:g}_nm" for value in window_wavelengths]
    if not np.isfinite(X).all():
        raise FloatingPointError("Engineered features contain non-finite values.")
    details = {
        "rwi_requested_centers_nm": list(RWI_WAVELENGTHS_NM),
        "rwi_actual_centers_nm": actual_centers,
        "window_half_width_nm": WINDOW_HALF_WIDTH_NM,
        "window_wavelengths_nm": window_wavelengths,
        "feature_count": int(X.shape[1]),
        "rwi_formula": "(Ra - (Rb - Rc)) / (Ra + (Rb - Rc))",
    }
    return X, names, details


def model_definitions() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    definitions = {
        "RF": (
            Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            criterion="gini",
                            random_state=RANDOM_SEED,
                            n_jobs=1,
                        ),
                    )
                ]
            ),
            {
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.5, 1.0],
            },
        ),
        "PLS_DA": (
            Pipeline(
                [("scale", StandardScaler()), ("model", PLSDAClassifier())]
            ),
            {"model__n_components": [2, 3, 4, 5, 6, 8, 10]},
        ),
        "PCA_LDA": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("pca", PCA(random_state=RANDOM_SEED)),
                    ("model", LinearDiscriminantAnalysis(solver="svd", tol=0.0001)),
                ]
            ),
            {"pca__n_components": [2, 3, 5, 8, 10, 15, 20, 25, 30]},
        ),
    }
    missing = set(BASE_LEARNERS) - set(definitions)
    if missing:
        raise ValueError(f"Unknown BASE_LEARNERS: {sorted(missing)}")
    return {name: definitions[name] for name in BASE_LEARNERS}


def make_grouped_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    splits = list(splitter.split(X, y, groups))
    for fold, (fit_indices, holdout_indices) in enumerate(splits, start=1):
        if set(groups[fit_indices]) & set(groups[holdout_indices]):
            raise AssertionError(f"Group overlap in fold {fold}.")
        if set(np.unique(y[holdout_indices])) != set(range(N_CLASSES)):
            raise AssertionError(f"Fold {fold} is missing at least one class.")
    return splits


def aligned_probabilities(model: Pipeline, X: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    aligned = np.zeros((len(X), N_CLASSES), dtype=float)
    for source_column, class_integer in enumerate(classes):
        aligned[:, int(class_integer)] = probabilities[:, source_column]
    row_sums = aligned.sum(axis=1, keepdims=True)
    if not np.isfinite(aligned).all() or np.any(row_sums <= 0):
        raise FloatingPointError("Invalid base-learner probabilities.")
    return aligned / row_sums


def normalize_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return dict(parameters)


def tune_one_model(
    pipeline: Pipeline,
    grid: dict[str, list[Any]],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> GridSearchCV:
    inner_splits = make_grouped_splits(
        X, y, groups, INNER_TUNING_FOLDS, seed
    )
    # PCA and PLS cannot use more latent components than there are input
    # features (and PCA is also bounded by the number of fitting rows).  The
    # engineered 3 nm-band feature set currently has only 10 columns, while
    # the broader PCA search grid also contains 15--30 components.  Filter
    # those impossible candidates instead of allowing GridSearchCV's
    # error_score="raise" to abort the entire run.
    smallest_fit_size = min(len(fit_indices) for fit_indices, _ in inner_splits)
    max_components = min(X.shape[1], smallest_fit_size)
    valid_grid = {name: list(values) for name, values in grid.items()}
    for parameter, values in valid_grid.items():
        if parameter.endswith("__n_components"):
            valid_grid[parameter] = [
                value for value in values if int(value) <= max_components
            ]
            if not valid_grid[parameter]:
                raise ValueError(
                    f"No valid values remain for {parameter}; the maximum is "
                    f"{max_components} for X with shape {X.shape}."
                )
    search = GridSearchCV(
        pipeline,
        valid_grid,
        scoring=SCORING,
        cv=inner_splits,
        refit=True,
        n_jobs=N_JOBS,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X, y, groups=groups)
    return search


def probability_column_names() -> list[str]:
    return [
        f"{model_name}__P_{class_name}"
        for model_name in BASE_LEARNERS
        for class_name in CLASS_ORDER
    ]


def save_meta_features(
    path: Path,
    meta_features: np.ndarray,
    metadata: pd.DataFrame,
    fold_assignment: np.ndarray | None = None,
) -> None:
    frame = metadata.reset_index(drop=True).copy()
    if fold_assignment is not None:
        frame["oof_fold"] = fold_assignment
    for index, name in enumerate(probability_column_names()):
        frame[name] = meta_features[:, index]
    frame.to_csv(path, index=False)


def nested_training_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    definitions = model_definitions()
    outer_splits = make_grouped_splits(
        X, y, groups, OUTER_OOF_FOLDS, RANDOM_SEED
    )
    meta_features = np.full((len(y), len(BASE_LEARNERS) * N_CLASSES), np.nan)
    fold_assignment = np.full(len(y), -1, dtype=int)
    tuning_rows: list[dict[str, Any]] = []

    for outer_fold, (fit_indices, holdout_indices) in enumerate(outer_splits, start=1):
        print(f"Nested stacking OOF fold {outer_fold}/{OUTER_OOF_FOLDS}", flush=True)
        fold_assignment[holdout_indices] = outer_fold
        for learner_index, (name, (pipeline, grid)) in enumerate(definitions.items()):
            search = tune_one_model(
                clone(pipeline),
                grid,
                X[fit_indices],
                y[fit_indices],
                groups[fit_indices],
                RANDOM_SEED + 100 * outer_fold + learner_index,
            )
            start = learner_index * N_CLASSES
            stop = start + N_CLASSES
            meta_features[holdout_indices, start:stop] = aligned_probabilities(
                search.best_estimator_, X[holdout_indices]
            )
            best_index = int(search.best_index_)
            tuning_rows.append(
                {
                    "outer_fold": outer_fold,
                    "base_learner": name,
                    "inner_cv_macro_f1_mean": float(search.best_score_),
                    "inner_cv_macro_f1_std": float(
                        search.cv_results_["std_test_score"][best_index]
                    ),
                    "best_parameters": json.dumps(json_safe(search.best_params_)),
                }
            )

    if np.isnan(meta_features).any() or np.any(fold_assignment < 1):
        raise AssertionError("Every training row must receive exactly one OOF prediction.")
    tuning_frame = pd.DataFrame(tuning_rows)
    tuning_frame.to_csv(output_dir / "nested_base_tuning_by_outer_fold.csv", index=False)
    save_meta_features(
        output_dir / "training_OOF_meta_features.csv",
        meta_features,
        metadata,
        fold_assignment,
    )
    return meta_features, fold_assignment, tuning_frame


def tune_full_training_base_models(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, Pipeline], dict[str, Any]]:
    fitted: dict[str, Pipeline] = {}
    frozen: dict[str, Any] = {}
    for learner_index, (name, (pipeline, grid)) in enumerate(model_definitions().items()):
        print(f"Tuning {name} on all 60 training cubes...", flush=True)
        search = tune_one_model(
            clone(pipeline),
            grid,
            X,
            y,
            groups,
            RANDOM_SEED + 1000 + learner_index,
        )
        results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        results.to_csv(output_dir / f"full_training_grid_{name}.csv", index=False)
        best_index = int(search.best_index_)
        fitted[name] = search.best_estimator_
        frozen[name] = {
            "best_parameters": json_safe(search.best_params_),
            "training_grouped_cv_macro_f1_mean": float(search.best_score_),
            "training_grouped_cv_macro_f1_std": float(
                search.cv_results_["std_test_score"][best_index]
            ),
        }
    return fitted, frozen


def meta_model(C: float) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=3000,
        random_state=RANDOM_SEED,
    )


def tune_meta_model(
    meta_features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    output_dir: Path,
) -> tuple[LogisticRegression, dict[str, Any]]:
    splits = make_grouped_splits(
        meta_features, y, groups, META_TUNING_FOLDS, RANDOM_SEED + 2000
    )
    search = GridSearchCV(
        meta_model(1.0),
        {"C": list(META_C_VALUES)},
        scoring=SCORING,
        cv=splits,
        refit=True,
        n_jobs=N_JOBS,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(meta_features, y, groups=groups)
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        output_dir / "meta_learner_grouped_cv_grid.csv", index=False
    )
    best_index = int(search.best_index_)
    record = {
        "C": float(search.best_params_["C"]),
        "training_OOF_grouped_cv_macro_f1_mean": float(search.best_score_),
        "training_OOF_grouped_cv_macro_f1_std": float(
            search.cv_results_["std_test_score"][best_index]
        ),
    }
    return search.best_estimator_, record


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def save_evaluation(
    output_dir: Path,
    prefix: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, float]:
    metrics = prediction_metrics(y_true, y_pred)
    frame = metadata.reset_index(drop=True).copy()
    frame["true_class"] = [INTEGER_TO_CLASS[int(value)] for value in y_true]
    frame["predicted_class"] = [INTEGER_TO_CLASS[int(value)] for value in y_pred]
    frame["correct"] = y_true == y_pred
    for class_integer, class_name in enumerate(CLASS_ORDER):
        frame[f"MLSD_probability_{class_name}"] = probabilities[:, class_integer]
    frame.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / f"{prefix}_overall_metrics.csv", index=False)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(output_dir / f"{prefix}_classification_report.csv")
    counts = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        totals,
        out=np.zeros_like(counts, dtype=float),
        where=totals != 0,
    )
    rows = [f"true_{name}" for name in CLASS_ORDER]
    columns = [f"predicted_{name}" for name in CLASS_ORDER]
    pd.DataFrame(counts, index=rows, columns=columns).to_csv(
        output_dir / f"{prefix}_confusion_counts.csv"
    )
    pd.DataFrame(normalized, index=rows, columns=columns).to_csv(
        output_dir / f"{prefix}_confusion_row_normalized.csv"
    )
    return metrics


def grouped_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    iterations: int,
) -> pd.DataFrame:
    unique_groups = np.unique(groups)
    indices_by_group = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(RANDOM_SEED + 9000)
    samples: list[dict[str, float]] = []
    for _ in range(iterations):
        sampled = rng.choice(unique_groups, len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled])
        samples.append(prediction_metrics(y_true[indices], y_pred[indices]))
    frame = pd.DataFrame(samples)
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "bootstrap_unit": "physical_sample_id",
                "iterations": iterations,
                "ci_lower_2.5_percent": float(frame[metric].quantile(0.025)),
                "ci_upper_97.5_percent": float(frame[metric].quantile(0.975)),
            }
            for metric in frame.columns
        ]
    )


def meta_features_from_fitted_models(
    fitted_models: dict[str, Pipeline], X: np.ndarray
) -> np.ndarray:
    parts = [aligned_probabilities(fitted_models[name], X) for name in BASE_LEARNERS]
    return np.column_stack(parts)


def fixed_parameter_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    metadata: pd.DataFrame,
    frozen_models: dict[str, Any],
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    definitions = model_definitions()
    splits = make_grouped_splits(X, y, groups, OUTER_OOF_FOLDS, RANDOM_SEED + 3000)
    meta_features = np.full((len(y), len(BASE_LEARNERS) * N_CLASSES), np.nan)
    fold_assignment = np.full(len(y), -1, dtype=int)
    for fold, (fit_indices, holdout_indices) in enumerate(splits, start=1):
        fold_assignment[holdout_indices] = fold
        for learner_index, name in enumerate(BASE_LEARNERS):
            pipeline = clone(definitions[name][0])
            pipeline.set_params(
                **normalize_parameters(frozen_models[name]["best_parameters"])
            )
            pipeline.fit(X[fit_indices], y[fit_indices])
            start = learner_index * N_CLASSES
            meta_features[holdout_indices, start : start + N_CLASSES] = (
                aligned_probabilities(pipeline, X[holdout_indices])
            )
    if np.isnan(meta_features).any() or np.any(fold_assignment < 1):
        raise AssertionError("Incomplete fixed-parameter OOF feature matrix.")
    save_meta_features(output_path, meta_features, metadata, fold_assignment)
    return meta_features, fold_assignment


def run_information(feature_names: list[str], feature_details: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": sha256_file(DATA_CSV),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST_CSV),
        "random_seed": RANDOM_SEED,
        "outer_oof_folds": OUTER_OOF_FOLDS,
        "inner_tuning_folds": INNER_TUNING_FOLDS,
        "meta_tuning_folds": META_TUNING_FOLDS,
        "base_learners": list(BASE_LEARNERS),
        "feature_names": feature_names,
        "feature_details": feature_details,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }


def run_development(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    metadata: pd.DataFrame,
    masks: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    feature_names: list[str],
    feature_details: dict[str, Any],
) -> None:
    completion_marker = OUTPUT_ROOT / "LOCKED_TEST_COMPLETED.json"
    if completion_marker.exists():
        raise RuntimeError("Locked test already completed; MLSD development is frozen.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / f"development_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    X_train, y_train = X[masks["train"]], y[masks["train"]]
    groups_train = groups[masks["train"]]
    train_metadata = metadata.loc[masks["train"]]
    X_validation, y_validation = X[masks["validation"]], y[masks["validation"]]
    validation_metadata = metadata.loc[masks["validation"]]

    oof_features, _, _ = nested_training_oof(
        X_train, y_train, groups_train, train_metadata, output_dir
    )
    fitted_base, frozen_base = tune_full_training_base_models(
        X_train, y_train, groups_train, output_dir
    )
    fitted_meta, frozen_meta = tune_meta_model(
        oof_features, y_train, groups_train, output_dir
    )

    validation_meta_features = meta_features_from_fitted_models(
        fitted_base, X_validation
    )
    save_meta_features(
        output_dir / "validation_meta_features.csv",
        validation_meta_features,
        validation_metadata,
    )
    validation_probabilities = fitted_meta.predict_proba(validation_meta_features)
    validation_prediction = fitted_meta.predict(validation_meta_features)
    validation_metrics = save_evaluation(
        output_dir,
        "validation_MLSD",
        y_validation,
        validation_prediction,
        validation_probabilities,
        validation_metadata,
    )

    manifest.to_csv(output_dir / "split_manifest_used.csv", index=False)
    frozen = {
        **run_information(feature_names, feature_details),
        "status": "development_complete_test_not_evaluated",
        "base_models": frozen_base,
        "meta_model": frozen_meta,
        "validation_metrics": validation_metrics,
        "stacking_description": {
            "training_meta_features": "Nested grouped out-of-fold probabilities",
            "probabilities_per_base_learner": N_CLASSES,
            "meta_feature_count": len(BASE_LEARNERS) * N_CLASSES,
            "meta_classifier": "multinomial logistic regression",
        },
    }
    FROZEN_CONFIG_JSON.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    (output_dir / "frozen_MLSD_config.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )
    print("\nMLSD development completed without evaluating the test partition.")
    print("Validation macro-F1:", f"{validation_metrics['macro_f1']:.4f}")
    print("Frozen configuration:", FROZEN_CONFIG_JSON)


def run_final_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    metadata: pd.DataFrame,
    masks: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    feature_names: list[str],
    feature_details: dict[str, Any],
) -> None:
    if not FROZEN_CONFIG_JSON.exists():
        raise FileNotFoundError("Run MLSD development before final_test.")
    frozen = json.loads(FROZEN_CONFIG_JSON.read_text(encoding="utf-8"))
    if frozen.get("status") != "development_complete_test_not_evaluated":
        raise ValueError("Unexpected frozen MLSD status.")
    if frozen.get("data_sha256") != sha256_file(DATA_CSV):
        raise ValueError("Data file differs from MLSD development.")
    if frozen.get("split_manifest_sha256") != sha256_file(SPLIT_MANIFEST_CSV):
        raise ValueError("Split manifest differs from MLSD development.")
    if frozen.get("base_learners") != list(BASE_LEARNERS):
        raise ValueError("Base learners differ from the frozen configuration.")
    if frozen.get("feature_names") != feature_names:
        raise ValueError("Feature columns differ from MLSD development.")
    if frozen.get("feature_details") != json_safe(feature_details):
        raise ValueError("Feature settings differ from MLSD development.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    completion_marker = OUTPUT_ROOT / "LOCKED_TEST_COMPLETED.json"
    if completion_marker.exists() and not ALLOW_COMPLETED_TEST_RERUN:
        raise RuntimeError("Locked MLSD test has already been completed.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / f"locked_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    development_mask = masks["train"] | masks["validation"]
    X_development, y_development = X[development_mask], y[development_mask]
    groups_development = groups[development_mask]
    development_metadata = metadata.loc[development_mask]
    X_test, y_test = X[masks["test"]], y[masks["test"]]
    groups_test = groups[masks["test"]]
    test_metadata = metadata.loc[masks["test"]]

    development_oof, _ = fixed_parameter_oof(
        X_development,
        y_development,
        groups_development,
        development_metadata,
        frozen["base_models"],
        output_dir / "development_OOF_meta_features.csv",
    )
    final_meta = meta_model(float(frozen["meta_model"]["C"]))
    final_meta.fit(development_oof, y_development)

    definitions = model_definitions()
    final_base: dict[str, Pipeline] = {}
    for name in BASE_LEARNERS:
        pipeline = clone(definitions[name][0])
        pipeline.set_params(
            **normalize_parameters(frozen["base_models"][name]["best_parameters"])
        )
        pipeline.fit(X_development, y_development)
        final_base[name] = pipeline
        joblib.dump(pipeline, output_dir / f"fitted_base_{name}.joblib")
    joblib.dump(final_meta, output_dir / "fitted_meta_logistic_regression.joblib")

    test_meta_features = meta_features_from_fitted_models(final_base, X_test)
    save_meta_features(
        output_dir / "locked_test_meta_features.csv", test_meta_features, test_metadata
    )
    test_probabilities = final_meta.predict_proba(test_meta_features)
    test_prediction = final_meta.predict(test_meta_features)
    test_metrics = save_evaluation(
        output_dir,
        "locked_test_MLSD",
        y_test,
        test_prediction,
        test_probabilities,
        test_metadata,
    )
    grouped_bootstrap_ci(
        y_test, test_prediction, groups_test, BOOTSTRAP_ITERATIONS
    ).to_csv(output_dir / "locked_test_group_bootstrap_95ci.csv", index=False)
    manifest.to_csv(output_dir / "split_manifest_used.csv", index=False)

    run_record = {
        **run_information(feature_names, feature_details),
        "status": "locked_test_completed",
        "frozen_config_sha256": sha256_file(FROZEN_CONFIG_JSON),
        "test_metrics": test_metrics,
        "test_evaluated_after_all_choices_were_frozen": True,
        "output_directory": str(output_dir.resolve()),
    }
    (output_dir / "locked_test_run_record.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    completion_marker.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print("\nMLSD locked test completed once using frozen settings.")
    print(json.dumps(test_metrics, indent=2))
    print("Results:", output_dir)


def main() -> None:
    if RUN_MODE not in {"development", "final_test"}:
        raise ValueError("RUN_MODE must be development or final_test.")
    loaded = load_spectral_data(DATA_CSV)
    metadata = loaded["metadata"]
    y = loaded["y"]
    groups = loaded["groups"]
    manifest = load_split_manifest(SPLIT_MANIFEST_CSV, np.unique(groups))
    masks = create_masks(groups, manifest)
    X, feature_names, feature_details = build_features(
        loaded["spectra"], loaded["wavelengths"]
    )
    print(
        f"Loaded {len(y)} spectra, {len(np.unique(groups))} physical samples, "
        f"and {X.shape[1]} MLSD input features."
    )
    print(
        "Split: "
        + ", ".join(
            f"{name}={len(np.unique(groups[mask]))} cubes/{int(mask.sum())} spectra"
            for name, mask in masks.items()
        )
    )
    if RUN_MODE == "development":
        run_development(
            X, y, groups, metadata, masks, manifest, feature_names, feature_details
        )
    else:
        run_final_test(
            X, y, groups, metadata, masks, manifest, feature_names, feature_details
        )


if __name__ == "__main__":
    main()
