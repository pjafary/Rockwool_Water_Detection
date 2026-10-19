#!/usr/bin/env python3
"""MLSD spectral-preprocessing sensitivity analysis.

This development-only analysis compares preprocessing strategies using the
fixed 60-cube training and 20-cube validation partitions. The locked 20-cube
test partition is verified but never fitted, transformed, predicted, or scored.

Each preprocessing method is placed inside every base-learner pipeline. Thus,
MSC reference spectra, scaling, PCA, and classifier parameters are learned only
from the applicable training fold. The MLSD stack uses nested group-aware OOF
probabilities and a multinomial logistic-regression meta-learner.

IMPORTANT: SOURCE_DATA_CSV must contain the UNSMOOTHED spectra. Supplying the
already moving-average-smoothed file would compare additional transformations
after smoothing rather than compare alternative preprocessing strategies.
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

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
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

# Point this to the original combined UNSMOOTHED spectra file.
SOURCE_DATA_CSV = SCRIPT_DIR / "All_classes_raw.csv"
SPLIT_MANIFEST_CSV = SCRIPT_DIR / "physical_sample_split_60_20_20_seed42.csv"
OUTPUT_ROOT = SCRIPT_DIR / "results_MLSD_preprocessing_sensitivity"

# This safeguard must remain False for the reviewer-facing comparison. Change
# it only for a clearly labelled diagnostic run on already smoothed data.
INPUT_IS_ALREADY_MOVING_AVERAGED = False
ALLOW_ALREADY_PREPROCESSED_INPUT = False

RANDOM_SEED = 42
OUTER_OOF_FOLDS = 5
INNER_TUNING_FOLDS = 4
META_TUNING_FOLDS = 5
N_JOBS = -1
SCORING = "f1_macro"
BOOTSTRAP_ITERATIONS = 2000

# Updated leakage-safe wavelength result and the user's reduced ±1 nm windows.
RWI_WAVELENGTHS_NM = (560.0, 1600.0, 1920.0)
WINDOW_HALF_WIDTH_NM = 1
WAVELENGTH_TOLERANCE_NM = 0.51
DENOMINATOR_EPSILON = 1e-12

# Manuscript baseline: i=1 gives a three-point moving-average window.
MOVING_AVERAGE_HALF_WIDTH = 1
SAVGOL_WINDOW_LENGTH = 3
SAVGOL_POLYORDER = 2

PREPROCESSING_METHODS = (
    "none",
    "moving_average",
    "snv",
    "msc",
    "savgol_smoothing",
    "savgol_first_derivative",
    "savgol_second_derivative",
)

BASE_LEARNERS = ("RF", "PLS_DA", "PCA_LDA")
META_C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)

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

MODULE_NAME = "MLSD_preprocessing_sensitivity_60_20"
if __name__ == "__main__":
    sys.modules[MODULE_NAME] = sys.modules[__name__]


class SpectralPreprocessor(BaseEstimator, TransformerMixin):
    """Fold-safe spectral preprocessing, including train-fitted MSC."""

    def __init__(
        self,
        method: str,
        moving_average_half_width: int = MOVING_AVERAGE_HALF_WIDTH,
        savgol_window_length: int = SAVGOL_WINDOW_LENGTH,
        savgol_polyorder: int = SAVGOL_POLYORDER,
        wavelength_step_nm: float = 1.0,
        epsilon: float = 1e-12,
    ):
        self.method = method
        self.moving_average_half_width = moving_average_half_width
        self.savgol_window_length = savgol_window_length
        self.savgol_polyorder = savgol_polyorder
        self.wavelength_step_nm = wavelength_step_nm
        self.epsilon = epsilon

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SpectralPreprocessor":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Spectral matrix must be two-dimensional.")
        if self.method not in PREPROCESSING_METHODS:
            raise ValueError(f"Unsupported preprocessing method: {self.method}")
        if self.method == "msc":
            self.reference_ = X.mean(axis=0)
            centered = self.reference_ - self.reference_.mean()
            self.reference_centered_ = centered
            self.reference_sum_squares_ = float(np.dot(centered, centered))
            if self.reference_sum_squares_ < self.epsilon:
                raise ValueError("MSC reference spectrum has near-zero variance.")
        self.n_features_in_ = X.shape[1]
        return self

    def _moving_average(self, X: np.ndarray) -> np.ndarray:
        half_width = int(self.moving_average_half_width)
        if half_width < 0:
            raise ValueError("moving_average_half_width must be non-negative.")
        if half_width == 0:
            return X.copy()
        window = 2 * half_width + 1
        padded = np.pad(X, ((0, 0), (half_width, half_width)), mode="edge")
        cumulative = np.cumsum(padded, axis=1, dtype=float)
        cumulative = np.pad(cumulative, ((0, 0), (1, 0)), mode="constant")
        return (cumulative[:, window:] - cumulative[:, :-window]) / window

    def _snv(self, X: np.ndarray) -> np.ndarray:
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True, ddof=1)
        std = np.where(std < self.epsilon, 1.0, std)
        return (X - mean) / std

    def _msc(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "reference_")
        sample_mean = X.mean(axis=1, keepdims=True)
        sample_centered = X - sample_mean
        slopes = (
            sample_centered @ self.reference_centered_
        ) / self.reference_sum_squares_
        slopes = np.where(
            np.abs(slopes) < self.epsilon,
            np.where(slopes < 0, -self.epsilon, self.epsilon),
            slopes,
        )
        intercepts = sample_mean.ravel() - slopes * self.reference_.mean()
        return (X - intercepts[:, None]) / slopes[:, None]

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Spectrum length differs from the fitted data.")
        if self.method == "none":
            result = X.copy()
        elif self.method == "moving_average":
            result = self._moving_average(X)
        elif self.method == "snv":
            result = self._snv(X)
        elif self.method == "msc":
            result = self._msc(X)
        elif self.method == "savgol_smoothing":
            result = savgol_filter(
                X,
                window_length=self.savgol_window_length,
                polyorder=self.savgol_polyorder,
                deriv=0,
                delta=self.wavelength_step_nm,
                axis=1,
                mode="interp",
            )
        elif self.method == "savgol_first_derivative":
            result = savgol_filter(
                X,
                window_length=self.savgol_window_length,
                polyorder=self.savgol_polyorder,
                deriv=1,
                delta=self.wavelength_step_nm,
                axis=1,
                mode="interp",
            )
        elif self.method == "savgol_second_derivative":
            result = savgol_filter(
                X,
                window_length=self.savgol_window_length,
                polyorder=self.savgol_polyorder,
                deriv=2,
                delta=self.wavelength_step_nm,
                axis=1,
                mode="interp",
            )
        else:
            raise ValueError(f"Unsupported preprocessing method: {self.method}")
        if not np.isfinite(result).all():
            raise FloatingPointError(f"{self.method} produced non-finite spectra.")
        return result


class RWIWindowFeatures(BaseEstimator, TransformerMixin):
    """Calculate RWI and reflectance values within ±1 nm of each RWI band."""

    def __init__(
        self,
        wavelengths: tuple[float, ...],
        centers_nm: tuple[float, float, float] = RWI_WAVELENGTHS_NM,
        window_half_width_nm: int = WINDOW_HALF_WIDTH_NM,
        tolerance_nm: float = WAVELENGTH_TOLERANCE_NM,
        denominator_epsilon: float = DENOMINATOR_EPSILON,
    ):
        self.wavelengths = wavelengths
        self.centers_nm = centers_nm
        self.window_half_width_nm = window_half_width_nm
        self.tolerance_nm = tolerance_nm
        self.denominator_epsilon = denominator_epsilon

    def _nearest_index(self, target: float) -> tuple[int, float]:
        wavelengths = np.asarray(self.wavelengths, dtype=float)
        index = int(np.argmin(np.abs(wavelengths - target)))
        actual = float(wavelengths[index])
        if abs(actual - target) > self.tolerance_nm:
            raise ValueError(f"Requested {target:g} nm; nearest is {actual:g} nm.")
        return index, actual

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RWIWindowFeatures":
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != len(self.wavelengths):
            raise ValueError("Spectral matrix/wavelength dimensions do not match.")
        center_matches = [self._nearest_index(value) for value in self.centers_nm]
        self.center_indices_ = [item[0] for item in center_matches]
        self.actual_centers_nm_ = [item[1] for item in center_matches]
        requested: list[float] = []
        for center in self.centers_nm:
            requested.extend(
                float(value)
                for value in np.arange(
                    center - self.window_half_width_nm,
                    center + self.window_half_width_nm + 1,
                    1.0,
                )
            )
        self.window_indices_ = sorted(
            {self._nearest_index(value)[0] for value in requested}
        )
        wavelengths = np.asarray(self.wavelengths, dtype=float)
        self.window_wavelengths_nm_ = [
            float(wavelengths[index]) for index in self.window_indices_
        ]
        self.feature_names_ = ["RWI"] + [
            f"R_{value:g}_nm" for value in self.window_wavelengths_nm_
        ]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "center_indices_")
        X = np.asarray(X, dtype=float)
        a, b, c = (X[:, index] for index in self.center_indices_)
        numerator = a - (b - c)
        denominator = a + (b - c)
        denominator = np.where(
            np.abs(denominator) < self.denominator_epsilon,
            np.where(
                denominator < 0,
                -self.denominator_epsilon,
                self.denominator_epsilon,
            ),
            denominator,
        )
        result = np.column_stack(
            [numerator / denominator, X[:, self.window_indices_]]
        )
        if not np.isfinite(result).all():
            raise FloatingPointError("RWI/window transformation produced non-finite values.")
        return result


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_components: int = 2, max_iter: int = 1000, tol: float = 1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSDAClassifier":
        self.classes_ = unique_labels(y)
        target = np.zeros((len(y), len(self.classes_)), dtype=float)
        lookup = {label: index for index, label in enumerate(self.classes_)}
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        scores = np.asarray(self.model_.predict(X), dtype=float)
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


SpectralPreprocessor.__module__ = MODULE_NAME
RWIWindowFeatures.__module__ = MODULE_NAME
PLSDAClassifier.__module__ = MODULE_NAME


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
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
        raise FileNotFoundError(
            f"Unsmoothed source data not found: {path}\n"
            "Set SOURCE_DATA_CSV to the original combined unsmoothed CSV."
        )
    if INPUT_IS_ALREADY_MOVING_AVERAGED and not ALLOW_ALREADY_PREPROCESSED_INPUT:
        raise ValueError(
            "The input is marked as already moving-average smoothed. A valid "
            "preprocessing comparison requires unsmoothed spectra."
        )
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
        raise ValueError("Expected 100 samples, each containing all 11 VWC classes.")
    step = float(np.median(np.diff(wavelengths)))
    return {
        "spectra": spectra,
        "wavelengths": wavelengths,
        "wavelength_step_nm": step,
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
    if set(manifest["physical_sample_id"]) != set(int(value) for value in physical_ids):
        raise ValueError("Manifest sample IDs do not match the spectral data.")
    if manifest["split"].value_counts().to_dict() != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Manifest must contain {EXPECTED_SPLIT_COUNTS}.")
    return manifest.sort_values("physical_sample_id").reset_index(drop=True)


def create_masks(groups: np.ndarray, manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    lookup = manifest.set_index("physical_sample_id")["split"].to_dict()
    split = np.asarray([lookup[int(group)] for group in groups])
    masks = {name: split == name for name in EXPECTED_SPLIT_COUNTS}
    names = list(EXPECTED_SPLIT_COUNTS)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if set(groups[masks[left]]) & set(groups[masks[right]]):
                raise AssertionError(f"Physical-sample overlap: {left}/{right}")
    return masks


def preprocessing_label(method: str) -> str:
    labels = {
        "none": "No preprocessing",
        "moving_average": "Moving average (3 point)",
        "snv": "SNV",
        "msc": "MSC",
        "savgol_smoothing": "Savitzky-Golay smoothing",
        "savgol_first_derivative": "Savitzky-Golay first derivative",
        "savgol_second_derivative": "Savitzky-Golay second derivative",
    }
    return labels[method]


def base_model_definitions(
    preprocessing_method: str,
    wavelengths: np.ndarray,
    wavelength_step_nm: float,
) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    preprocessing = SpectralPreprocessor(
        method=preprocessing_method,
        wavelength_step_nm=wavelength_step_nm,
    )
    feature_builder = RWIWindowFeatures(wavelengths=tuple(wavelengths.tolist()))
    common = [("preprocess", preprocessing), ("features", feature_builder)]
    return {
        "RF": (
            Pipeline(
                common
                + [
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
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", 1.0],
            },
        ),
        "PLS_DA": (
            Pipeline(
                common
                + [
                    ("scale", StandardScaler()),
                    ("model", PLSDAClassifier()),
                ]
            ),
            {"model__n_components": [2, 3, 5, 8]},
        ),
        "PCA_LDA": (
            Pipeline(
                common
                + [
                    ("scale", StandardScaler()),
                    ("pca", PCA(random_state=RANDOM_SEED)),
                    ("model", LinearDiscriminantAnalysis(solver="svd", tol=0.0001)),
                ]
            ),
            {"pca__n_components": [2, 3, 5, 8]},
        ),
    }


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
            raise AssertionError(f"Fold {fold} is missing one or more classes.")
    return splits


def aligned_probabilities(model: Pipeline, X: np.ndarray) -> np.ndarray:
    probability = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    aligned = np.zeros((len(X), N_CLASSES), dtype=float)
    for source, class_integer in enumerate(classes):
        aligned[:, int(class_integer)] = probability[:, source]
    totals = aligned.sum(axis=1, keepdims=True)
    if not np.isfinite(aligned).all() or np.any(totals <= 0):
        raise FloatingPointError("Invalid base-learner probabilities.")
    return aligned / totals


def tune_model(
    pipeline: Pipeline,
    grid: dict[str, list[Any]],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> GridSearchCV:
    splits = make_grouped_splits(
        X, y, groups, INNER_TUNING_FOLDS, seed
    )
    search = GridSearchCV(
        pipeline,
        grid,
        scoring=SCORING,
        cv=splits,
        refit=True,
        n_jobs=N_JOBS,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X, y, groups=groups)
    return search


def meta_probability_columns() -> list[str]:
    return [
        f"{learner}__P_{class_name}"
        for learner in BASE_LEARNERS
        for class_name in CLASS_ORDER
    ]


def save_meta_features(
    path: Path,
    features: np.ndarray,
    metadata: pd.DataFrame,
    fold_assignment: np.ndarray | None = None,
) -> None:
    frame = metadata.reset_index(drop=True).copy()
    if fold_assignment is not None:
        frame["oof_fold"] = fold_assignment
    for index, name in enumerate(meta_probability_columns()):
        frame[name] = features[:, index]
    frame.to_csv(path, index=False)


def generate_nested_oof(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    metadata: pd.DataFrame,
    wavelengths: np.ndarray,
    wavelength_step_nm: float,
    output_dir: Path,
) -> np.ndarray:
    definitions = base_model_definitions(method, wavelengths, wavelength_step_nm)
    outer_splits = make_grouped_splits(
        X, y, groups, OUTER_OOF_FOLDS, RANDOM_SEED
    )
    features = np.full((len(y), len(BASE_LEARNERS) * N_CLASSES), np.nan)
    fold_assignment = np.full(len(y), -1, dtype=int)
    tuning_rows: list[dict[str, Any]] = []
    for outer_fold, (fit_indices, holdout_indices) in enumerate(outer_splits, start=1):
        print(
            f"  {preprocessing_label(method)} OOF fold "
            f"{outer_fold}/{OUTER_OOF_FOLDS}",
            flush=True,
        )
        fold_assignment[holdout_indices] = outer_fold
        for learner_index, learner in enumerate(BASE_LEARNERS):
            pipeline, grid = definitions[learner]
            search = tune_model(
                clone(pipeline),
                grid,
                X[fit_indices],
                y[fit_indices],
                groups[fit_indices],
                RANDOM_SEED + 100 * outer_fold + learner_index,
            )
            start = learner_index * N_CLASSES
            features[holdout_indices, start : start + N_CLASSES] = (
                aligned_probabilities(search.best_estimator_, X[holdout_indices])
            )
            best_index = int(search.best_index_)
            tuning_rows.append(
                {
                    "outer_fold": outer_fold,
                    "base_learner": learner,
                    "inner_cv_macro_f1_mean": float(search.best_score_),
                    "inner_cv_macro_f1_std": float(
                        search.cv_results_["std_test_score"][best_index]
                    ),
                    "best_parameters": json.dumps(json_safe(search.best_params_)),
                }
            )
    if np.isnan(features).any() or np.any(fold_assignment < 1):
        raise AssertionError("Incomplete OOF probability matrix.")
    pd.DataFrame(tuning_rows).to_csv(
        output_dir / "nested_base_tuning_by_outer_fold.csv", index=False
    )
    save_meta_features(
        output_dir / "training_OOF_meta_features.csv",
        features,
        metadata,
        fold_assignment,
    )
    return features


def fit_full_training_base_models(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    wavelengths: np.ndarray,
    wavelength_step_nm: float,
    output_dir: Path,
) -> tuple[dict[str, Pipeline], dict[str, Any]]:
    fitted: dict[str, Pipeline] = {}
    frozen: dict[str, Any] = {}
    definitions = base_model_definitions(method, wavelengths, wavelength_step_nm)
    for learner_index, learner in enumerate(BASE_LEARNERS):
        pipeline, grid = definitions[learner]
        search = tune_model(
            clone(pipeline),
            grid,
            X,
            y,
            groups,
            RANDOM_SEED + 1000 + learner_index,
        )
        pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
            output_dir / f"full_training_grid_{learner}.csv", index=False
        )
        best_index = int(search.best_index_)
        fitted[learner] = search.best_estimator_
        frozen[learner] = {
            "best_parameters": json_safe(search.best_params_),
            "training_grouped_cv_macro_f1_mean": float(search.best_score_),
            "training_grouped_cv_macro_f1_std": float(
                search.cv_results_["std_test_score"][best_index]
            ),
        }
    return fitted, frozen


def tune_meta_model(
    features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    output_dir: Path,
) -> tuple[LogisticRegression, dict[str, Any]]:
    splits = make_grouped_splits(
        features, y, groups, META_TUNING_FOLDS, RANDOM_SEED + 2000
    )
    search = GridSearchCV(
        LogisticRegression(
            solver="lbfgs", max_iter=3000, random_state=RANDOM_SEED
        ),
        {"C": list(META_C_VALUES)},
        scoring=SCORING,
        cv=splits,
        refit=True,
        n_jobs=N_JOBS,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(features, y, groups=groups)
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
    frame.to_csv(output_dir / "validation_MLSD_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(
        output_dir / "validation_MLSD_overall_metrics.csv", index=False
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(
        output_dir / "validation_MLSD_classification_report.csv"
    )
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
        output_dir / "validation_MLSD_confusion_counts.csv"
    )
    pd.DataFrame(normalized, index=rows, columns=columns).to_csv(
        output_dir / "validation_MLSD_confusion_row_normalized.csv"
    )
    return metrics


def grouped_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    seed_offset: int,
) -> pd.DataFrame:
    unique_groups = np.unique(groups)
    indices_by_group = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(RANDOM_SEED + 9000 + seed_offset)
    samples: list[dict[str, float]] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled = rng.choice(unique_groups, len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled])
        samples.append(prediction_metrics(y_true[indices], y_pred[indices]))
    frame = pd.DataFrame(samples)
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "bootstrap_unit": "physical_sample_id",
                "iterations": BOOTSTRAP_ITERATIONS,
                "ci_lower_2.5_percent": float(frame[metric].quantile(0.025)),
                "ci_upper_97.5_percent": float(frame[metric].quantile(0.975)),
            }
            for metric in frame.columns
        ]
    )


def evaluate_one_preprocessing(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    train_metadata: pd.DataFrame,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    groups_validation: np.ndarray,
    validation_metadata: pd.DataFrame,
    wavelengths: np.ndarray,
    wavelength_step_nm: float,
    output_dir: Path,
    method_index: int,
) -> dict[str, Any]:
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=False)
    oof_features = generate_nested_oof(
        method,
        X_train,
        y_train,
        groups_train,
        train_metadata,
        wavelengths,
        wavelength_step_nm,
        method_dir,
    )
    fitted_base, frozen_base = fit_full_training_base_models(
        method,
        X_train,
        y_train,
        groups_train,
        wavelengths,
        wavelength_step_nm,
        method_dir,
    )
    fitted_meta, frozen_meta = tune_meta_model(
        oof_features, y_train, groups_train, method_dir
    )
    validation_parts = [
        aligned_probabilities(fitted_base[learner], X_validation)
        for learner in BASE_LEARNERS
    ]
    validation_features = np.column_stack(validation_parts)
    save_meta_features(
        method_dir / "validation_meta_features.csv",
        validation_features,
        validation_metadata,
    )
    probabilities = fitted_meta.predict_proba(validation_features)
    prediction = fitted_meta.predict(validation_features)
    metrics = save_evaluation(
        method_dir,
        y_validation,
        prediction,
        probabilities,
        validation_metadata,
    )
    grouped_bootstrap_ci(
        y_validation, prediction, groups_validation, method_index
    ).to_csv(method_dir / "validation_group_bootstrap_95ci.csv", index=False)
    record = {
        "preprocessing_method": method,
        "preprocessing_label": preprocessing_label(method),
        "base_models": frozen_base,
        "meta_model": frozen_meta,
        "validation_metrics": metrics,
    }
    (method_dir / "method_configuration_and_results.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return {
        "preprocessing_method": method,
        "preprocessing": preprocessing_label(method),
        **metrics,
    }


def main() -> None:
    loaded = load_spectral_data(SOURCE_DATA_CSV)
    spectra = loaded["spectra"]
    wavelengths = loaded["wavelengths"]
    wavelength_step_nm = loaded["wavelength_step_nm"]
    metadata = loaded["metadata"]
    y = loaded["y"]
    groups = loaded["groups"]
    manifest = load_split_manifest(SPLIT_MANIFEST_CSV, np.unique(groups))
    masks = create_masks(groups, manifest)

    # Deliberately exclude the locked test partition from every analysis object.
    X_train = spectra[masks["train"]]
    y_train = y[masks["train"]]
    groups_train = groups[masks["train"]]
    train_metadata = metadata.loc[masks["train"]]
    X_validation = spectra[masks["validation"]]
    y_validation = y[masks["validation"]]
    groups_validation = groups[masks["validation"]]
    validation_metadata = metadata.loc[masks["validation"]]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"Loaded {len(y)} spectra and {len(np.unique(groups))} physical samples."
    )
    print(
        "Sensitivity analysis uses only train=60 cubes/660 spectra and "
        "validation=20 cubes/220 spectra. Locked test is not evaluated."
    )
    print(
        f"RWI centers={RWI_WAVELENGTHS_NM}; local windows=±{WINDOW_HALF_WIDTH_NM} nm."
    )

    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(PREPROCESSING_METHODS):
        print(f"\n=== {preprocessing_label(method)} ===", flush=True)
        row = evaluate_one_preprocessing(
            method,
            X_train,
            y_train,
            groups_train,
            train_metadata,
            X_validation,
            y_validation,
            groups_validation,
            validation_metadata,
            wavelengths,
            wavelength_step_nm,
            output_dir,
            method_index,
        )
        rows.append(row)
        print(
            f"Validation macro P/R/F1: {row['macro_precision']:.4f} / "
            f"{row['macro_recall']:.4f} / {row['macro_f1']:.4f}",
            flush=True,
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["macro_f1", "macro_recall", "macro_precision"], ascending=False
    )
    comparison.insert(0, "rank_by_macro_f1", np.arange(1, len(comparison) + 1))
    ordered_columns = [
        "rank_by_macro_f1",
        "preprocessing_method",
        "preprocessing",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "weighted_f1",
    ]
    comparison = comparison[ordered_columns]
    comparison.to_csv(
        output_dir / "preprocessing_comparison_macro_metrics.csv", index=False
    )
    manifest.to_csv(output_dir / "split_manifest_used.csv", index=False)

    configuration = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "training and validation only; locked test not evaluated",
        "source_data_csv": str(SOURCE_DATA_CSV.resolve()),
        "source_data_sha256": sha256_file(SOURCE_DATA_CSV),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST_CSV),
        "input_marked_already_moving_averaged": INPUT_IS_ALREADY_MOVING_AVERAGED,
        "rwi_wavelengths_nm": list(RWI_WAVELENGTHS_NM),
        "window_half_width_nm": WINDOW_HALF_WIDTH_NM,
        "moving_average_half_width": MOVING_AVERAGE_HALF_WIDTH,
        "moving_average_points": 2 * MOVING_AVERAGE_HALF_WIDTH + 1,
        "savgol_window_length": SAVGOL_WINDOW_LENGTH,
        "savgol_polyorder": SAVGOL_POLYORDER,
        "preprocessing_methods": list(PREPROCESSING_METHODS),
        "outer_oof_folds": OUTER_OOF_FOLDS,
        "inner_tuning_folds": INNER_TUNING_FOLDS,
        "meta_tuning_folds": META_TUNING_FOLDS,
        "random_seed": RANDOM_SEED,
        "base_learners": list(BASE_LEARNERS),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (output_dir / "run_configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    (output_dir / "COMPLETED.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "locked_test_evaluated": False,
                "comparison_table": "preprocessing_comparison_macro_metrics.csv",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nComparative validation table:")
    print(
        comparison[
            ["preprocessing", "macro_precision", "macro_recall", "macro_f1"]
        ].to_string(index=False)
    )
    print("\nResults:", output_dir)


if __name__ == "__main__":
    main()
