#!/usr/bin/env python3
"""Full-spectrum 1D-CNN for rockwool moisture classification.

Development uses 60 training and 20 validation samples to select settings.
Final testing refits on all 80 development samples and evaluates 20 held-out
samples using the frozen settings. See README.md for setup and execution.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


# User settings
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CSV = SCRIPT_DIR / "All_classes_smoothed.csv"
SPLIT_MANIFEST_CSV = SCRIPT_DIR / "physical_sample_split_60_20_20_seed42.csv"
OUTPUT_ROOT = SCRIPT_DIR / "results_CNN1D_60_20_20"
FROZEN_CONFIG_JSON = OUTPUT_ROOT / "frozen_CNN1D_config.json"

RUN_MODE = "final_test"  # "development" or "final_test"
ALLOW_COMPLETED_TEST_RERUN = False

RANDOM_SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 250
MIN_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 30
EPOCH_REPORT_INTERVAL = 10
NUM_WORKERS = 0  # Avoid worker spawning on Windows.
BOOTSTRAP_ITERATIONS = 2000

LEARNING_RATES = (1e-4, 5e-4, 1e-3)
WEIGHT_DECAYS = (1e-5, 1e-4)

STANDARDIZE_SPECTRA = True
STANDARD_DEVIATION_EPSILON = 1e-12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONV1_FILTERS = 16
CONV1_KERNEL = 7
CONV2_FILTERS = 32
CONV2_KERNEL = 5
CONV3_FILTERS = 64
CONV3_KERNEL = 3
POOL_SIZE = 2
HIDDEN_UNITS = 64
DROPOUT_RATE = 0.3

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


class SpectralDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


class SpectralCNN1D(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, CONV1_FILTERS, kernel_size=CONV1_KERNEL, padding="same"),
            nn.BatchNorm1d(CONV1_FILTERS),
            nn.ReLU(),
            nn.MaxPool1d(POOL_SIZE),
            nn.Conv1d(
                CONV1_FILTERS,
                CONV2_FILTERS,
                kernel_size=CONV2_KERNEL,
                padding="same",
            ),
            nn.BatchNorm1d(CONV2_FILTERS),
            nn.ReLU(),
            nn.MaxPool1d(POOL_SIZE),
            nn.Conv1d(
                CONV2_FILTERS,
                CONV3_FILTERS,
                kernel_size=CONV3_KERNEL,
                padding="same",
            ),
            nn.BatchNorm1d(CONV3_FILTERS),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CONV3_FILTERS, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_UNITS, n_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


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
    spectra = matrix[order, :].T.astype(np.float32)

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
    if set(manifest["physical_sample_id"]) != set(int(x) for x in physical_ids):
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


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not STANDARDIZE_SPECTRA:
        return np.zeros(X.shape[1], dtype=np.float32), np.ones(
            X.shape[1], dtype=np.float32
        )
    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < STANDARD_DEVIATION_EPSILON, 1.0, std).astype(np.float32)
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    transformed = ((X - mean) / std).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise FloatingPointError("Standardized spectra contain non-finite values.")
    return transformed


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        SpectralDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
        generator=generator if shuffle else None,
    )


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


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_true: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * len(labels)
            total_count += len(labels)
            all_true.append(labels.cpu().numpy())
            all_probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    y_true = np.concatenate(all_true)
    probabilities = np.concatenate(all_probabilities)
    prediction = probabilities.argmax(axis=1)
    return total_loss / total_count, y_true, prediction, probabilities


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for inputs, labels in loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total_count += len(labels)
    return total_loss / total_count


def copy_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_validation_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    learning_rate: float,
    weight_decay: float,
    candidate_index: int,
) -> dict[str, Any]:
    set_global_seed(RANDOM_SEED)
    model = SpectralCNN1D().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_loader = make_loader(
        X_train, y_train, shuffle=True, seed=RANDOM_SEED
    )
    validation_loader = make_loader(
        X_validation, y_validation, shuffle=False, seed=RANDOM_SEED
    )

    best_key = (-np.inf, -np.inf, -np.inf)
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = np.inf
    best_metrics: dict[str, float] | None = None
    best_prediction: np.ndarray | None = None
    best_probabilities: np.ndarray | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        validation_loss, y_true, prediction, probabilities = evaluate_model(
            model, validation_loader, criterion
        )
        if not np.array_equal(y_true, y_validation):
            raise AssertionError("Validation loader changed observation order.")
        metrics = prediction_metrics(y_true, prediction)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
        key = (metrics["macro_f1"], metrics["accuracy"], -validation_loss)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy_state_to_cpu(model)
            best_loss = validation_loss
            best_metrics = metrics
            best_prediction = prediction.copy()
            best_probabilities = probabilities.copy()
        if epoch == 1 or epoch % EPOCH_REPORT_INTERVAL == 0:
            print(
                f"  candidate {candidate_index}: epoch {epoch}/{MAX_EPOCHS}, "
                f"val macro-F1={metrics['macro_f1']:.4f}",
                flush=True,
            )
        if (
            epoch >= MIN_EPOCHS
            and best_epoch > 0
            and epoch - best_epoch >= EARLY_STOPPING_PATIENCE
        ):
            break

    if (
        best_state is None
        or best_metrics is None
        or best_prediction is None
        or best_probabilities is None
    ):
        raise RuntimeError("CNN candidate did not produce a valid checkpoint.")
    return {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
        "validation_loss": float(best_loss),
        "validation_metrics": best_metrics,
        "prediction": best_prediction,
        "probabilities": best_probabilities,
        "state_dict": best_state,
        "history": pd.DataFrame(history),
    }


def train_fixed_epochs(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
) -> tuple[nn.Module, pd.DataFrame]:
    if epochs < 1:
        raise ValueError("Frozen epoch count must be at least 1.")
    set_global_seed(RANDOM_SEED)
    model = SpectralCNN1D().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loader = make_loader(X, y, shuffle=True, seed=RANDOM_SEED + 5000)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, criterion, optimizer)
        history.append({"epoch": epoch, "development_training_loss": loss})
        if epoch == 1 or epoch % EPOCH_REPORT_INTERVAL == 0 or epoch == epochs:
            print(f"  final refit epoch {epoch}/{epochs}, loss={loss:.6f}", flush=True)
    return model, pd.DataFrame(history)


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
        frame[f"probability_{class_name}"] = probabilities[:, class_integer]
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
) -> pd.DataFrame:
    unique_groups = np.unique(groups)
    indices_by_group = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(RANDOM_SEED + 9000)
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


def architecture_record() -> dict[str, Any]:
    return {
        "conv1_filters": CONV1_FILTERS,
        "conv1_kernel": CONV1_KERNEL,
        "conv2_filters": CONV2_FILTERS,
        "conv2_kernel": CONV2_KERNEL,
        "conv3_filters": CONV3_FILTERS,
        "conv3_kernel": CONV3_KERNEL,
        "pool_size": POOL_SIZE,
        "adaptive_average_pool_output": 1,
        "hidden_units": HIDDEN_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "output_classes": N_CLASSES,
    }


def run_information(wavelengths: np.ndarray) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": sha256_file(DATA_CSV),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST_CSV),
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "minimum_epochs": MIN_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "learning_rate_candidates": list(LEARNING_RATES),
        "weight_decay_candidates": list(WEIGHT_DECAYS),
        "standardize_spectra": STANDARDIZE_SPECTRA,
        "architecture": architecture_record(),
        "wavelength_count": int(len(wavelengths)),
        "wavelength_min_nm": float(wavelengths.min()),
        "wavelength_max_nm": float(wavelengths.max()),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_used": DEVICE,
        },
    }


def run_development(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    masks: dict[str, np.ndarray],
    manifest: pd.DataFrame,
) -> None:
    completion_marker = OUTPUT_ROOT / "LOCKED_TEST_COMPLETED.json"
    if completion_marker.exists():
        raise RuntimeError("Locked CNN test already completed; development is frozen.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / f"development_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    X_train_raw, y_train = spectra[masks["train"]], y[masks["train"]]
    X_validation_raw, y_validation = (
        spectra[masks["validation"]],
        y[masks["validation"]],
    )
    validation_metadata = metadata.loc[masks["validation"]]
    mean, std = fit_standardizer(X_train_raw)
    X_train = apply_standardizer(X_train_raw, mean, std)
    X_validation = apply_standardizer(X_validation_raw, mean, std)
    np.savez_compressed(output_dir / "training_standardizer.npz", mean=mean, std=std)

    candidates = list(itertools.product(LEARNING_RATES, WEIGHT_DECAYS))
    candidate_rows: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []
    print(f"Evaluating {len(candidates)} CNN hyperparameter candidates on validation.")
    for candidate_index, (learning_rate, weight_decay) in enumerate(candidates, start=1):
        print(
            f"\nCNN candidate {candidate_index}/{len(candidates)}: "
            f"lr={learning_rate:g}, weight_decay={weight_decay:g}",
            flush=True,
        )
        result = train_validation_candidate(
            X_train,
            y_train,
            X_validation,
            y_validation,
            learning_rate,
            weight_decay,
            candidate_index,
        )
        result["history"].to_csv(
            output_dir / f"training_history_candidate_{candidate_index:02d}.csv",
            index=False,
        )
        candidate_results.append(result)
        candidate_rows.append(
            {
                "candidate": candidate_index,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "selected_epoch": result["best_epoch"],
                "validation_loss": result["validation_loss"],
                **{
                    f"validation_{key}": value
                    for key, value in result["validation_metrics"].items()
                },
            }
        )

    comparison = pd.DataFrame(candidate_rows).sort_values(
        ["validation_macro_f1", "validation_accuracy", "validation_loss"],
        ascending=[False, False, True],
    )
    comparison.to_csv(output_dir / "CNN_hyperparameter_validation_results.csv", index=False)
    best_candidate_number = int(comparison.iloc[0]["candidate"])
    selected = candidate_results[best_candidate_number - 1]
    validation_metrics = save_evaluation(
        output_dir,
        "validation_CNN1D",
        y_validation,
        selected["prediction"],
        selected["probabilities"],
        validation_metadata,
    )
    torch.save(
        {
            "state_dict": selected["state_dict"],
            "mean": mean,
            "std": std,
            "wavelengths": wavelengths,
            "learning_rate": selected["learning_rate"],
            "weight_decay": selected["weight_decay"],
            "selected_epoch": selected["best_epoch"],
            "class_order": CLASS_ORDER,
        },
        output_dir / "selected_development_CNN1D.pt",
    )
    manifest.to_csv(output_dir / "split_manifest_used.csv", index=False)

    frozen = {
        **run_information(wavelengths),
        "status": "development_complete_test_not_evaluated",
        "selected_learning_rate": float(selected["learning_rate"]),
        "selected_weight_decay": float(selected["weight_decay"]),
        "selected_epoch": int(selected["best_epoch"]),
        "validation_metrics": validation_metrics,
        "selection_rule": "highest validation macro-F1, then accuracy, then lower loss",
    }
    FROZEN_CONFIG_JSON.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    (output_dir / "frozen_CNN1D_config.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )
    print("\nCNN development completed without evaluating the test partition.")
    print(
        "Selected: lr=",
        selected["learning_rate"],
        "weight_decay=",
        selected["weight_decay"],
        "epoch=",
        selected["best_epoch"],
    )
    print("Validation macro-F1:", f"{validation_metrics['macro_f1']:.4f}")
    print("Frozen configuration:", FROZEN_CONFIG_JSON)


def run_final_test(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    metadata: pd.DataFrame,
    masks: dict[str, np.ndarray],
    manifest: pd.DataFrame,
) -> None:
    if not FROZEN_CONFIG_JSON.exists():
        raise FileNotFoundError("Run CNN development before final_test.")
    frozen = json.loads(FROZEN_CONFIG_JSON.read_text(encoding="utf-8"))
    if frozen.get("status") != "development_complete_test_not_evaluated":
        raise ValueError("Unexpected frozen CNN status.")
    if frozen.get("data_sha256") != sha256_file(DATA_CSV):
        raise ValueError("Data file differs from CNN development.")
    if frozen.get("split_manifest_sha256") != sha256_file(SPLIT_MANIFEST_CSV):
        raise ValueError("Split manifest differs from CNN development.")
    if frozen.get("architecture") != architecture_record():
        raise ValueError("CNN architecture differs from development.")
    if frozen.get("standardize_spectra") != STANDARDIZE_SPECTRA:
        raise ValueError("CNN standardization setting differs from development.")
    if frozen.get("wavelength_count") != int(len(wavelengths)):
        raise ValueError("Wavelength count differs from development.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    completion_marker = OUTPUT_ROOT / "LOCKED_TEST_COMPLETED.json"
    if completion_marker.exists() and not ALLOW_COMPLETED_TEST_RERUN:
        raise RuntimeError("Locked CNN test has already been completed.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / f"locked_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    development_mask = masks["train"] | masks["validation"]
    X_development_raw, y_development = spectra[development_mask], y[development_mask]
    X_test_raw, y_test = spectra[masks["test"]], y[masks["test"]]
    groups_test = groups[masks["test"]]
    test_metadata = metadata.loc[masks["test"]]
    mean, std = fit_standardizer(X_development_raw)
    X_development = apply_standardizer(X_development_raw, mean, std)
    X_test = apply_standardizer(X_test_raw, mean, std)
    np.savez_compressed(output_dir / "development_standardizer.npz", mean=mean, std=std)

    learning_rate = float(frozen["selected_learning_rate"])
    weight_decay = float(frozen["selected_weight_decay"])
    selected_epoch = int(frozen["selected_epoch"])
    print(
        f"Refitting CNN on 80 development cubes for {selected_epoch} frozen epochs..."
    )
    model, history = train_fixed_epochs(
        X_development,
        y_development,
        learning_rate,
        weight_decay,
        selected_epoch,
    )
    history.to_csv(output_dir / "final_refit_training_history.csv", index=False)
    criterion = nn.CrossEntropyLoss()
    test_loader = make_loader(X_test, y_test, shuffle=False, seed=RANDOM_SEED)
    test_loss, observed_y, prediction, probabilities = evaluate_model(
        model, test_loader, criterion
    )
    if not np.array_equal(observed_y, y_test):
        raise AssertionError("Test loader changed observation order.")
    test_metrics = save_evaluation(
        output_dir,
        "locked_test_CNN1D",
        y_test,
        prediction,
        probabilities,
        test_metadata,
    )
    test_metrics["cross_entropy_loss"] = float(test_loss)
    pd.DataFrame([test_metrics]).to_csv(
        output_dir / "locked_test_CNN1D_overall_metrics.csv", index=False
    )
    grouped_bootstrap_ci(y_test, prediction, groups_test).to_csv(
        output_dir / "locked_test_group_bootstrap_95ci.csv", index=False
    )
    torch.save(
        {
            "state_dict": copy_state_to_cpu(model),
            "mean": mean,
            "std": std,
            "wavelengths": wavelengths,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": selected_epoch,
            "architecture": architecture_record(),
            "class_order": CLASS_ORDER,
        },
        output_dir / "fitted_CNN1D_locked_test.pt",
    )
    manifest.to_csv(output_dir / "split_manifest_used.csv", index=False)

    run_record = {
        **run_information(wavelengths),
        "status": "locked_test_completed",
        "frozen_config_sha256": sha256_file(FROZEN_CONFIG_JSON),
        "selected_learning_rate": learning_rate,
        "selected_weight_decay": weight_decay,
        "selected_epoch": selected_epoch,
        "test_metrics": test_metrics,
        "test_evaluated_after_all_choices_were_frozen": True,
        "output_directory": str(output_dir.resolve()),
    }
    (output_dir / "locked_test_run_record.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    completion_marker.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print("\nCNN locked test completed once using frozen settings.")
    print(json.dumps(test_metrics, indent=2))
    print("Results:", output_dir)


def main() -> None:
    if RUN_MODE not in {"development", "final_test"}:
        raise ValueError("RUN_MODE must be development or final_test.")
    set_global_seed(RANDOM_SEED)
    loaded = load_spectral_data(DATA_CSV)
    spectra = loaded["spectra"]
    wavelengths = loaded["wavelengths"]
    metadata = loaded["metadata"]
    y = loaded["y"]
    groups = loaded["groups"]
    manifest = load_split_manifest(SPLIT_MANIFEST_CSV, np.unique(groups))
    masks = create_masks(groups, manifest)
    print(
        f"Loaded {len(y)} full spectra with {spectra.shape[1]} wavelengths from "
        f"{len(np.unique(groups))} physical samples. Device: {DEVICE}."
    )
    print(
        "Split: "
        + ", ".join(
            f"{name}={len(np.unique(groups[mask]))} cubes/{int(mask.sum())} spectra"
            for name, mask in masks.items()
        )
    )
    if RUN_MODE == "development":
        run_development(spectra, wavelengths, y, metadata, masks, manifest)
    else:
        run_final_test(
            spectra, wavelengths, y, groups, metadata, masks, manifest
        )


if __name__ == "__main__":
    main()
