#!/usr/bin/env python3
"""Compare six fixed-setting base learners and three-learner MLSD with grouped CV.

INSTALL (Python >=3.10):
    python -m pip install numpy pandas scipy scikit-learn matplotlib
RUN:
    python rockwool_grouped_cv.py --csv All_classes_smoothed.csv
    python rockwool_grouped_cv.py --csv All_classes_smoothed.csv --window-nm 5
    python rockwool_grouped_cv.py --csv data.csv --wavelengths 560 1600 1920

INPUT: wide CSV, one wavelength column (case-insensitive), one column per
spectrum, e.g. dry17, sub9-17, ..., sub0-17. All those spectra must actually
come from physical sample 17. The script cannot establish that correspondence.
Default ordinal class order: dry,sub9,sub8,...,sub0 (increasing moisture).
Labels are NOT converted to VWC percentages because corrected means vary.
Files must have identical wavelength grids. IDs are GLOBAL across input files;
the same ID in different files remains the same physical sample.

If names do not unambiguously encode true physical IDs, supply --metadata map.csv
with columns file,column,class,sample_id; file is the input basename. It must
contain exactly one row for every spectrum. Ambiguous names such as sub312 are
rejected without metadata. Do not invent IDs from column order.

METHOD:
* Randomly partition physical IDs into five nonoverlapping outer groups.
* Use identical outer partitions, features and fixed base settings for models.
* All imputation/scaling/PLS/PCA fitting occurs inside each training partition.
* MLSD uses grouped inner five-fold OOF probabilities of RF, PLS-DA, PCA-LDA;
  a fixed-C logistic meta-classifier is fitted on their concatenated outputs.
  Refit base learners on the entire outer training set before outer evaluation.
* No wavelength search, hyperparameter tuning, CNN, augmentation, smoothing,
  final deployment fit, or separate holdout is performed here.
* Default triplet is (570,1560,1940) nm; RWI=(Ra-(Rb-Rc))/(Ra+(Rb-Rc)).
  Default input is ONE RWI feature for EVERY model. --window-nm 5 adds the
  measured bands within +/-5 nm of each selected wavelength for EVERY model.
  On a complete 1-nm grid this gives 34 features. No interpolation is performed.
* Table 6 settings are used, except SVM probability=False: SVM is standalone
  only, so no probabilities are needed. This avoids libsvm's ungrouped internal
  probability-calibration CV and retains its native SVC class predictions.
* PLS-DA follows Table 6: one-hot-target PLS, then logistic regression on scores.

INTERPRETATION:
This reproduces the fixed-setting development-stage CV described in the revised
text. It does NOT remove prior selection bias if wavelengths, hyperparameters,
or the three-learner ensemble were chosen using these same samples. The inner
folds generate stacking inputs; they are NOT nested hyperparameter selection.
Use a separate evaluation protocol for the independent-validation claim.

OUTPUTS (new output directory required; default timestamped):
summary_mean_sd.csv: unweighted outer-fold mean and sample SD (ddof=1).
fold_metrics.csv, pooled_metrics.csv, per_class_by_fold.csv,
per_class_pooled.csv: accuracy, macro P/R/F1, ordinal metrics, class supports.
outer_predictions.csv: one outer-held-out prediction per spectrum per model.
paired_comparison.csv: MLSD-minus-base outer-fold differences in percentage points.
confusion_*.csv and confusion_matrices.png: pooled outer-held-out predictions.
comparison.png: macro-F1 fold means +/- SD; not confidence intervals.
outer_assignments.csv, split_audit.json, meta_oof_fold_*.csv: auditable splits.
run_config.json, features.csv, sample_inventory.csv, README_results.txt.
Metrics use fractions [0,1], except rank MAE and labelled percentage-point fields.
Pooled metrics and unweighted fold means are distinct; do not interchange them.
Every class must be present for each physical sample (replicates allowed), and
every fit is checked for all classes. Unequal replicate counts weight spectra
unequally; sample_inventory.csv makes this visible. Fold SD is not a CI.
"""

import argparse
import hashlib
import json
import platform
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_NAMES = ["PLS-DA", "PCA-LDA", "SVM", "KNN", "ANN", "RF"]
STACK_NAMES = ["RF", "PLS-DA", "PCA-LDA"]


class PLSDA(ClassifierMixin, BaseEstimator):
    """PLS projection against one-hot targets followed by logistic classification."""
    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_, encoded = np.unique(y, return_inverse=True)
        target = np.eye(len(self.classes_))[encoded]
        # Rank cap avoids requesting more components than the training data allow.
        rank = np.linalg.matrix_rank(X - X.mean(axis=0))
        if rank == 0:
            raise ValueError("Training features have zero rank; check spectra/RWI.")
        self.pls_ = PLSRegression(n_components=min(self.n_components, X.shape[1], rank),
                                  scale=True, max_iter=500, tol=1e-6)
        self.pls_.fit(X, target)
        self.lr_ = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, tol=1e-4)
        self.lr_.fit(self.pls_.transform(X), y)
        return self

    def predict(self, X):
        return self.lr_.predict(self.pls_.transform(X))

    def predict_proba(self, X):
        return self.lr_.predict_proba(self.pls_.transform(X))


def make_model(name, p, seed, jobs):
    models = {
        "PLS-DA": PLSDA(n_components=min(5, p)),
        "PCA-LDA": make_pipeline(PCA(n_components=min(10, p), svd_solver="auto",
                                     whiten=False, random_state=seed),
                                 LinearDiscriminantAnalysis(solver="svd", tol=1e-4)),
        "SVM": SVC(C=1.0, kernel="rbf", gamma="scale", probability=False,
                   tol=1e-3, max_iter=-1, random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance", p=2),
        "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                             solver="adam", alpha=1e-4, batch_size="auto",
                             learning_rate_init=0.001, max_iter=1000, tol=1e-4,
                             early_stopping=False, n_iter_no_change=10,
                             shuffle=True, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=30, class_weight="balanced",
                                     max_features="sqrt", random_state=seed,
                                     n_jobs=jobs),
    }
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), models[name])


def parse_name(name):
    label = name.strip().lower()
    match = re.fullmatch(r"(dry|sub[0-9])[-_](\d+)", label)
    if match is None:
        match = re.fullmatch(r"(dry)(\d+)", label)
    if match is None:
        raise ValueError(f"Ambiguous column {name!r}; use sub3-17/dry17 or --metadata.")
    return match.group(1), str(int(match.group(2)))


def load_data(paths, metadata_path, classes):
    metadata = None
    if metadata_path:
        metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
        required = {"file", "column", "class", "sample_id"}
        if not required.issubset(metadata.columns):
            raise ValueError(f"Metadata requires columns {sorted(required)}")
        if metadata.duplicated(["file", "column"]).any():
            raise ValueError("Duplicate metadata file/column keys.")
        metadata = metadata.set_index(["file", "column"])
    if len({p.name for p in paths}) != len(paths):
        raise ValueError("Input CSV basenames must be unique.")
    spectra, records, wavelengths = [], [], None
    for path in paths:
        # Reject duplicate headers before pandas silently renames them.
        import csv
        with path.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
        if len(header) != len(set(header)):
            raise ValueError(f"Duplicate spectrum headers in {path}")
        df = pd.read_csv(path)
        wl_columns = [c for c in df.columns if c.strip().lower() == "wavelength"]
        if len(wl_columns) != 1:
            raise ValueError(f"Exactly one wavelength column required in {path}")
        wl_col = wl_columns[0]
        wl = pd.to_numeric(df[wl_col], errors="raise").to_numpy(float)
        if not np.isfinite(wl).all() or len(np.unique(wl)) != len(wl):
            raise ValueError(f"Nonfinite or duplicate wavelengths in {path}")
        order = np.argsort(wl)
        wl = wl[order]
        if wavelengths is None:
            wavelengths = wl
        elif len(wl) != len(wavelengths) or not np.allclose(wl, wavelengths, rtol=0, atol=1e-8):
            raise ValueError("Wavelength grids differ; align them explicitly first.")
        for column in df.columns:
            if column == wl_col:
                continue
            if metadata is None:
                label, group = parse_name(column)
            else:
                key = (path.name, column)
                if key not in metadata.index:
                    raise ValueError(f"Missing metadata for {key}")
                row = metadata.loc[key]
                label, group = row["class"].strip().lower(), row["sample_id"].strip()
                if not group:
                    raise ValueError(f"Empty physical sample ID for {key}")
            if label not in classes:
                raise ValueError(f"Unknown class {label!r}; check --class-order.")
            values = pd.to_numeric(df[column], errors="raise").to_numpy(float)[order]
            if np.isinf(values).any():
                raise ValueError(f"Infinite reflectance in {path.name}/{column}")
            spectra.append(values)
            records.append({"file": path.name, "column": column,
                            "sample_id": group, "class": label})
    if not spectra:
        raise ValueError("No spectra found.")
    meta = pd.DataFrame(records)
    if metadata is not None and len(metadata) != len(meta):
        raise ValueError("Metadata includes unused rows; supply exactly the input spectra.")
    inventory = pd.crosstab(meta.sample_id, meta["class"]).reindex(columns=classes, fill_value=0)
    if (inventory == 0).any().any():
        raise ValueError("Some IDs lack moisture classes. Check physical ID mapping and class order.")
    return np.vstack(spectra), wavelengths, meta, inventory


def build_features(spectra, wls, requested, tolerance, window):
    indices = [int(np.argmin(abs(wls - value))) for value in requested]
    actual = wls[indices]
    if len(set(indices)) != 3 or np.any(abs(actual - requested) > tolerance):
        raise ValueError(f"Cannot map triplet {requested} within {tolerance} nm; nearest={actual}")
    a, b, c = spectra[:, indices].T
    denominator = a + (b - c)
    scale = np.maximum(abs(a) + abs(b) + abs(c), np.finfo(float).tiny)
    bad = np.isfinite(denominator) & (abs(denominator) <= 1e-12 * scale)
    if bad.any():
        raise ValueError(f"RWI denominator zero/near-zero for {bad.sum()} spectra; inspect data.")
    with np.errstate(invalid="ignore", divide="ignore"):
        rwi = (a - (b - c)) / denominator
    if np.isinf(rwi).any():
        raise ValueError("Nonfinite RWI; inspect denominator and input data.")
    X, names = rwi[:, None], ["RWI"]
    if window > 0:
        selected = np.flatnonzero(np.any(abs(wls[:, None] - actual) <= window + 1e-8, axis=1))
        X = np.column_stack([X, spectra[:, selected]])
        names += [f"R_{wls[i]:g}_nm" for i in selected]
    return X, names, actual


def grouped_splits(y, groups, folds, seed, audit, context):
    ids = np.unique(groups)
    if len(ids) < folds:
        raise ValueError(f"{context}: insufficient physical samples for {folds} folds")
    result = []
    # KFold operates on unique physical IDs, NEVER on individual spectra.
    for fold, (train_g, test_g) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(ids), 1):
        train = np.flatnonzero(np.isin(groups, ids[train_g]))
        test = np.flatnonzero(np.isin(groups, ids[test_g]))
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("Physical group leakage detected")
        if set(y[train]) != set(y) or set(y[test]) != set(y):
            raise ValueError(f"{context}: all classes must occur in every partition")
        audit.append({"context": context, "fold": fold,
                      "train_ids": ids[train_g].tolist(), "evaluation_ids": ids[test_g].tolist(),
                      "n_train_spectra": len(train), "n_evaluation_spectra": len(test),
                      "overlap": 0})
        result.append((train, test))
    return result


def fit_model(name, X, y, seed, jobs):
    if np.isnan(X).all(axis=0).any():
        raise ValueError("A feature is entirely missing in a training partition.")
    model = make_model(name, X.shape[1], seed, jobs)
    model.fit(X, y)
    return model


def aligned_probabilities(model, X, k):
    if not np.array_equal(model.classes_, np.arange(k)):
        raise AssertionError("Unexpected probability class order")
    prob = model.predict_proba(X)
    if not np.isfinite(prob).all() or not np.allclose(prob.sum(axis=1), 1):
        raise ValueError("Invalid class probabilities")
    return prob


def metrics(y, pred, k):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=np.arange(k),
                                               average="macro", zero_division=0)
    error = abs(y - pred)
    return {"accuracy": accuracy_score(y, pred), "macro_precision": p,
            "macro_recall": r, "macro_f1": f,
            "within_one_class_accuracy": np.mean(error <= 1),
            "exactly_one_class_error": np.mean(error == 1),
            "more_than_one_class_error": np.mean(error > 1),
            "mean_absolute_rank_error": np.mean(error)}


def class_metrics(y, pred, classes, name, fold):
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=np.arange(len(classes)),
                                                     zero_division=0)
    return [{"model": name, "fold": fold, "class": label, "precision": p[i],
             "recall": r[i], "f1": f[i], "support": int(support[i])}
            for i, label in enumerate(classes)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, default=Path("rockwool_cv_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--wavelengths", nargs=3, type=float, default=[570, 1560, 1940])
    parser.add_argument("--wavelength-tolerance", type=float, default=1.0)
    parser.add_argument("--window-nm", type=float, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--class-order", default="dry,sub9,sub8,sub7,sub6,sub5,sub4,sub3,sub2,sub1,sub0")
    args = parser.parse_args()
    if args.window_nm < 0 or args.wavelength_tolerance < 0:
        parser.error("Window and wavelength tolerance must be nonnegative.")
    classes = [c.strip().lower() for c in args.class_order.split(",")]
    if len(classes) < 2 or len(set(classes)) != len(classes):
        parser.error("Provide at least two distinct classes in increasing moisture order.")
    spectra, wls, metadata, inventory = load_data(args.csv, args.metadata, classes)
    X, feature_names, actual = build_features(spectra, wls, np.array(args.wavelengths),
                                             args.wavelength_tolerance, args.window_nm)
    y = np.array([classes.index(label) for label in metadata["class"]])
    groups = metadata.sample_id.to_numpy()
    k = len(classes)
    audit = []
    splits = grouped_splits(y, groups, args.folds, args.seed, audit, "outer")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)  # Never silently overwrite an earlier run.
    metadata.to_csv(out / "sample_inventory.csv", index=False)
    inventory.to_csv(out / "sample_class_counts.csv")
    pd.DataFrame({"feature": feature_names, "missing_count": np.isnan(X).sum(axis=0)}).to_csv(out / "features.csv", index=False)
    assignments = metadata.copy()
    assignments["outer_fold"] = 0
    for fold, (_, test) in enumerate(splits, 1):
        assignments.loc[test, "outer_fold"] = fold
    assignments.to_csv(out / "outer_assignments.csv", index=False)
    config = {"arguments": {key: str(value) if isinstance(value, Path) else
                             [str(v) for v in value] if key == "csv" else value
                             for key, value in vars(args).items()},
              "actual_wavelengths_nm": actual.tolist(), "feature_names": feature_names,
              "class_order": classes, "physical_samples": len(inventory), "spectra": len(y),
              "stack_learners": STACK_NAMES, "selection": "fixed settings; no tuning",
              "SD_ddof": 1, "sklearn_version": sklearn.__version__,
              "python_version": platform.python_version(), "numpy_version": np.__version__,
              "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in args.csv + ([args.metadata] if args.metadata else [])},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "estimator_parameters": {name: {key: str(value) for key, value in
                 make_model(name, X.shape[1], args.seed, args.jobs).get_params().items()}
                 for name in BASE_NAMES}}
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Loaded {len(y)} spectra, {len(inventory)} physical samples, {X.shape[1]} features.", flush=True)
    print(f"Actual RWI triplet: {actual}; MLSD: {STACK_NAMES}", flush=True)
    print("Fixed-setting development CV; prior wavelength/hyperparameter selection is not undone.", flush=True)
    predictions = {name: np.full(len(y), -1, int) for name in BASE_NAMES + ["MLSD"]}
    fold_rows, class_rows = [], []
    for fold, (train, test) in enumerate(splits, 1):
        print(f"Outer fold {fold}/{args.folds}: {len(set(groups[train]))} train / {len(set(groups[test]))} evaluation samples", flush=True)
        Xt, yt, gt = X[train], y[train], groups[train]
        inner = grouped_splits(yt, gt, args.inner_folds, args.seed + fold, audit, f"outer_{fold}_stacking")
        meta_train = np.full((len(train), k * len(STACK_NAMES)), np.nan)
        inner_assignment = np.zeros(len(train), int)
        for inner_fold, (it, iv) in enumerate(inner, 1):
            inner_assignment[iv] = inner_fold
            for j, name in enumerate(STACK_NAMES):
                model = fit_model(name, Xt[it], yt[it], args.seed, args.jobs)
                meta_train[iv, j*k:(j+1)*k] = aligned_probabilities(model, Xt[iv], k)
        if not np.isfinite(meta_train).all() or (inner_assignment == 0).any():
            raise AssertionError("Incomplete OOF meta-training matrix")
        oof = metadata.iloc[train].reset_index(drop=True).copy()
        oof["inner_fold"] = inner_assignment
        for j, name in enumerate(STACK_NAMES):
            for c, label in enumerate(classes):
                oof[f"{name}_P_{label}"] = meta_train[:, j*k+c]
        oof.to_csv(out / f"meta_oof_fold_{fold}.csv", index=False)
        meta_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, tol=1e-4)
        meta_model.fit(meta_train, yt)
        fitted = {}
        for name in BASE_NAMES:
            # These exact fitted learners serve BOTH standalone and MLSD evaluation.
            fitted[name] = fit_model(name, Xt, yt, args.seed, args.jobs)
            predictions[name][test] = fitted[name].predict(X[test])
        meta_test = np.hstack([aligned_probabilities(fitted[name], X[test], k) for name in STACK_NAMES])
        predictions["MLSD"][test] = meta_model.predict(meta_test)
        for name in predictions:
            pred = predictions[name][test]
            row = {"model": name, "fold": fold, **metrics(y[test], pred, k)}
            fold_rows.append(row)
            class_rows.extend(class_metrics(y[test], pred, classes, name, fold))
            print(f"  {name:8s} macro-F1={row['macro_f1']:.4f}", flush=True)
        # Checkpoint completed fold metrics and split audit.
        pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
        (out / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if any((pred < 0).any() for pred in predictions.values()):
        raise AssertionError("Not all spectra received an outer prediction")
    folds = pd.DataFrame(fold_rows)
    metric_names = list(metrics(y, y, k))
    summary = folds.groupby("model", sort=False)[metric_names].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(out / "summary_mean_sd.csv")
    pd.DataFrame(class_rows).to_csv(out / "per_class_by_fold.csv", index=False)
    pooled, pooled_class, saved_predictions = [], [], []
    fig, axes = plt.subplots(3, 3, figsize=(17, 15), constrained_layout=True)
    for ax, (name, pred) in zip(axes.flat, predictions.items()):
        pooled.append({"model": name, **metrics(y, pred, k)})
        pooled_class.extend(class_metrics(y, pred, classes, name, "pooled"))
        rows = assignments.copy()
        rows["model"], rows["prediction"] = name, [classes[i] for i in pred]
        rows["true_rank"], rows["predicted_rank"] = y, pred
        saved_predictions.append(rows)
        cm = confusion_matrix(y, pred, labels=np.arange(k))
        normalized = cm / cm.sum(axis=1, keepdims=True)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / f"confusion_{name}_counts.csv")
        pd.DataFrame(normalized, index=classes, columns=classes).to_csv(out / f"confusion_{name}_normalized.csv")
        ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        ax.set(title=name, xlabel="Predicted class", ylabel="True class",
               xticks=np.arange(k), yticks=np.arange(k), xticklabels=classes, yticklabels=classes)
        ax.tick_params(axis="x", labelrotation=90)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{normalized[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if normalized[i,j] > 0.5 else "black")
    for ax in list(axes.flat)[len(predictions):]:
        ax.set_visible(False)
    fig.savefig(out / "confusion_matrices.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(pooled).to_csv(out / "pooled_metrics.csv", index=False)
    pd.DataFrame(pooled_class).to_csv(out / "per_class_pooled.csv", index=False)
    pd.concat(saved_predictions, ignore_index=True).to_csv(out / "outer_predictions.csv", index=False)
    paired = []
    for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
        wide = folds.pivot(index="fold", columns="model", values=metric)
        for name in BASE_NAMES:
            delta = 100 * (wide["MLSD"] - wide[name])
            paired.append({"comparator": name, "metric": metric,
                           "MLSD_minus_base_mean_pp": delta.mean(),
                           "difference_sd_pp": delta.std(ddof=1),
                           **{f"fold_{i}_difference_pp": v for i, v in delta.items()}})
    pd.DataFrame(paired).to_csv(out / "paired_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(summary.index, 100 * summary.macro_f1_mean,
           yerr=100 * summary.macro_f1_std, capsize=4,
           color=["#237A57" if name == "MLSD" else "#567C9C" for name in summary.index])
    ax.set(ylabel="Macro-F1 (%)", title="Grouped outer CV: mean ± sample SD")
    fig.savefig(out / "comparison.png", dpi=200)
    plt.close(fig)
    (out / "README_results.txt").write_text(__doc__, encoding="utf-8")
    print("\nOUTER CV RESULTS (mean ± SD; fractions)")
    for name, row in summary.iterrows():
        print(name + ": " + "; ".join(f"{m}={row[m+'_mean']:.4f} ± {row[m+'_std']:.4f}"
                                      for m in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]))
    print(f"\nSaved outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
