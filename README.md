# Rockwool Water Detection

Spectral data and Python workflows for classifying moisture levels in horticultural rockwool. The repository includes the Rockwool Water Index (RWI), conventional classifiers, a Multi-learner Spectral Drought (MLSD) stacking model, a full-spectrum 1D-CNN, and preprocessing sensitivity analysis.

## Data and scripts

The files in [`Data/`](Data/) contain **1,100 spectra from 100 physical samples**, each measured at 11 moisture levels. Each CSV has 2,301 wavelength rows spanning **200–2500 nm at 1 nm intervals**, a `Wavelength` column, and spectrum columns such as `dry-1` and `sub9-1`. The numeric suffix identifies the physical sample across moisture levels.

| File | Purpose |
| --- | --- |
| `Data/All_classes_raw_4dp.csv` | Unsmoothed spectra, rounded to four decimal places |
| `Data/All_classes_smoothed_4dp.csv` | Smoothed spectra, rounded to four decimal places |
| `Codes/rockwool_grouped_cv.py` | Compare six standalone classifiers and MLSD using sample-grouped five-fold CV |
| `Codes/MLSD_60_20_20.py` | Develop and evaluate MLSD with a fixed sample-level 60/20/20 split |
| `Codes/CNN1D_60_20_20.py` | Develop and evaluate the full-spectrum 1D-CNN using the same split |
| `Codes/MLSD_preprocessing_sensitivity_60_20.py` | Compare seven preprocessing methods on the training and validation partitions |
| `Codes/plot_class_average_spectra.py` | Export class-average spectra, standard deviations, and PNG/PDF plots |

Class order runs from `dry`, through `sub9`, to `sub0` with increasing moisture. Several scripts retain legacy 0–50% VWC display mappings; check `CLASS_TO_VWC` and `VWC_BY_CLASS` against the verified experimental values before publishing labels. The shared CSVs retain all observations, but rounding may affect numerical reproducibility relative to full-precision data.

## Setup

Use Python 3.10 or newer in a dedicated environment. Run the following from the repository root:

```bash
python -m pip install numpy pandas scipy "scikit-learn>=1.2" matplotlib joblib
```

For the CNN, also install PyTorch:

```bash
python -m pip install torch
```

The CNN uses CUDA when available and otherwise runs on the CPU. The repository does not currently pin a complete environment; retain the package versions recorded in each run when comparing results.

## Quick start: grouped cross-validation

This workflow runs directly with the included data and does not require a split manifest:

```bash
python Codes/rockwool_grouped_cv.py --csv Data/All_classes_smoothed_4dp.csv
```

It compares PLS-DA, PCA–LDA, RBF-SVM, KNN, ANN, RF, and MLSD. The ensemble combines RF, PLS-DA, and PCA–LDA probabilities using a logistic-regression meta-classifier. All spectra from one physical sample stay together in both the outer evaluation and internal stacking folds.

The default input is **RWI alone**, using `(570, 1560, 1940) nm`:

```text
RWI = (Ra - (Rb - Rc)) / (Ra + (Rb - Rc))
```

To add the same ±5 nm spectral windows to every model:

```bash
python Codes/rockwool_grouped_cv.py --csv Data/All_classes_smoothed_4dp.csv --window-nm 5
```

On the supplied grid, these options provide 1 and 34 input features, respectively. Use `--wavelengths 560 1600 1920` to specify the alternative triplet, or `--help` for other options.

Results are saved in a timestamped `rockwool_cv_*` folder. Start with `summary_mean_sd.csv` for fold means and sample SDs, `pooled_metrics.csv` for pooled held-out performance, and `outer_predictions.csv` for individual predictions. Confusion matrices, comparison plots, per-class metrics, ordinal metrics, and split audits are also saved.

**This is fixed-setting development CV:** it does not search wavelengths or tune hyperparameters, and it does not remove bias from choices previously made using the same dataset. Fold SD is not a confidence interval.

## MLSD and CNN: 60/20/20 workflow

These scripts require `physical_sample_split_60_20_20_seed42.csv`, which can be modified based on your preference. Place the original experiment's manifest in `Codes/`. It must contain the columns `physical_sample_id` and `split`, one row per sample, and exactly 60 `train`, 20 `validation`, and 20 `test` assignments. Reuse the same manifest across models; the seed alone does not reconstruct the original assignments.

Before the first run, edit the settings near the top of **both** training scripts:

```python
DATA_CSV = SCRIPT_DIR.parent / "Data" / "All_classes_smoothed_4dp.csv"
RUN_MODE = "development"
```

The checked-in scripts currently have `RUN_MODE = "final_test"`; development must first produce their frozen configuration files.

```bash
python "Codes/MLSD_60_20_20.py"
python Codes/CNN1D_60_20_20.py
```

MLSD uses RWI plus **±1 nm windows around `(560, 1600, 1920) nm`**, giving 10 spectral/index inputs. Its three learners produce 33 class-probability inputs for the meta-classifier. Development includes grouped base-learner tuning within stacking folds and evaluation on the 20 validation samples. The CNN uses the full spectrum and selects its training settings and epoch using validation performance.

After all development decisions are fixed, set `RUN_MODE = "final_test"` and rerun each script. They refit using the combined 80 development samples and evaluate the 20 test samples. Keep the data, manifest, frozen configurations, and completion markers unchanged between stages. Outputs include predictions, class reports, confusion matrices, fitted models, and physical-sample bootstrap confidence intervals under each script's `OUTPUT_ROOT`.

## Preprocessing and spectral plots

For preprocessing sensitivity, use the **unsmoothed** CSV and the same split manifest. Set:

```python
SOURCE_DATA_CSV = SCRIPT_DIR.parent / "Data" / "All_classes_raw_4dp.csv"
```

Then run:

```bash
python Codes/MLSD_preprocessing_sensitivity_60_20.py
```

The script compares no preprocessing, a three-point moving average, SNV, MSC, Savitzky–Golay smoothing, and first and second derivatives. It uses 60 training and 20 validation samples; the test partition is not scored. The main output is `preprocessing_comparison_macro_metrics.csv` in the timestamped results folder.

The checked-in Savitzky–Golay settings are a **3-point window and polynomial order 2**, not the 11-point window described in the earlier draft. This combination leaves the smoothing-only signal effectively unchanged. Choose and document an appropriate longer odd window if smoothing is intended, before comparing methods.

For class-average plots, update the plotting script's input:

```python
INPUT_CSV = Path(__file__).resolve().parent.parent / "Data" / "All_classes_raw_4dp.csv"
```

Run `python Codes/plot_class_average_spectra.py`. Set `SHOW_STANDARD_DEVIATION = True` to display variability bands.

## License

See [LICENSE](LICENSE) for the GNU General Public License v3.0.
