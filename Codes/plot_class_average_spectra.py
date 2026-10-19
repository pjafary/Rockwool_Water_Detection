"""Export class-average reflectance spectra as PNG, PDF, and a mean/SD CSV.

Input columns: Wavelength, dry-1, ..., sub9-100. Configure INPUT_CSV below.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


# User settings
INPUT_CSV = Path(__file__).with_name("All_classes_raw.csv")
OUTPUT_DIR = Path(__file__).with_name("class_average_results_raw")

WAVELENGTH_MIN = 200   # Set to None to use the minimum wavelength in the CSV.
WAVELENGTH_MAX = 2500  # Set to None to use the maximum wavelength in the CSV.

SHOW_STANDARD_DEVIATION = False
STD_ALPHA = 0.10
FIGURE_DPI = 600


CLASS_ORDER = ["dry"] + [f"sub{i}" for i in range(9, -1, -1)]
VWC_BY_CLASS = {
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


def identify_class_columns(columns):
    """Return {class_name: [sample columns]} from names such as sub5-37."""
    pattern = re.compile(r"^(dry|sub\d+)-(\d+)$", flags=re.IGNORECASE)
    groups = {}

    for column in columns:
        match = pattern.fullmatch(str(column).strip())
        if match:
            class_name = match.group(1).lower()
            sample_number = int(match.group(2))
            groups.setdefault(class_name, []).append((sample_number, column))

    return {
        class_name: [column for _, column in sorted(items)]
        for class_name, items in groups.items()
    }


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"CSV file not found: {INPUT_CSV}\n"
            "Place this script beside All_classes_smoothed.csv or update "
            "INPUT_CSV near the top of the script."
        )

    data = pd.read_csv(INPUT_CSV, low_memory=False)

    if "Wavelength" not in data.columns:
        raise ValueError("The CSV must contain a column named 'Wavelength'.")

    data["Wavelength"] = pd.to_numeric(data["Wavelength"], errors="coerce")
    data = data.dropna(subset=["Wavelength"]).copy()

    if WAVELENGTH_MIN is not None:
        data = data.loc[data["Wavelength"] >= WAVELENGTH_MIN]
    if WAVELENGTH_MAX is not None:
        data = data.loc[data["Wavelength"] <= WAVELENGTH_MAX]

    data = data.sort_values("Wavelength").reset_index(drop=True)

    class_columns = identify_class_columns(data.columns)
    missing_classes = [name for name in CLASS_ORDER if name not in class_columns]
    if missing_classes:
        raise ValueError(
            "The following expected classes were not detected: "
            + ", ".join(missing_classes)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame({"Wavelength": data["Wavelength"]})

    moisture_cmap = LinearSegmentedColormap.from_list(
        "rockwool_moisture", ["#8C510A", "#DFC27D", "#80CDC1", "#01665E"]
    )
    colors = moisture_cmap(np.linspace(0, 1, len(CLASS_ORDER)))

    fig, ax = plt.subplots(figsize=(11, 6.5))

    print("Detected spectra per class:")
    for color, class_name in zip(colors, CLASS_ORDER):
        columns = class_columns[class_name]
        class_data = data[columns].apply(pd.to_numeric, errors="coerce")

        mean_spectrum = class_data.mean(axis=1, skipna=True)
        std_spectrum = class_data.std(axis=1, skipna=True, ddof=1)

        summary[f"{class_name}_mean"] = mean_spectrum
        summary[f"{class_name}_std"] = std_spectrum

        vwc = VWC_BY_CLASS[class_name]
        label = f"{class_name.capitalize()} ({vwc}% VWC)"
        ax.plot(
            data["Wavelength"],
            mean_spectrum,
            color=color,
            linewidth=1.45,
            label=label,
        )

        if SHOW_STANDARD_DEVIATION:
            ax.fill_between(
                data["Wavelength"].to_numpy(),
                (mean_spectrum - std_spectrum).to_numpy(),
                (mean_spectrum + std_spectrum).to_numpy(),
                color=color,
                alpha=STD_ALPHA,
                linewidth=0,
            )

        print(f"  {class_name:>4}: {len(columns)} spectra")

    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Reflectance (%)", fontsize=12)
    ax.set_title("Mean Reflectance Spectra of Rockwool VWC Classes", fontsize=13)
    ax.grid(True, color="0.88", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()

    png_path = OUTPUT_DIR / "average_spectra_by_vwc.png"
    pdf_path = OUTPUT_DIR / "average_spectra_by_vwc.pdf"
    csv_path = OUTPUT_DIR / "class_average_spectra.csv"

    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    summary.to_csv(csv_path, index=False)
    plt.show()

    print(f"\nSaved figure: {png_path}")
    print(f"Saved vector figure: {pdf_path}")
    print(f"Saved numerical summary: {csv_path}")


if __name__ == "__main__":
    main()
