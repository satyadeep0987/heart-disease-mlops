
"""Generate reproducible EDA reports for the Heart Disease dataset."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_clean.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "figures"

NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

TARGET = "target"


def save_figure(output_path: Path) -> None:
    """Save and close a figure to reduce memory consumption."""
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot class balance."""
    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x=TARGET,
        order=[0, 1],
    )

    plt.title("Heart Disease Class Distribution")
    plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
    plt.ylabel("Number of Patients")

    save_figure(output_dir / "target_distribution.png")


def plot_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot distributions of numerical features."""
    axes = df[NUMERIC_FEATURES].hist(
        bins=20,
        figsize=(12, 8),
        edgecolor="black",
    )

    figure = axes.flat[0].figure
    figure.suptitle("Numerical Feature Distributions", fontsize=15)
    figure.tight_layout()

    save_figure(output_dir / "feature_histograms.png")


def plot_correlation(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot Pearson correlations for numerical features and target."""
    columns = [*NUMERIC_FEATURES, TARGET]
    correlation = df[columns].corr()

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
    )

    plt.title("Numerical Feature Correlation Heatmap")

    save_figure(output_dir / "correlation_heatmap.png")


def plot_missing_values(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot missing-value counts by feature."""
    missing = df.isna().sum().sort_values(ascending=False)

    plt.figure(figsize=(11, 5))
    missing.plot(kind="bar")

    plt.title("Missing Values by Feature")
    plt.xlabel("Features")
    plt.ylabel("Missing Count")
    plt.xticks(rotation=45, ha="right")

    save_figure(output_dir / "missing_values.png")


def plot_age_vs_target(df: pd.DataFrame, output_dir: Path) -> None:
    """Compare patient age distributions by target class."""
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x=TARGET,
        y="age",
        order=[0, 1],
    )

    plt.title("Age Distribution by Heart Disease Status")
    plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
    plt.ylabel("Age")

    save_figure(output_dir / "age_vs_target.png")


def generate_summary(df: pd.DataFrame, output_dir: Path) -> dict:
    """Generate machine-readable dataset summary."""
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "feature_count": int(len(df.columns) - 1),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": {
            str(label): int(count)
            for label, count in df[TARGET].value_counts().items()
        },
        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "numeric_statistics": json.loads(
            df[NUMERIC_FEATURES].describe().to_json()
        ),
    }

    summary_path = output_dir / "eda_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


def run_eda(
    input_path: Path = DATA_PATH,
    output_dir: Path = REPORT_DIR,
) -> dict:
    """Execute all EDA steps."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_csv(input_path)

    required = {*NUMERIC_FEATURES, TARGET}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required EDA columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Cannot generate EDA for empty dataset")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values")

    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    plot_target_distribution(df, output_dir)
    plot_histograms(df, output_dir)
    plot_correlation(df, output_dir)
    plot_missing_values(df, output_dir)
    plot_age_vs_target(df, output_dir)

    summary = generate_summary(df, output_dir)

    logger.info("EDA artifacts generated at: %s", output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    summary = run_eda(args.input, args.output)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
