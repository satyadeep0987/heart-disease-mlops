
"""Validate the selected model against cross-validation quality gates."""

import argparse
import json
import math
from pathlib import Path


def check_model_quality(
    report_path: Path,
    min_roc_auc: float = 0.80,
) -> dict:
    """Fail if the selected model does not meet the required CV ROC-AUC."""

    if not 0 <= min_roc_auc <= 1:
        raise ValueError("ROC-AUC threshold must be between 0 and 1")

    report = json.loads(
        Path(report_path).read_text(encoding="utf-8")
    )

    if report.get("selection_metric") != "cv_roc_auc":
        raise ValueError("Unexpected model selection metric")

    if report.get("holdout_used_for_selection") is not False:
        raise ValueError("Holdout data must not be used for selection")

    selected = report["selected_model"]
    metrics = report["models"][selected]
    score = float(metrics["best_cv_roc_auc"])

    if not math.isfinite(score) or score < min_roc_auc:
        raise ValueError(
            f"Model quality gate failed: ROC-AUC={score:.4f}, "
            f"required={min_roc_auc:.4f}"
        )

    print(
        f"Model quality gate PASSED | "
        f"model={selected} | CV ROC-AUC={score:.4f}"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/model_tuning.json"),
    )
    parser.add_argument(
        "--min-roc-auc",
        type=float,
        default=0.80,
    )
    args = parser.parse_args()

    check_model_quality(args.report, args.min_roc_auc)


if __name__ == "__main__":
    main()
