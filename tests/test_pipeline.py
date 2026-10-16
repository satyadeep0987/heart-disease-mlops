
"""Unit tests for data pipeline orchestration."""

import hashlib
import json
from unittest.mock import patch

import pytest

from src.data.pipeline import calculate_sha256, run_pipeline


def test_calculate_sha256(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"hello")

    expected = hashlib.sha256(b"hello").hexdigest()

    assert calculate_sha256(path) == expected


def test_pipeline_success(tmp_path):
    raw = tmp_path / "raw.csv"
    processed = tmp_path / "processed.csv"
    figures = tmp_path / "figures"
    manifest_path = tmp_path / "manifest.json"

    def fake_download(output_path, force=False):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("raw dataset", encoding="utf-8")
        return output_path

    def fake_process(input_path, output_path):
        output_path.write_text("processed dataset", encoding="utf-8")
        return output_path

    def fake_eda(input_path, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "total_rows": 2,
            "feature_count": 13,
            "class_distribution": {"0": 1, "1": 1},
        }

    with (
        patch(
            "src.data.pipeline.download_dataset",
            side_effect=fake_download,
        ),
        patch(
            "src.data.pipeline.validate_csv",
            return_value={"rows": 2},
        ),
        patch(
            "src.data.pipeline.process_dataset",
            side_effect=fake_process,
        ),
        patch(
            "src.data.pipeline.run_eda",
            side_effect=fake_eda,
        ),
    ):
        result = run_pipeline(
            raw_path=raw,
            processed_path=processed,
            figures_path=figures,
            manifest_path=manifest_path,
        )

    assert result["status"] == "success"
    assert result["processed_data"]["features"] == 13
    assert manifest_path.exists()

    saved = json.loads(manifest_path.read_text())
    assert saved["processed_data"]["sha256"] == calculate_sha256(processed)


def test_pipeline_fails_on_invalid_data(tmp_path):
    raw = tmp_path / "raw.csv"
    manifest = tmp_path / "manifest.json"

    raw.write_text("invalid data", encoding="utf-8")

    with (
        patch(
            "src.data.pipeline.download_dataset",
            return_value=raw,
        ),
        patch(
            "src.data.pipeline.validate_csv",
            side_effect=ValueError("Invalid dataset"),
        ),
        patch("src.data.pipeline.process_dataset") as process,
    ):
        with pytest.raises(ValueError, match="Invalid dataset"):
            run_pipeline(
                raw_path=raw,
                manifest_path=manifest,
            )

        process.assert_not_called()

    assert not manifest.exists()
