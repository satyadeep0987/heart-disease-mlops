
"""Tests for reproducible dataset acquisition."""

from unittest.mock import patch

import pandas as pd

from src.data.download import download_dataset


def test_download_and_cache(tmp_path):
    output = tmp_path / "heart.csv"

    class FakeData:
        features = pd.DataFrame({"age": [45, 60]})
        targets = pd.DataFrame({"num": [0, 2]})

    class FakeDataset:
        data = FakeData()

    with patch(
        "src.data.download.fetch_ucirepo",
        return_value=FakeDataset(),
    ) as fetch:
        first_path = download_dataset(output)
        second_path = download_dataset(output)

    assert first_path == output
    assert second_path == output
    assert output.exists()

    result = pd.read_csv(output)
    assert result.shape == (2, 2)
    assert result["num"].tolist() == [0, 2]

    # Second invocation must reuse the cached CSV.
    assert fetch.call_count == 1
