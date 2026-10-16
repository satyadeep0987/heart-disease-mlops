
"""Pydantic request and response schemas for model inference."""

import math

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.train import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class PatientFeatures(BaseModel):
    """Validate UCI Heart Disease model input."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    age: float = Field(ge=1, le=120)
    trestbps: float | None = Field(default=None, gt=0, le=350)
    chol: float | None = Field(default=None, gt=0, le=800)
    thalach: float | None = Field(default=None, gt=0, le=250)
    oldpeak: float | None = Field(default=None, ge=0, le=10)

    sex: int | None = Field(default=None, ge=0, le=1)
    cp: int | None = Field(default=None, ge=1, le=4)
    fbs: int | None = Field(default=None, ge=0, le=1)
    restecg: int | None = Field(default=None, ge=0, le=2)
    exang: int | None = Field(default=None, ge=0, le=1)
    slope: int | None = Field(default=None, ge=1, le=3)
    ca: int | None = Field(default=None, ge=0, le=3)
    thal: int | None = Field(default=None)

    @model_validator(mode="after")
    def validate_thal(self):
        """Validate the original UCI thal encoding."""
        if self.thal is not None and self.thal not in (3, 6, 7):
            raise ValueError("thal must be 3, 6, or 7")
        return self

    @model_validator(mode="after")
    def validate_finite_numbers(self):
        """Reject NaN and infinite floating-point inputs."""
        for column in NUMERIC_FEATURES:
            value = getattr(self, column)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{column} must be finite")
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Convert validated JSON input into model feature order."""
        values = self.model_dump()

        return pd.DataFrame(
            [{column: values[column] for column in FEATURE_ORDER}],
            columns=FEATURE_ORDER,
        ).astype("float64")


class PredictionResponse(BaseModel):
    """Contract for a successful model prediction."""

    prediction: int = Field(ge=0, le=1)
    probability_no_disease: float = Field(ge=0, le=1)
    probability_disease: float = Field(ge=0, le=1)
    model_version: str
    model_name: str


class HealthResponse(BaseModel):
    status: str
    service: str
