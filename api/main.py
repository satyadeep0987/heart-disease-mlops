
"""FastAPI application for Heart Disease model inference."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.model_loader import LoadedModel, load_model
from api.schemas import HealthResponse, PatientFeatures, PredictionResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the application starts."""
    logger.info("Starting Heart Disease Prediction API")

    app.state.model = load_model()

    logger.info(
        "Model ready: %s version %s",
        app.state.model.name,
        app.state.model.version,
    )

    yield

    logger.info("Shutting down Heart Disease Prediction API")
    app.state.model = None


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Educational ML prediction service with MLOps integration",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe independent of model readiness."""
    return HealthResponse(
        status="healthy",
        service="heart-disease-api",
    )


@app.get("/ready")
def ready(request: Request):
    """Readiness probe for the loaded model."""
    model = getattr(request.app.state, "model", None)

    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready"},
        )

    return {
        "status": "ready",
        "model_name": model.name,
        "model_version": model.version,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request):
    """Predict heart disease class using the loaded pipeline."""
    model: LoadedModel | None = getattr(
        request.app.state,
        "model",
        None,
    )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available",
        )

    try:
        features = patient.to_dataframe()

        prediction = int(model.pipeline.predict(features)[0])
        probabilities = model.pipeline.predict_proba(features)[0]

        return PredictionResponse(
            prediction=prediction,
            probability_no_disease=float(probabilities[0]),
            probability_disease=float(probabilities[1]),
            model_version=model.version,
            model_name=model.name,
        )

    except Exception:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Model inference failed",
        ) from None
