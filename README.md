# Heart Disease Prediction — End-to-End MLOps

An end-to-end Machine Learning Operations (MLOps) project for heart disease classification, covering data ingestion, preprocessing, model experimentation, model registry, automated testing, containerization, cloud deployment, and monitoring.

**GitHub Repository:** https://github.com/satyadeep0987/heart-disease-mlops

**Cloud Platform:** Google Cloud Platform (GCP)

**Deployment Platform:** Google Kubernetes Engine (GKE Autopilot)

> **Disclaimer:** This project is intended for educational and demonstration purposes. Its predictions are not intended for clinical diagnosis or treatment decisions.

## 1. Project Overview

This project demonstrates the complete lifecycle of a machine learning application:

1. Download the UCI Heart Disease dataset.
2. Validate, preprocess, and analyze the data.
3. Train and compare Logistic Regression and Random Forest models.
4. Perform hyperparameter tuning and cross-validation.
5. Track experiments and register models using MLflow.
6. Package the selected model as a reproducible artifact.
7. Serve predictions through a FastAPI REST API.
8. Automate testing, model training, and Docker builds with GitHub Actions.
9. Publish verified Docker images to Google Artifact Registry.
10. Deploy the application on GKE.
11. Monitor API performance and infrastructure health using Prometheus metrics and Google Cloud Monitoring.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.12 |
| Dataset | UCI Heart Disease |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Experiment tracking | MLflow |
| Model serialization | Joblib |
| API framework | FastAPI, Pydantic |
| Unit testing | Pytest |
| Linting | Ruff |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Container registry | Google Artifact Registry |
| Orchestration | Kubernetes, GKE Autopilot |
| Monitoring | Prometheus, Google Cloud Monitoring |
| Logging | Google Cloud Logging |
| Authentication | GitHub OIDC and Workload Identity Federation |

## 3. System Architecture

UCI Dataset → Data Validation → Preprocessing & EDA → Model Training → Hyperparameter Tuning → Quality Gate → Model Packaging → FastAPI → Docker → GitHub Actions → Google Artifact Registry → GKE → Cloud Monitoring

MLflow separately records model experiments, evaluation metrics, artifacts, and registered model versions.

## 4. Project Structure

    heart-disease-mlops/
    ├── .github/
    │   └── workflows/
    │       ├── ci.yml
    │       ├── model-training.yml
    │       ├── docker-ci.yml
    │       └── gcp-auth-test.yml
    ├── api/
    │   ├── main.py
    │   ├── schemas.py
    │   ├── model_loader.py
    │   └── observability.py
    ├── src/
    │   ├── data/
    │   │   ├── download.py
    │   │   ├── validation.py
    │   │   ├── preprocessing.py
    │   │   ├── eda.py
    │   │   └── pipeline.py
    │   └── models/
    │       ├── train.py
    │       ├── evaluate.py
    │       ├── tune.py
    │       ├── package.py
    │       ├── tracking.py
    │       └── promote.py
    ├── k8s/
    │   ├── deployment.yaml
    │   ├── service.yaml
    │   ├── ingress.yaml
    │   └── podmonitoring.yaml
    ├── scripts/
    │   └── check_model_quality.py
    ├── tests/
    ├── data/
    ├── models/
    │   └── production/
    ├── reports/
    ├── Dockerfile
    ├── requirements.txt
    ├── requirements-serving.txt
    └── README.md

Some data, reports, and trained model artifacts are generated at runtime and may be excluded from Git.

## 5. Dataset and Preprocessing

**Source:** UCI Machine Learning Repository, Heart Disease dataset (ID 45).

The application uses 13 input features:

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, and `thal`.

The original target variable `num` is converted into a binary classification label:

- **0:** No recorded heart disease.
- **1:** Heart disease present (`num > 0`).

The preprocessing pipeline performs schema validation, data cleaning, missing-value handling, and feature transformations. Training and preprocessing are combined in Scikit-learn pipelines to reduce data leakage.

## 6. Model Training and Evaluation

Two classification algorithms are evaluated:

**Logistic Regression:** Provides a relatively simple, interpretable baseline.

**Random Forest:** Captures nonlinear relationships and feature interactions.

The implementation includes an 80/20 stratified train-test split, five-fold cross-validation, hyperparameter tuning with GridSearchCV, and classification evaluation.

Evaluation includes ROC-AUC, confusion matrices, and other classification metrics.

An automated quality gate requires the configured minimum cross-validation ROC-AUC of **0.80** before packaging.

The exact evaluation results and selected model should be taken from the generated reports and model metadata rather than hardcoded into this document.

## 7. MLflow Experiment Tracking

MLflow provides experiment tracking and model registration.

The implementation supports experiment logging, candidate model evaluation, model artifacts, and registry aliases.

Example local MLflow server:

    mlflow server \
      --backend-store-uri sqlite:///mlflow.db \
      --default-artifact-root ./mlartifacts \
      --host 127.0.0.1 \
      --port 5000

MLflow interface: http://127.0.0.1:5000

Registered model: `heart-disease-classifier`

Model alias: `candidate`

Production Docker serving defaults to the locally packaged and verified model artifact instead of requiring the MLflow server to remain available.

## 8. Local Setup

**Requirements:** Python 3.12, Git, and Docker.

Clone the repository:

    git clone https://github.com/satyadeep0987/heart-disease-mlops.git
    cd heart-disease-mlops

Create a Python virtual environment:

    python3.12 -m venv .venv
    source .venv/bin/activate

Install dependencies using the requirement files maintained in the repository.

Run the data pipeline:

    python -m src.data.pipeline

Train and tune models:

    python -m src.models.train
    python -m src.models.tune

Package the selected model:

    python -m src.models.package

Run unit tests:

    python -m pytest -q

Run lint checks:

    python -m ruff check .

## 9. FastAPI Model Serving

Start the inference service:

    uvicorn api.main:app --host 127.0.0.1 --port 8000

Available endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Application health |
| `/ready` | GET | Model readiness |
| `/predict` | POST | Heart disease classification |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Interactive API documentation |

### Prediction request example

    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{
        "age": 52,
        "sex": 1,
        "cp": 2,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3
      }'

The response includes the predicted class, class probabilities, and model information.

## 10. Docker Containerization

Build the Docker image after generating the production model artifact:

    docker build -t heart-disease-api:local .

Run locally:

    docker run --rm \
      -p 8000:8000 \
      --memory=1g \
      --cpus=1 \
      heart-disease-api:local

Test:

    curl http://localhost:8000/health

The container uses Python 3.12, a non-root application user, and a packaged model with integrity verification.

## 11. GitHub Actions CI/CD

GitHub Actions automates:

- Static code checks and unit tests.
- Data preparation and model training.
- Hyperparameter tuning and model quality checks.
- Model packaging.
- Docker image building and smoke testing.
- Secure authentication to GCP using Workload Identity Federation.
- Publishing verified images to Artifact Registry.

The cloud publishing workflow runs from the approved `main` branch.

Service-account JSON private keys are not required for GitHub-to-GCP authentication.

## 12. Google Cloud Deployment

**Project:** `heart-disease-mlops-511812`

**Region:** `asia-south1`

**Artifact Registry repository:** `heart-disease-images`

**GKE cluster:** `heart-disease-gke`

**Kubernetes namespace:** `mlops`

**Deployment:** `heart-disease-api`

**Kubernetes Service:** `heart-disease-service`

**Ingress:** `heart-disease-ingress`

The application is deployed through a Kubernetes Deployment, ClusterIP Service, and GKE Ingress.

The Deployment uses readiness, liveness, and startup probes, resource requests and limits, and an immutable container image digest.

Example deployment commands:

    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/ingress.yaml

Check deployment:

    kubectl get deployments,pods,services,ingress -n mlops

For local access without a public load balancer:

    kubectl port-forward \
      -n mlops \
      service/heart-disease-service \
      8000:80

The public endpoint is intended only for temporary educational demonstration until HTTPS and authentication are implemented.

## 13. Monitoring and Observability

The application exposes Prometheus metrics including:

- `heart_api_requests_total`
- `heart_api_request_duration_seconds`

Google Cloud Managed Service for Prometheus collects metrics using `k8s/podmonitoring.yaml`.

The Cloud Monitoring dashboard includes:

1. API request throughput.
2. API P95 response latency.
3. HTTP requests over 15 minutes.
4. GKE container CPU usage.
5. GKE container memory usage.

Google Cloud Logging captures application and container logs.

An uptime check monitors the `/health` endpoint and provides availability alerts.

The application avoids logging raw patient feature payloads.

## 14. Testing and Quality Assurance

The project uses automated tests and runtime checks covering data validation, preprocessing, model training, packaging, model loading, and API behavior.

GitHub Actions verifies that the built Docker container starts successfully, reports readiness, handles prediction requests, and exposes metrics.

The container's model artifact is checked for integrity before loading.

## 15. Security and Limitations

This is a learning and demonstration deployment, not a production clinical application.

Current limitations include:

- No clinical validation or regulatory approval.
- Public HTTP Ingress without user authentication.
- No demonstrated data-drift detection or automatic model retraining based on drift.
- MLflow experiment registry is local rather than a separately hosted production registry.
- The model depends on the UCI dataset and should not be assumed to generalize to clinical populations.

Production use would require stronger security, TLS, identity and access controls, monitoring, privacy assessment, data governance, and clinical validation.

## 16. Cost Management and Cleanup

GKE clusters and load balancers can incur charges while active.

Remove the public Ingress after completing the demonstration:

    kubectl delete -f k8s/ingress.yaml

After collecting all deployment evidence, delete the GKE cluster:

    gcloud container clusters delete heart-disease-gke \
      --project=heart-disease-mlops-511812 \
      --location=asia-south1

Verify that associated billable infrastructure has also been removed. Google Cloud budget alerts do not automatically stop resource charges.

## 17. Conclusion

This project demonstrates an end-to-end MLOps workflow, from dataset ingestion and machine learning experimentation to containerized model inference on Kubernetes.

It combines reproducible training, automated validation, CI/CD, model artifact management, cloud deployment, logging, metrics, and uptime monitoring into a practical machine learning engineering implementation.

**Status:** Core implementation and cloud monitoring completed; final evaluation evidence, documentation review, and demonstration submission remain to be finalized.