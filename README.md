# Heart Disease Prediction — MLOps Assignment 01

Production-style learning project for UCI Heart Disease classification. **Phase 1 is scaffolding only**: this ZIP has no downloaded dataset, trained model, live API, CI/CD, or cloud resources yet.

## Architecture roadmap
UCI data → data validation → EDA → sklearn Pipeline → Logistic Regression / Random Forest → MLflow → FastAPI → Docker → GitHub Actions → Google Artifact Registry → GKE Autopilot → metrics/logs.

## MacBook M1 (8 GB RAM, 60 GB free) — lightweight setup

**Install now:** Python 3.11 or 3.12, Git, VS Code. Do **not** install Docker Desktop, local Kubernetes, MLflow server, or JupyterLab for Phase 1. Google Cloud resources are not needed until deployment.

In Terminal (from the project root):

```bash
uname -m           # should show arm64 on M1
df -h /
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m scripts.check_setup
python -m pytest -q
python -m ruff check .
```

If Python 3.12 is not available, install it via `brew install python@3.12`. Keep system Python 3.14 unchanged. Always use the `.venv` interpreter.

**Install only when needed in later phases:**

```bash
# Model experimentation / plots (Phase 2–4)
python -m pip install -r requirements-ml.txt

# Browser-based EDA notebook, optional (Phase 2)
python -m pip install -r requirements-notebook.txt
```

For a memory-conscious notebook workflow, use VS Code with its Python/Jupyter extensions instead of launching several notebook servers, and close Chrome tabs during model training. Keep cross-validation parallelism at `n_jobs=1` (or at most 2) on an 8 GB M1. Do not run Docker + notebook server + MLflow UI simultaneously when not needed.

## GitHub setup
Create an **empty** GitHub repository named `heart-disease-mlops`, then:

```bash
git init
git branch -M main
git add .
git commit -m "chore: initialize M1-friendly MLOps scaffold"
git remote add origin https://github.com/YOUR_USERNAME/heart-disease-mlops.git
git push -u origin main
```

Replace `YOUR_USERNAME`. GitHub authentication must already be configured (SSH or GitHub CLI/credential manager).

## Folder responsibilities
- `src/data/`: dataset download, schema checks and preprocessing (implemented Phase 2–3)
- `src/models/`: model training, evaluation, inference (Phase 3–4)
- `notebooks/`: EDA (Phase 2)
- `api/`: FastAPI endpoints (Phase 5)
- `tests/`: checks for the source code (expanded phase by phase)
- `.github/workflows/`: CI and CD (Phase 8+)
- `deployment/kubernetes/`: GKE manifests (Phase 10)
- `scripts/`: developer setup / smoke tests
- `reports/`: chart and report outputs

## Disk and memory strategy
- Only create one `.venv` per project; keep MLflow/Jupyter as optional install groups.
- Keep training and model selection on Mac; let GitHub Actions build the Linux AMD64 Docker image and Google Cloud host GKE.
- Defer Docker Desktop until Phase 7; on an 8 GB M1, start with **2 GB RAM**, **1–2 CPUs** and **10–15 GB disk limit** if available in your Docker Desktop version. Tune only if necessary.
- Use Docker `linux/amd64` images for GKE nodes running AMD64. Building them in GitHub Actions avoids ARM emulation overhead on the Mac.
- Keep 15–20 GB free for macOS swap, updates and development. Never run `docker system prune --volumes` casually—it may delete persisted data.

## Current Phase 1 exit criteria
- Python 3.12 `arm64`, `.venv` activated
- `python -m scripts.check_setup` prints `Setup OK`
- `python -m pytest -q` passes
- `python -m ruff check .` passes
- Initial commit pushed to GitHub

Next: Phase 2 — UCI dataset acquisition and validation plus EDA.

## Google Cloud architecture (planned, not deployed)
- GitHub Actions (OIDC / Workload Identity Federation) → Google Artifact Registry → GKE Autopilot → external LoadBalancer → FastAPI.
- Cloud Logging + Cloud Monitoring for operational monitoring; Prometheus metrics from `/metrics`, dashboard added later.
- Local MLflow during development; model artifacts are versioned and promoted only after quality gates pass.
- **Security:** never commit cloud keys or patient data. Use GitHub-to-Google Cloud Workload Identity Federation rather than a long-lived service account JSON key.
- **Costs:** GKE is not universally free; clusters, pods, networking and load balancers may incur charges. Set billing budgets/alerts and delete unused cluster and load balancer resources.
- Google Cloud setup and credentials intentionally deferred until deployment phases.
See `docs/GCP_DEPLOYMENT_PLAN.md` for the roadmap.
