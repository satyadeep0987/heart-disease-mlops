# Google Cloud deployment plan (later phases)

**Selected machine:** Apple M1 / 8 GB RAM / 60 GB free. Python 3.14 system, Python 3.12 project (`.venv`).

## Service mapping
| Responsibility | Service |
|---|---|
| Git hosting + build/tests | GitHub + GitHub Actions |
| Authentication for GitHub deployment | Workload Identity Federation (OIDC; no JSON keys) |
| Container images | Artifact Registry (Docker repository) |
| Managed Kubernetes | GKE Autopilot |
| Application exposure | Kubernetes LoadBalancer Service |
| Logs + metrics | Cloud Logging, Cloud Monitoring; `/metrics` for Prometheus |
| Experiments | local MLflow for assignment; remote tracking optional |

## Deployment pipeline
1. PR: Ruff, Pytest, deterministic training/evaluation, model quality gate.
2. `main`: build linux/amd64 Docker image on a GitHub Linux runner; smoke test.
3. Authenticate to Google Cloud using GitHub OIDC Workload Identity Federation.
4. Push immutable SHA-tagged image into Artifact Registry.
5. Deploy manifest to GKE Autopilot and check rollout/readiness and `/predict`.
6. Capture screenshots, logs and report evidence; teardown cloud resources afterward.

## Deployment prerequisites (not required in Phase 1)
- Google Cloud project with billing enabled and budget alerts.
- Required APIs, IAM roles, Artifact Registry repository, and GKE cluster.
- GitHub workload identity provider and narrowly scoped service account.
- **Cost warning:** Cluster management free-tier credits do not cover pod compute, network/LoadBalancer and other cloud charges.

## Resource-conscious defaults
- Keep cluster off until deployment exercise; do not install a local Kubernetes cluster.
- Use a small inference workload (initially one pod); demonstrate multiple replicas only if desired and affordable.
- No protected health information. Data is a public educational dataset.
