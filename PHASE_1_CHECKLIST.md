# Phase 1 checklist — M1 8 GB / 60 GB free

- [ ] `uname -m` reports `arm64`
- [ ] Python 3.11 or 3.12 available (`python3.12 --version` / `python3.11 --version`)
- [ ] Extract ZIP, open project in VS Code
- [ ] Create `.venv` with supported Python, activate it
- [ ] `python -m pip install -r requirements-dev.txt`
- [ ] `python -m scripts.check_setup` prints `Setup OK`
- [ ] `python -m pytest -q` passes
- [ ] `python -m ruff check .` passes
- [ ] Create GitHub repo and push initial commit
- [ ] Capture screenshots of test results and commit for final report

**Deferred:** MLflow, JupyterLab, Docker Desktop, Kubernetes, Google Cloud resources. These aren't required to complete Phase 1.
