# CI/CD Pipeline — Maritime Navigation AI System

## Overview

This project uses **GitHub Actions** for continuous integration. Every push to
`fatemabranch` or `main`, and every pull request targeting those branches,
triggers a suite of automated checks.

**Important:** The CI pipeline is *validation only*. No code is deployed, no
Docker images are pushed, and no cloud resources are modified.

---

## Workflow files

| File | Purpose |
|---|---|
| `.github/workflows/ci.yml` | Main pipeline — lint, test, Docker config, Streamlit syntax |
| `.github/workflows/docker-build.yml` | Build Docker images without pushing |
| `.github/workflows/code-quality.yml` | Deep quality checks: black, flake8, credentials scan, import validation |

---

## Status badges

Add these to the top of `README.md` after the repository is on GitHub:

```markdown
![CI](https://github.com/{owner}/{repo}/actions/workflows/ci.yml/badge.svg?branch=fatemabranch)
![Docker Build](https://github.com/{owner}/{repo}/actions/workflows/docker-build.yml/badge.svg?branch=fatemabranch)
![Code Quality](https://github.com/{owner}/{repo}/actions/workflows/code-quality.yml/badge.svg?branch=fatemabranch)
```

Replace `{owner}` and `{repo}` with the actual GitHub username and repository name.

---

## What each job checks

### `ci.yml`

#### `python-lint`
- **flake8** — PEP 8 compliance, syntax errors, undefined names
- **black** — consistent code formatting (line length 120)
- Covers: `src/`, `api/`, `mlops/`

#### `frontend-lint`
- **ESLint** via `npm run lint`
- Skipped automatically if `frontend/package.json` does not exist

#### `python-tests`
- **pytest** on `tests/test_api.py` and `tests/test_ml.py`
- API tests skip if `localhost:8000` is unreachable
- ML tests skip if `.pkl` model files are not present
- Heavy deps (sklearn, xgboost) are installed but failures are non-fatal

#### `docker-validate`
- `docker compose config --quiet` on `docker-compose.yml`
- Also validates `docker/docker-compose.mlflow.yml` if it exists

#### `streamlit-syntax`
- `python -m py_compile` on all files under `src/dashboard/`

---

### `docker-build.yml`

#### `compose-validate`
- Parses `docker-compose.yml` and the MLflow override with Docker Compose

#### `dockerfile-check`
- Extracts all `build.dockerfile` paths from `docker-compose.yml`
- Verifies each referenced Dockerfile exists on disk

#### `docker-build`
- Runs `docker compose build --no-cache` for all services
- Images are built locally and discarded — never pushed

---

### `code-quality.yml`

#### `black`
- Fails if any Python file does not match black's expected format

#### `flake8`
- Style and error detection; prints statistics

#### `credentials-scan`
- Heuristic grep for `password =`, `secret =`, `api_key =`, `token =`
- Excludes `.example` files, test fixtures, and known placeholder values
- Warns but does not fail (avoids false-positive blocking merges)

#### `import-check`
- `py_compile` on every `.py` file — catches syntax errors fast
- Attempts to import `common.config` and `api.main`
- Optional dependencies (mlflow, sklearn, snowflake, etc.) are tolerated

---

## Graceful skipping

Tests and checks are designed to skip — not fail — when services are
unavailable in CI:

| Scenario | Behaviour |
|---|---|
| API server not running | `test_api.py` skips via `require_api` fixture |
| Model `.pkl` files absent | `test_ml.py` skips via `require_models` fixture |
| sklearn/xgboost not installed | ML tests skip via `require_heavy_deps` fixture |
| `frontend/package.json` absent | ESLint job skips via `if:` condition |
| MLflow override file absent | docker-validate skips that step |

This means CI always turns green on a fresh checkout, and tests only run
when their dependencies are satisfied.

---

## Troubleshooting

### `black` fails
Run locally to auto-fix:
```bash
black --line-length=120 src/ api/ mlops/ tests/
```

### `flake8` fails
Common fixes:
- Line too long → wrap at 120 characters
- Unused import → remove or add `# noqa: F401`
- Undefined name → check imports and spelling

### Docker build fails
- Check that all `build.dockerfile` paths in `docker-compose.yml` exist
- Run `docker compose build` locally to see the full error
- Ensure base images are accessible (network/registry)

### pytest shows unexpected failures
- Run `pytest tests/ -v` locally to see which tests are failing vs skipping
- `SKIP` means a service or file is missing — not a bug
- `FAIL` means logic is broken — investigate the test output

---

## Future deployment structure

The current CI pipeline is validation only. When deployment is needed, the
recommended extension points are:

### AWS
```
.github/workflows/deploy-aws.yml
  → build Docker images
  → push to ECR
  → update ECS task definition
  → trigger ECS rolling deployment
```

### Azure
```
.github/workflows/deploy-azure.yml
  → build Docker images
  → push to ACR
  → deploy to Azure Container Apps or AKS
```

### Kubernetes
```
.github/workflows/deploy-k8s.yml
  → build and push images
  → helm upgrade --install maritime ./helm/maritime
  → kubectl rollout status
```

All deployment workflows should:
1. Be triggered only from `main` (not feature branches)
2. Require manual approval for production environments
3. Run the full CI pipeline as a prerequisite job
4. Never store credentials in workflow files — use GitHub Secrets

---

## GitHub Secrets required for future deployment

| Secret name | Purpose |
|---|---|
| `DOCKER_REGISTRY_USER` | Container registry username |
| `DOCKER_REGISTRY_TOKEN` | Container registry password/token |
| `AWS_ACCESS_KEY_ID` | AWS deployment credentials |
| `AWS_SECRET_ACCESS_KEY` | AWS deployment credentials |
| `SNOWFLAKE_PASSWORD` | Snowflake analytics warehouse |
| `MLFLOW_TRACKING_URI` | Remote MLflow server URL |

Add secrets via: **Settings → Secrets and variables → Actions → New repository secret**
