# Docker Compose → Kubernetes Migration Guide
## Maritime Navigation AI System

This guide walks through migrating the maritime AIS system from Docker Compose
to Kubernetes. Read all steps before executing any commands.

---

## Prerequisites

| Tool | Minimum version | Purpose |
|---|---|---|
| `kubectl` | 1.28+ | Apply manifests, inspect cluster state |
| `helm` | 3.12+ | Optional — for Ingress controllers, cert-manager |
| A running cluster | — | EKS / GKE / AKS / Minikube |
| Metrics Server | — | Required for HPA (`kubectl top`) |
| A container registry | — | ECR / GCR / ACR / GHCR |

---

## Step 0 — Required changes before any `kubectl apply`

Do these before running a single command:

### 0a. Build and push images

The manifests reference placeholder image names. Build each service and push
to your registry:

```bash
# Example for GHCR
export REGISTRY=ghcr.io/your-org

docker build -t $REGISTRY/maritime-api:latest         -f api/Dockerfile .
docker build -t $REGISTRY/maritime-streamlit:latest   -f src/dashboard/Dockerfile .
docker build -t $REGISTRY/maritime-live-scorer:latest -f src/ml/Dockerfile .
docker build -t $REGISTRY/maritime-frontend:latest    -f frontend/Dockerfile .

docker push $REGISTRY/maritime-api:latest
docker push $REGISTRY/maritime-streamlit:latest
docker push $REGISTRY/maritime-live-scorer:latest
docker push $REGISTRY/maritime-frontend:latest
```

### 0b. Update image references in deployments

Replace every `REPLACE_WITH_YOUR_IMAGE` in `k8s/deployments/`:

```bash
# Review all placeholder image names
grep -r "REPLACE_WITH_YOUR_IMAGE" k8s/deployments/
```

### 0c. Fill in secrets

```bash
# Copy the template — DO NOT commit the filled-in file
cp k8s/secrets/secrets-template.yaml k8s/secrets/secrets.yaml

# Base64-encode your values
echo -n "your-postgres-password" | base64
# Paste the output into secrets.yaml

# Verify the file is git-ignored
echo "k8s/secrets/secrets.yaml" >> .gitignore
```

### 0d. Choose a storageClassName

Find available storage classes on your cluster:

```bash
kubectl get storageclass
```

Edit `k8s/deployments/postgres-deployment.yaml` and
`k8s/deployments/kafka-deployment.yaml` to uncomment and set
`storageClassName`.

### 0e. Set REACT_APP_API_URL

In `k8s/deployments/frontend-deployment.yaml`, replace
`REPLACE_WITH_YOUR_API_HOSTNAME` with the external hostname or IP of
the FastAPI LoadBalancer service. You may need to deploy FastAPI first
and get its external IP, then update and redeploy the frontend.

---

## Step 1 — Create the namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

Verify:

```bash
kubectl get namespace maritime-system
```

---

## Step 2 — Apply ConfigMaps

```bash
kubectl apply -f k8s/configmaps/ -n maritime-system
```

Verify:

```bash
kubectl describe configmap maritime-app-config -n maritime-system
```

---

## Step 3 — Apply Secrets

```bash
# Apply YOUR filled-in secrets file, not the template
kubectl apply -f k8s/secrets/secrets.yaml -n maritime-system
```

Verify (values will be redacted):

```bash
kubectl get secret -n maritime-system
```

---

## Step 4 — Deploy stateful services first

PostgreSQL and Kafka must be running and healthy before the application
services start:

```bash
kubectl apply -f k8s/deployments/postgres-deployment.yaml -n maritime-system
kubectl apply -f k8s/deployments/kafka-deployment.yaml    -n maritime-system
```

Wait for readiness:

```bash
kubectl rollout status statefulset/postgres -n maritime-system
kubectl rollout status statefulset/zookeeper -n maritime-system
kubectl rollout status statefulset/kafka     -n maritime-system

# Watch pods
kubectl get pods -n maritime-system -w
```

---

## Step 5 — Deploy application services

```bash
kubectl apply -f k8s/deployments/fastapi-deployment.yaml      -n maritime-system
kubectl apply -f k8s/deployments/streamlit-deployment.yaml    -n maritime-system
kubectl apply -f k8s/deployments/live-scorer-deployment.yaml  -n maritime-system
kubectl apply -f k8s/deployments/frontend-deployment.yaml     -n maritime-system
```

Wait for rollout:

```bash
kubectl rollout status deployment/fastapi     -n maritime-system
kubectl rollout status deployment/streamlit   -n maritime-system
kubectl rollout status deployment/live-scorer -n maritime-system
kubectl rollout status deployment/frontend    -n maritime-system
```

---

## Step 6 — Apply Services

```bash
kubectl apply -f k8s/services/ -n maritime-system
```

Get external IPs (may take a few minutes on cloud providers):

```bash
kubectl get service -n maritime-system
```

---

## Step 7 — Apply HPA

HPA requires the Metrics Server. Install it first if not already present:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Verify
kubectl top nodes
```

Then apply HPA:

```bash
kubectl apply -f k8s/hpa/ -n maritime-system
```

Verify:

```bash
kubectl get hpa -n maritime-system
```

---

## Apply all at once (after completing Step 0)

The `deploy.sh` helper applies everything in the correct order:

```bash
# Dry run (default — no changes applied)
./k8s/deploy.sh

# Actually apply
./k8s/deploy.sh apply
```

---

## Verifying the migration

```bash
# All pods running
kubectl get pods -n maritime-system

# Check FastAPI health
kubectl port-forward service/fastapi-service 8000:8000 -n maritime-system
curl http://localhost:8000/health

# Check Streamlit
kubectl port-forward service/streamlit-service 8501:8501 -n maritime-system
# Open http://localhost:8501

# Check logs
kubectl logs deployment/fastapi     -n maritime-system --tail=50
kubectl logs deployment/live-scorer -n maritime-system --tail=50
```

---

## Rolling back

```bash
# Roll back a single deployment
kubectl rollout undo deployment/fastapi -n maritime-system

# Roll back to a specific revision
kubectl rollout history deployment/fastapi -n maritime-system
kubectl rollout undo deployment/fastapi --to-revision=2 -n maritime-system
```

---

## Teardown

```bash
# Delete all resources in the namespace (keeps PVCs by default)
kubectl delete namespace maritime-system

# To also delete PersistentVolumeClaims (DATA LOSS):
kubectl delete pvc --all -n maritime-system
```
