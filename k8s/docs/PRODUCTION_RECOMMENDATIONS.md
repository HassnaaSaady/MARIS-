# Production Recommendations — Maritime Navigation AI System

These recommendations must be addressed before running this system in a
production or customer-facing environment. The Kubernetes manifests in this
repository are development-grade templates; they need hardening before
production use.

---

## Security

### TLS / HTTPS

All external traffic must be encrypted in transit.

**Action:** Deploy an Ingress controller with TLS termination:
```bash
# Option 1: NGINX Ingress + cert-manager (free Let's Encrypt certs)
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx

helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager --set installCRDs=true
```

Create an Ingress resource:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: maritime-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
    - hosts: [your-domain.com]
      secretName: maritime-tls
  rules:
    - host: your-domain.com
      http:
        paths:
          - path: /api
            pathType: Prefix
            backend:
              service: { name: fastapi-service, port: { number: 8000 } }
          - path: /
            pathType: Prefix
            backend:
              service: { name: frontend-service, port: { number: 80 } }
```

### Secrets Management

Plain Kubernetes Secrets are base64-encoded, not encrypted at rest by default.

**Action:** Use one of:
- **AWS Secrets Manager** + [External Secrets Operator](https://external-secrets.io)
- **HashiCorp Vault** + Vault Agent sidecar or ESO
- **Azure Key Vault** + Secrets Store CSI driver
- Enable etcd encryption at rest in your cluster

Never commit `secrets.yaml` to version control.

### Network Policies

By default, all pods can communicate with each other. Enforce least-privilege:

```yaml
# Example: allow only FastAPI to reach PostgreSQL
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: postgres-ingress-policy
  namespace: maritime-system
spec:
  podSelector:
    matchLabels:
      app: postgres
  policyTypes: [Ingress]
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: fastapi
        - podSelector:
            matchLabels:
              app: live-scorer
        - podSelector:
            matchLabels:
              app: streamlit
      ports:
        - protocol: TCP
          port: 5432
```

### RBAC

Create a dedicated ServiceAccount for each deployment with the minimum
permissions required. Do not use the `default` ServiceAccount.

---

## Observability

### Monitoring (Prometheus + Grafana)

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace
```

Add annotations to deployments to enable scraping:
```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"
```

Add a `/metrics` endpoint to FastAPI using `prometheus-fastapi-instrumentator`.

### Logging (structured logs → centralized sink)

Options:
- **EFK stack** — Elasticsearch + Fluentd/Fluent Bit + Kibana
- **Loki + Grafana** — lighter weight, integrates with the Prometheus stack
- **AWS CloudWatch Container Insights** / **GCP Cloud Logging** — managed

Configure all services to output JSON-structured logs to stdout/stderr.
Kubernetes collects stdout automatically; the log aggregator picks it up.

### Alerting

Set up alerts for:
- Pod restarts > 3 in 5 minutes (CrashLoopBackOff)
- PostgreSQL connections > 80% of `max_connections`
- Kafka consumer lag > threshold (live-scorer falling behind)
- HPA at max replicas (capacity limit reached)
- PVC usage > 80% (storage warning)

---

## Data Persistence and Backups

### PostgreSQL Backups

```bash
# Example: daily pg_dump via a CronJob
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
spec:
  schedule: "0 2 * * *"   # 02:00 UTC daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: postgres:15-alpine
              command:
                - sh
                - -c
                - pg_dump -h postgres-service -U maritime maritime | gzip > /backup/maritime-$(date +%Y%m%d).sql.gz
```

Use a managed database (RDS, Cloud SQL) for automated backups and PITR.

### PVC Snapshots

Use your cloud provider's volume snapshot capability or Velero:
```bash
helm install velero vmware-tanzu/velero \
  --set configuration.provider=aws \
  --set snapshotsEnabled=true
```

### Persistent Storage Classes

For production:
- AWS: `gp3` (better IOPS/cost than `gp2`)
- GCP: `premium-rwo`
- Azure: `managed-premium`

Do NOT use `Reclaim Policy: Delete` for database PVCs — set `Retain`.

---

## Resource Management

### Set resource requests and limits on ALL containers

Containers without resource requests are scheduled as `BestEffort` and are
the first to be evicted under memory pressure.

Containers without limits can starve other pods on the same node.

Review and right-size all values in the deployment files based on actual
profiling data from a load test.

### Pod Disruption Budgets

Prevent all replicas from being evicted simultaneously during node drains:
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: fastapi-pdb
  namespace: maritime-system
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: fastapi
```

---

## Managed Services (Strongly Recommended)

| Component | Self-managed K8s risk | Recommended alternative |
|---|---|---|
| PostgreSQL | Data loss if PVC misconfigured; complex HA setup | AWS RDS / Cloud SQL / Azure DB |
| Kafka | Broker ID conflicts, partition rebalancing; JVM tuning | AWS MSK / Azure Event Hubs / Strimzi |
| Snowflake | N/A — already a managed SaaS | Keep as-is |
| MLflow tracking | Stateful server — data loss if not backed up | MLflow on RDS backend (already configured) |

Using managed services eliminates the operational burden of backups, upgrades,
HA configuration, and storage management for data-critical components.

---

## Checklist before go-live

- [ ] TLS termination configured for all external endpoints
- [ ] Secrets stored in a vault or secrets manager, not in `secrets.yaml`
- [ ] Network Policies defined for all inter-pod communication
- [ ] RBAC ServiceAccounts created (no `default` account usage)
- [ ] Resource requests and limits set on every container
- [ ] PodDisruptionBudgets defined for FastAPI and frontend
- [ ] PostgreSQL on managed service or Velero backup configured
- [ ] Kafka on Strimzi or managed service
- [ ] Prometheus + Grafana or equivalent monitoring deployed
- [ ] Structured JSON logging enabled on all services
- [ ] Alerts defined for CrashLoopBackOff, storage, consumer lag
- [ ] Ingress controller with TLS deployed
- [ ] `storageClassName` set correctly for the target cluster
- [ ] All `REPLACE_WITH_YOUR_IMAGE` placeholders replaced
- [ ] All `REPLACE_WITH_...` secret placeholders replaced
- [ ] `secrets.yaml` added to `.gitignore`
- [ ] Disaster recovery and restore procedure documented and tested
