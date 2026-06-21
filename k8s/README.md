# Kubernetes — Maritime Navigation AI System

## Docker Compose vs Kubernetes

| | Docker Compose | Kubernetes |
|---|---|---|
| **Primary use** | Local development, demos, single-node staging | Multi-node production clusters |
| **Complexity** | Low — one `docker compose up -d` | High — namespaces, RBAC, manifests, operators |
| **Scaling** | Manual `--scale` on a single host | Automatic HPA across a node pool |
| **Self-healing** | Restart policy only | Liveness/readiness probes + controller reconciliation |
| **Networking** | Bridge network, port mapping | ClusterIP, LoadBalancer, Ingress |
| **Storage** | Named volumes on host | PersistentVolumeClaims + StorageClasses |
| **Secrets** | `.env` files or Docker secrets | Kubernetes Secrets + external vault integrations |
| **Status** | **Primary runtime (use this now)** | Future-ready templates (not deployed yet) |

---

## Why Docker Compose is kept as the default

The entire development, testing, and demo workflow is built around the
`docker-compose.yml` at the project root. It spins up all 12 services with a
single command, requires no cloud account, and runs on any laptop.

The Kubernetes manifests in this directory are **templates** prepared in
advance so that migration is straightforward when the project outgrows a
single host. They are not deployed and not tested against a live cluster.

---

## When to consider migrating to Kubernetes

- AIS ingest volume exceeds what a single host can handle (~50k messages/sec)
- You need zero-downtime rolling deployments for the FastAPI or scoring services
- Multiple teams need isolated environments (namespaces per team)
- Kafka consumers need independent horizontal scaling
- You require automatic failover for PostgreSQL

For smaller deployments, **Docker Compose + a beefy VM** is almost always
the right answer and far cheaper to operate.

---

## What is in this directory

```
k8s/
├── README.md                       ← this file
├── namespace.yaml                  ← maritime-system namespace
├── deploy.sh                       ← helper script (dry-run by default)
├── configmaps/
│   └── app-config.yaml             ← non-secret environment variables
├── secrets/
│   └── secrets-template.yaml       ← placeholder secrets (MUST be replaced)
├── deployments/
│   ├── postgres-deployment.yaml    ← StatefulSet + PVC
│   ├── kafka-deployment.yaml       ← Zookeeper + Kafka StatefulSets + PVCs
│   ├── fastapi-deployment.yaml     ← Deployment, 2 replicas
│   ├── frontend-deployment.yaml    ← React Deployment, 2 replicas
│   ├── streamlit-deployment.yaml   ← Streamlit Deployment
│   └── live-scorer-deployment.yaml ← Live scorer (1 replica recommended)
├── services/
│   └── services.yaml               ← ClusterIP + LoadBalancer services
├── hpa/
│   └── hpa.yaml                    ← HPA for FastAPI and frontend only
└── docs/
    ├── MIGRATION_GUIDE.md          ← step-by-step Docker Compose → K8s
    ├── SCALING_STRATEGY.md         ← what can scale and how
    └── PRODUCTION_RECOMMENDATIONS.md ← production checklist
```

---

## Quick orientation

Before applying any manifest:
1. Replace all placeholder image names (search for `REPLACE_WITH_YOUR_IMAGE`)
2. Fill in real secret values in `secrets/secrets-template.yaml` and rename it
3. Choose an appropriate `storageClassName` for your cluster
4. Review resource requests/limits against your node sizes
5. Read `docs/PRODUCTION_RECOMMENDATIONS.md` in full

See `docs/MIGRATION_GUIDE.md` for the ordered deployment sequence.
