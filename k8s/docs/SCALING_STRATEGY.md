# Scaling Strategy — Maritime Navigation AI System

## Summary table

| Service | Can scale horizontally? | HPA? | Notes |
|---|---|---|---|
| FastAPI | ✅ Yes | ✅ Yes | Stateless — add replicas freely |
| React frontend | ✅ Yes | ✅ Yes | Static file serving — add replicas freely |
| Streamlit | ⚠️ With care | ❌ No | Session state is in-process |
| Live scorer | ⚠️ With care | ❌ No | Kafka consumer group + partition alignment required |
| PostgreSQL | ❌ Not horizontally | ❌ No | Use read replicas or a managed service |
| Kafka | ❌ Not via HPA | ❌ No | Use Strimzi or managed Kafka |
| Zookeeper | ❌ Odd number only | ❌ No | Scale to 3 for quorum |

---

## Services that scale safely

### FastAPI (replicas: 2–5)

The FastAPI service is fully stateless. Every request reads from PostgreSQL
and runs scoring logic in-process. Adding replicas increases throughput
linearly until the database becomes the bottleneck.

**HPA trigger:** CPU utilization > 70%
**Max replicas:** 5 (increase if node capacity allows)
**Bottleneck at scale:** PostgreSQL connection pool — set `pool_size` and
`max_overflow` in SQLAlchemy to avoid exhausting the PostgreSQL
`max_connections` limit.

```bash
# Scale manually
kubectl scale deployment/fastapi --replicas=3 -n maritime-system

# Or let HPA handle it
kubectl get hpa fastapi-hpa -n maritime-system -w
```

### React Frontend (replicas: 2–5)

The frontend is a pre-built React SPA served by nginx. Completely stateless.
Scales identically to any static file server.

**HPA trigger:** CPU utilization > 70%
**Note:** If you use sticky sessions (sessionAffinity: ClientIP), browser
sessions will always hit the same pod — less load balancing benefit.

---

## Services that need caution

### Streamlit (replicas: 1, with sticky sessions if > 1)

Streamlit stores widget state, DataFrame caches, and `st.session_state` in
the Python process. A user routed to a different pod by the load balancer
will see a reset dashboard.

**Option 1 (recommended):** Keep `replicas: 1`. Streamlit is a monitoring
dashboard, not a high-traffic application.

**Option 2:** Enable sticky sessions in the Service:
```yaml
spec:
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600
```
This keeps each client on the same pod, but does not help with pod restarts.

**Option 3:** Refactor to use external state (Redis or PostgreSQL) for all
`st.session_state` values that must survive pod changes.

### Live Scorer (replicas: 1 by default)

The live scorer is a Kafka consumer that scores every AIS message and writes
alerts to PostgreSQL. The scaling constraint is:

> **One consumer group partition = one active consumer.**

If `ais_raw` has 1 partition and you run 2 scorer replicas, one replica will
be idle. Worse, if you run 2 replicas in different consumer groups, every
message is processed twice and every alert is generated twice.

**To scale the scorer safely:**
1. Increase the `ais_raw` topic partition count:
   ```bash
   kafka-topics.sh --bootstrap-server kafka-service:9092 \
     --alter --topic ais_raw --partitions 3
   ```
2. Set a shared consumer group in the scorer configuration:
   ```
   KAFKA_CONSUMER_GROUP=maritime-scorers
   ```
3. Set `replicas: 3` — one replica per partition.
4. Implement alert deduplication downstream (e.g. PostgreSQL `ON CONFLICT DO NOTHING`).

---

## Services that should use managed services in production

### PostgreSQL

Running PostgreSQL as a Kubernetes StatefulSet requires you to manage:
- Backup schedules and restore procedures
- Point-in-time recovery (PITR)
- High availability (Patroni, pgpool-II, or Citus)
- Storage expansion without downtime
- Version upgrades

**Recommendation:** Use a managed PostgreSQL service:
- AWS RDS for PostgreSQL or Amazon Aurora
- Google Cloud SQL for PostgreSQL
- Azure Database for PostgreSQL Flexible Server

Change only `POSTGRES_HOST` and `POSTGRES_PORT` in the ConfigMap — no
application code changes needed.

### Kafka

Running Kafka as a StatefulSet requires you to manage:
- Broker ID assignment and ZooKeeper coordination
- Topic replication factor consistency during rolling upgrades
- Log retention and compaction policies
- JVM heap tuning for high-throughput

**Recommendation (in order of preference):**
1. **Strimzi Kafka Operator** — runs Kafka natively on Kubernetes with
   automated upgrades, TLS, SASL, and topic management via CRDs.
   ```bash
   helm repo add strimzi https://strimzi.io/charts/
   helm install strimzi-operator strimzi/strimzi-kafka-operator
   ```
2. **Confluent Platform for Kubernetes** — enterprise-grade, more features.
3. **AWS MSK** — fully managed, serverless option available.
4. **Azure Event Hubs (Kafka API)** — drop-in Kafka replacement on Azure.

### Snowflake

Snowflake is a SaaS data warehouse — there is nothing to deploy or scale on
Kubernetes. The `snowflake-secret` in `secrets-template.yaml` provides
credentials for the existing analytics warehouse. No Kubernetes manifests
are needed for Snowflake itself.
