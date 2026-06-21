#!/bin/bash
# deploy.sh — Maritime Navigation AI System
# Helper script to apply manifests in correct order.
#
# Usage:
#   ./k8s/deploy.sh           # Dry-run mode (default, safe)
#   ./k8s/deploy.sh apply     # Apply changes to cluster
#
# Default mode is dry-run. Use 'apply' argument to make actual changes.

set -e

NAMESPACE="maritime-system"
DRY_RUN="--dry-run=client"

if [[ "$1" == "apply" ]]; then
    DRY_RUN=""
    echo "=== APPLY MODE: Changes will be applied to the cluster ==="
else
    echo "=== DRY-RUN MODE: No changes will be made ==="
    echo "To apply changes, run: ./k8s/deploy.sh apply"
fi

echo ""

# Apply in order: namespace -> configmaps -> secrets -> deployments -> services -> hpa
FILES_CREATED=0
FILES_SKIPPED=0

apply_file() {
    local file=$1
    local desc=$2

    if [[ -f "$file" ]]; then
        echo "Applying: $file ($desc)"
        if [[ -n "$DRY_RUN" ]]; then
            kubectl apply -f "$file" -n "$NAMESPACE" $DRY_RUN
        else
            kubectl apply -f "$file" -n "$NAMESPACE"
        fi
        ((FILES_CREATED++)) || true
    else
        echo "Skipping: $file (not found)"
        ((FILES_SKIPPED++)) || true
    fi
}

# Create namespace first
echo "=== Step 1: Namespace ==="
apply_file "k8s/namespace.yaml" "namespace definition"

# Configmaps
echo "=== Step 2: ConfigMaps ==="
apply_file "k8s/configmaps/app-config.yaml" "app configuration"

# Secrets (optional - check if secrets.yaml exists)
echo "=== Step 3: Secrets ==="
if [[ -f "k8s/secrets/secrets.yaml" ]]; then
    apply_file "k8s/secrets/secrets.yaml" "secrets"
else
    echo "Skipping: k8s/secrets/secrets.yaml (not found - copy from secrets-template.yaml and customize)"
    ((FILES_SKIPPED++)) || true
fi

# Deployments
echo "=== Step 4: Deployments ==="
apply_file "k8s/deployments/postgres-deployment.yaml" "PostgreSQL"
apply_file "k8s/deployments/kafka-deployment.yaml" "Kafka + Zookeeper"
apply_file "k8s/deployments/fastapi-deployment.yaml" "FastAPI"
apply_file "k8s/deployments/frontend-deployment.yaml" "React Frontend"
apply_file "k8s/deployments/streamlit-deployment.yaml" "Streamlit"
apply_file "k8s/deployments/live-scorer-deployment.yaml" "Live Scorer"

# Services
echo "=== Step 5: Services ==="
apply_file "k8s/services/services.yaml" "All services"

# HPA
echo "=== Step 6: HorizontalPodAutoscalers ==="
apply_file "k8s/hpa/hpa.yaml" "autoscaling"

echo ""
echo "=== Summary ==="
echo "Files created: $FILES_CREATED"
echo "Files skipped: $FILES_SKIPPED"
echo "No existing files were modified."