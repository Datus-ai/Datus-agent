#!/bin/bash
# Uninstall Superset and clean up all related resources

NAMESPACE="default"
RELEASE="superset"

echo "=== Uninstalling Superset ==="

# 1. Uninstall Helm release (will remove most resources)
echo "[1/5] Uninstalling Helm release..."
helm uninstall $RELEASE -n $NAMESPACE --wait 2>/dev/null || echo "Helm release not found or already removed"

# 2. Delete any remaining workloads (fallback cleanup)
echo "[2/5] Deleting remaining workloads..."
kubectl delete all -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE --wait=true 2>/dev/null || true
kubectl delete job -n $NAMESPACE superset-init-db 2>/dev/null || true

# 3. Delete PVCs (persistent data, may not be removed by helm uninstall)
echo "[3/5] Deleting PVCs..."
kubectl delete pvc -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE --wait=true 2>/dev/null || true
kubectl delete pvc -n $NAMESPACE data-${RELEASE}-postgresql-0 redis-data-${RELEASE}-redis-master-0 2>/dev/null || true

# 4. Delete secrets and configmaps
echo "[4/5] Deleting secrets and configmaps..."
kubectl delete secret,configmap -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE 2>/dev/null || true

# 5. Kill port-forward processes
echo "[5/5] Killing port-forward processes..."
pkill -f "kubectl port-forward.*superset" 2>/dev/null || true

# Verify cleanup
echo ""
echo "=== Cleanup Verification ==="
REMAINING=$(kubectl get all,pvc,secret,configmap -n $NAMESPACE 2>/dev/null | grep -i $RELEASE || true)
if [ -z "$REMAINING" ]; then
    echo "All Superset resources have been cleaned up successfully!"
else
    echo "Warning: Some resources may still exist:"
    echo "$REMAINING"
fi
