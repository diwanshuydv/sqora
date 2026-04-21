#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K8S_DIR="$ROOT_DIR/k8s"
NAMESPACE="teaching"
MONITORING_NAMESPACE="monitoring"
PROFILE="${MINIKUBE_PROFILE:-teaching}"
DRIVER="${MINIKUBE_DRIVER:-docker}"
ENABLE_GPU="${MINIKUBE_ENABLE_GPU:-true}"
CPUS="${MINIKUBE_CPUS:-6}"
MEMORY="${MINIKUBE_MEMORY:-12288}"
DISK_SIZE="${MINIKUBE_DISK_SIZE:-80g}"
GRAFANA_NODEPORT="${GRAFANA_NODEPORT:-32000}"
# Host directory containing the trained vLLM model weights.
# Mounted into minikube at /mnt/sqora-model so the vllm initContainer can seed
# the PVC on first boot without needing to pull from HuggingFace.
MODEL_DIR="${VLLM_MODEL_DIR:-$ROOT_DIR/manim-trainer/output/merged_vllm_model}"

require() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

require minikube
require kubectl
require docker
require helm

if [[ ! -f "$ROOT_DIR/sqora/.env" ]]; then
  echo "Expected env file at $ROOT_DIR/sqora/.env" >&2
  exit 1
fi

GPU_START_ARGS=()
if [[ "$ENABLE_GPU" == "true" ]]; then
  require nvidia-smi
  require nvidia-ctk
  GPU_START_ARGS+=(--gpus=all)
fi

echo "Starting minikube profile $PROFILE..."
minikube start \
  --profile="$PROFILE" \
  --driver="$DRIVER" \
  --cpus="$CPUS" \
  --memory="$MEMORY" \
  --disk-size="$DISK_SIZE" \
  --mount \
  --mount-string="$MODEL_DIR:/mnt/sqora-model" \
  --force \
  "${GPU_START_ARGS[@]}"

echo "Enabling minikube addons..."
minikube addons enable ingress --profile="$PROFILE"
minikube addons enable metrics-server --profile="$PROFILE"
minikube addons enable storage-provisioner --profile="$PROFILE"
if [[ "$ENABLE_GPU" == "true" ]]; then
  minikube addons enable nvidia-device-plugin --profile="$PROFILE"
fi

eval "$(minikube -p "$PROFILE" docker-env)"

echo "Loading frontend build variables from sqora/.env..."
set -a
source "$ROOT_DIR/sqora/.env"
set +a

echo "Building images inside minikube..."
docker build -t teaching-sqora-backend:latest "$ROOT_DIR/sqora"
docker build -t teaching-headtts:latest "$ROOT_DIR/HeadTTS"
docker build -f "$K8S_DIR/Dockerfile.manim-runner" -t teaching-manim-runner:latest "$ROOT_DIR"
docker build \
  -f "$K8S_DIR/Dockerfile.frontend" \
  -t teaching-frontend:latest \
  --build-arg VITE_API_URL= \
  --build-arg VITE_TTS_URL=/tts \
  --build-arg VITE_FIREBASE_API_KEY="${VITE_FIREBASE_API_KEY:-}" \
  --build-arg VITE_FIREBASE_AUTH_DOMAIN="${VITE_FIREBASE_AUTH_DOMAIN:-}" \
  --build-arg VITE_FIREBASE_PROJECT_ID="${VITE_FIREBASE_PROJECT_ID:-}" \
  --build-arg VITE_FIREBASE_STORAGE_BUCKET="${VITE_FIREBASE_STORAGE_BUCKET:-}" \
  --build-arg VITE_FIREBASE_MESSAGING_SENDER_ID="${VITE_FIREBASE_MESSAGING_SENDER_ID:-}" \
  --build-arg VITE_FIREBASE_APP_ID="${VITE_FIREBASE_APP_ID:-}" \
  --build-arg VITE_FIREBASE_MEASUREMENT_ID="${VITE_FIREBASE_MEASUREMENT_ID:-}" \
  "$ROOT_DIR"

docker pull vllm/vllm-openai:latest
docker tag vllm/vllm-openai:latest teaching-vllm:latest

echo "Preparing namespaces and secrets..."
kubectl apply -f "$K8S_DIR/namespace.yaml"
kubectl create namespace "$MONITORING_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NAMESPACE" create secret generic sqora-dotenv \
  --from-file=.env="$ROOT_DIR/sqora/.env" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Installing monitoring stack..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
helm repo add grafana https://grafana.github.io/helm-charts >/dev/null
helm repo update >/dev/null

helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace "$MONITORING_NAMESPACE" \
  --create-namespace \
  -f "$K8S_DIR/helm/kube-prometheus-stack-values.yaml"

helm upgrade --install loki grafana/loki-stack \
  --namespace "$MONITORING_NAMESPACE" \
  --create-namespace \
  -f "$K8S_DIR/helm/loki-values.yaml"

echo "Applying application manifests..."
kubectl apply -k "$K8S_DIR"

echo "Waiting for core deployments..."
kubectl -n "$NAMESPACE" rollout status deployment/qdrant --timeout=10m
kubectl -n "$NAMESPACE" rollout status deployment/vllm --timeout=30m
kubectl -n "$NAMESPACE" rollout status deployment/headtts --timeout=10m
kubectl -n "$NAMESPACE" rollout status deployment/sqora-app --timeout=15m
kubectl -n "$NAMESPACE" rollout status deployment/frontend --timeout=10m

MINIKUBE_IP="$(minikube ip --profile="$PROFILE")"

echo
echo "Application URLs"
echo "  Frontend: http://$MINIKUBE_IP/"
echo "  Backend API: http://$MINIKUBE_IP/api"
echo "  HeadTTS: http://$MINIKUBE_IP/tts/v1/synthesize"
echo "  Grafana: http://$MINIKUBE_IP:$GRAFANA_NODEPORT"
echo
echo "Useful commands"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl get ingress -n $NAMESPACE"
echo "  kubectl get svc -n $MONITORING_NAMESPACE"
