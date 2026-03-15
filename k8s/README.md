# Kubernetes Deployment

This directory deploys the application into a `teaching` namespace on minikube running on the remote Linux server.

## Topology

- `frontend`: static Vite build served by NGINX
- `sqora-app`: main backend container plus the Manim runner sidecar
- `headtts`: standalone TTS service
- `vllm`: model-serving API
- `qdrant`: vector database with persistent storage
- `monitoring`: `kube-prometheus-stack` plus Loki via Helm

`HeadTTS` is deployed separately and exposed through ingress at `/tts`. The Manim runner stays inside the `sqora-app` pod because the worker code calls `http://localhost:8080/generate`.

## Prerequisites

- `minikube`
- `kubectl`
- `docker`
- `helm`
- A usable env file at [sqora/.env](/home/raid/sqora/sqora/.env)

## Deploy

Run:

```bash
./k8s/deploy.sh
```

The script:

1. Starts minikube.
2. Enables `ingress`, `metrics-server`, and `storage-provisioner`.
3. Builds all local images inside minikube's Docker daemon.
4. Pulls and retags `vllm/vllm-openai:latest` into the minikube image store.
5. Creates the `sqora-dotenv` secret from `sqora/.env`.
6. Installs `kube-prometheus-stack` and `loki-stack`.
7. Applies every manifest under `k8s/`.
8. Prints the frontend, API, TTS, and Grafana URLs.

## Remote Access Over SSH

Because minikube runs on the remote server, access it through SSH tunnels from your local machine.

First, on the remote server, get the minikube IP if you need it manually:

```bash
minikube ip --profile teaching
```

Recommended tunnels:

```bash
ssh -L 8080:$(ssh user@remote-server "minikube ip --profile teaching"):80 \
    -L 32000:$(ssh user@remote-server "minikube ip --profile teaching"):32000 \
    user@remote-server
```

If your shell does not like command substitution across SSH, resolve the IP first:

```bash
REMOTE_MINIKUBE_IP=$(ssh user@remote-server "minikube ip --profile teaching")
ssh -L 8080:${REMOTE_MINIKUBE_IP}:80 \
    -L 32000:${REMOTE_MINIKUBE_IP}:32000 \
    user@remote-server
```

Then open:

- App: `http://localhost:8080/`
- Backend API: `http://localhost:8080/api`
- HeadTTS: `http://localhost:8080/tts/v1/synthesize`
- Grafana: `http://localhost:32000/`

## Notes

- The frontend image bakes Firebase `VITE_*` values from `sqora/.env` during `docker build`.
- The `shared-media`, `qdrant-storage`, `huggingface-cache`, and `vllm-model-cache` PVCs use the default minikube storage class.
- `headtts` and `vllm` have HPAs. `sqora-app` stays at one replica because it owns the persistent user/media volume and the Manim worker uses localhost sidecar routing.
