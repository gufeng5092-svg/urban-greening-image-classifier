# Deployment

## Device Policy

The code uses `device: auto` by default:

1. CUDA, when available.
2. Apple MPS, when available.
3. CPU fallback.

For this repository, CUDA is optional and should not be assumed. On Apple Silicon development machines, MPS is the expected accelerator. Docker images default to CPU PyTorch because containers should be portable and should not pull CUDA libraries unless they are explicitly needed.

## macOS with MPS

Install the normal project dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Keep the inference config as:

```yaml
inference:
  device: auto
```

PyTorch will use MPS when it is available.

## Docker CPU Runtime

The provided Dockerfile installs CPU-only PyTorch:

```bash
docker build -t urban-greening-classifier .
docker run --rm -p 7860:7860 urban-greening-classifier
```

This avoids unnecessary CUDA dependencies in the API image.

## Optional Linux CUDA Runtime

Use CUDA only for a Linux host with a compatible NVIDIA driver and container runtime. One common approach is to create a CUDA-specific Dockerfile or override the PyTorch install step:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then set:

```yaml
inference:
  device: cuda
```

Do not enable CUDA by default in CI or portable CPU/MPS deployments.
