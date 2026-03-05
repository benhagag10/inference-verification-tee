# Changes Log

## 2026-03-04: Slim TEE Docker image (Dockerfile.tee + requirements-tee.txt)

### Problem
The original Docker image was **8.4GB** (compressed). The Tinfoil TEE ramdisk
(`/mnt/ramdisk`) only had **3.3GB free**, so the container image could never be
pulled, causing a 502 on all endpoints.

### Root cause
The original Dockerfile used `nvidia/cuda:12.4.0-devel` (includes CUDA compilers,
headers, static libraries) and `requirements.txt` included many packages only
needed for development, analysis, and notebooks — not for serving.

### What changed

**Base image: `devel` -> `runtime` (~2GB saved)**
- `devel` includes `nvcc`, CUDA headers, static libs for compiling CUDA code
- `runtime` includes only the shared libraries needed to *run* CUDA applications
- We don't compile any CUDA code at runtime, so `runtime` is sufficient

**Removed packages (~1.5GB saved):**

| Package | Why removed |
|---|---|
| `torchvision==0.22.1` (~500MB) | Never imported by api_server, generate, verify, or scoring_functions |
| `xformers` (~500MB) | vLLM ships its own attention kernels; xformers is redundant |
| `wandb` (~150MB) | Experiment tracking — not used during serving |
| `matplotlib>=3.9.4` (~150MB) | Plotting — analysis only |
| `seaborn>=0.13.2` (~50MB) | Plotting — analysis only |
| `pandas` (~100MB) | Data analysis — not imported in runtime code |
| `scikit-learn` (~100MB) | ML utilities — not imported in runtime code |
| `bitsandbytes==0.46.1` (~100MB) | Quantization library — not imported in runtime code |
| `accelerate>=1.0.1` (~100MB) | Model loading helper — vLLM handles its own loading |
| `ipykernel` (~50MB) | Jupyter notebook kernel — dev tool |
| `pytest` (~30MB) | Test framework — dev tool |
| `einops` (~10MB) | Tensor ops — not imported in runtime code |
| `jaxtyping` (~10MB) | Type annotations — not imported in runtime code |
| `hydra-core` (~30MB) | Config framework — not imported in runtime code |

**`--no-deps` install:** The `pip install .` step now uses `--no-deps` to
prevent pyproject.toml from re-installing the full dependency list.

### What did NOT change
- No code changes — api_server.py, generate.py, verify.py, scoring_functions/ are untouched
- No behavior changes — same imports, same endpoints, same results
- Original Dockerfile and requirements.txt preserved for local dev use

## 2026-03-04: Increase TEE memory + wire up Dockerfile.tee

### Changes
- `tinfoil-config.yml`: `memory: 16384` -> `memory: 32768` (32GB)
  - The TEE has no real disk; Docker images live on a ramdisk carved from system RAM
  - 16GB was not enough to fit the OS + Docker daemon + container image
  - 32GB gives ~16GB of ramdisk headroom for the slim image (~4.5GB)
- `tinfoil-config.yml`: bumped image tag from `v0.5.0` (pinned digest) to `v0.6.0`
  - Removed the `@sha256:...` digest pin; will be re-pinned after build
- `.github/workflows/docker-build.yml`: added `file: Dockerfile.tee`
  - CI now builds from the slim Dockerfile instead of the original
