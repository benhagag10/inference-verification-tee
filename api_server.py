"""
FastAPI server wrapping the inference verification pipeline.

Provides endpoints for TEE-based generation, verification, and classification
of LLM outputs using Gumbel Likelihood Scores (GLS).
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, str(Path(__file__).parent))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from inference_verification.generate import (
    GenerationConfig,
    load_prompts,
    generate_with_vllm,
)
from inference_verification.verify import (
    VerificationConfig,
    verify_outputs,
    classify_tokens,
)

app = FastAPI(
    title="Inference Verification API",
    description="Verifies LLM outputs for model weight exfiltration detection",
)


# --- Request / Response models ---

class VerifyRequest(BaseModel):
    """Request body for /verify endpoint."""
    n_prompts: int = Field(default=10, description="Number of prompts to sample from dataset")
    max_tokens: int = Field(default=100, description="Max tokens to generate per prompt")
    config: Optional[dict] = Field(default=None, description="Override generation/verification config")


class TokenResult(BaseModel):
    gls_score: float
    logit_rank: int
    classification: str


class VerifyResponse(BaseModel):
    model_name: str
    seed: int
    n_prompts: int
    total_tokens: int
    num_safe: int
    num_suspicious: int
    num_dangerous: int
    safe_pct: float
    suspicious_pct: float
    dangerous_pct: float
    gls_threshold: float
    logit_rank_threshold: int
    tokens: list[TokenResult]


# --- Default config ---

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_SEED = 42
DEFAULT_GLS_THRESHOLD = -5.0
DEFAULT_LOGIT_RANK_THRESHOLD = 10


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def get_config():
    return {
        "model_name": DEFAULT_MODEL,
        "seed": DEFAULT_SEED,
        "gls_threshold": DEFAULT_GLS_THRESHOLD,
        "logit_rank_threshold": DEFAULT_LOGIT_RANK_THRESHOLD,
        "gpu_available": torch.cuda.is_available(),
    }


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="CUDA not available")

    overrides = req.config or {}

    # Build generation config
    gen_cfg = GenerationConfig(
        model_name=overrides.get("model_name", DEFAULT_MODEL),
        n_prompts=req.n_prompts,
        max_tokens=req.max_tokens,
        temperature=overrides.get("temperature", 1.0),
        top_k=overrides.get("top_k", 50),
        top_p=overrides.get("top_p", 0.95),
        seed=overrides.get("seed", DEFAULT_SEED),
        gpu_memory_utilization=overrides.get("gpu_memory_utilization", 0.7),
    )

    # Build verification config
    ver_cfg = VerificationConfig(
        model_name=gen_cfg.model_name,
        temperature=gen_cfg.temperature,
        top_k=gen_cfg.top_k,
        top_p=gen_cfg.top_p,
        seed=gen_cfg.seed,
        gls_threshold=overrides.get("gls_threshold", DEFAULT_GLS_THRESHOLD),
        logit_rank_threshold=overrides.get("logit_rank_threshold", DEFAULT_LOGIT_RANK_THRESHOLD),
    )

    # Step 1: Load prompts from dataset
    prompts = load_prompts(gen_cfg)

    # Step 2: Generate with vLLM
    outputs = generate_with_vllm(gen_cfg, prompts)

    # Step 3: Verify outputs
    verification_results = verify_outputs(ver_cfg, outputs)

    # Step 4: Classify tokens
    classification = classify_tokens(
        verification_results,
        gls_threshold=ver_cfg.gls_threshold,
        logit_rank_threshold=ver_cfg.logit_rank_threshold,
    )

    # Build per-token results
    tokens = []
    for i, result in enumerate(verification_results):
        gls = result["sampled_gumbel_scores"]
        if isinstance(gls, torch.Tensor):
            gls = gls.item()
        tokens.append(TokenResult(
            gls_score=float(gls),
            logit_rank=int(result["logit_rank"]),
            classification=classification["classifications"][i].value,
        ))

    total = len(tokens)
    num_safe = classification["num_safe"]
    num_suspicious = classification["num_suspicious"]
    num_dangerous = classification["num_dangerous"]

    return VerifyResponse(
        model_name=gen_cfg.model_name,
        seed=gen_cfg.seed,
        n_prompts=req.n_prompts,
        total_tokens=total,
        num_safe=num_safe,
        num_suspicious=num_suspicious,
        num_dangerous=num_dangerous,
        safe_pct=round(num_safe / total * 100, 2) if total > 0 else 0,
        suspicious_pct=round(num_suspicious / total * 100, 2) if total > 0 else 0,
        dangerous_pct=round(num_dangerous / total * 100, 2) if total > 0 else 0,
        gls_threshold=ver_cfg.gls_threshold,
        logit_rank_threshold=ver_cfg.logit_rank_threshold,
        tokens=tokens,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
