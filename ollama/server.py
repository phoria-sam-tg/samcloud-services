"""
Model Service - FastAPI server

Unified management for all model backends (Ollama + llama-server) on slice-test.
Handles SAMcloud resource leasing, model lifecycle, and health reporting.

Endpoints:
  GET  /health              - Health check
  GET  /status              - Full status (all backends, models, leases, resources)
  GET  /models              - List all managed + available models
  POST /models/load         - Load a model (Ollama or llama-server)
  POST /models/unload       - Unload a model (releases lease)
  POST /v1/chat/completions - OpenAI-compatible chat (routes to correct backend)
  POST /v1/completions      - OpenAI-compatible completion
"""

import os
import json
import time
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from .manager import ModelManager, Backend
from .samcloud import SamcloudClient
from .ollama_client import OllamaClient
from .llama_client import LlamaServerClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("model-service")

SC_TOKEN = os.environ.get("SC_TOKEN", "sc_agent_448090436817362f5250c0d0f83bef53")
SERVICE_PORT = int(os.environ.get("SERVICE_PORT", "8800"))

mgr: Optional[ModelManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mgr
    sc = SamcloudClient(token=SC_TOKEN)
    mgr = ModelManager(sc=sc)

    # Discover and adopt running models
    adopted = mgr.discover()
    for m in adopted:
        log.info(f"Adopted: {m.name} ({m.backend.value}) ~{m.memory_mb}MB on port {m.port}")

    # Claim leases for everything running
    leases = mgr.claim_leases()
    for l in leases:
        log.info(f"Lease: {l}")

    mgr.start_background_tasks()
    log.info(f"Model Service ready - managing {len(mgr.models)} models")

    yield

    log.info("Shutting down...")
    results = mgr.shutdown()
    for r in results:
        log.info(f"  {r}")


app = FastAPI(
    title="Model Service",
    description="Unified model serving with SAMcloud resource leasing",
    lifespan=lifespan,
)


# -- Request models --

class LoadRequest(BaseModel):
    model: str
    backend: str = "auto"  # "ollama", "llama-server", or "auto"
    port: int = 8000  # for llama-server
    ctx_size: int = 12288
    gpu_layers: int = 99

class UnloadRequest(BaseModel):
    model: str
    force: bool = False

class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    stream: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# -- Endpoints --

@app.get("/health")
async def health():
    return {"status": "ok", "service": "model-service", "models": len(mgr.models)}


@app.get("/status")
async def status():
    return mgr.status()


@app.get("/models")
async def list_models():
    return {
        "managed": {
            name: {
                "backend": mm.backend.value,
                "port": mm.port,
                "memory_mb": mm.memory_mb,
                "lease_id": mm.lease_id,
                "idle_seconds": int(time.time() - mm.last_used),
                "request_count": mm.request_count,
            }
            for name, mm in mgr.models.items()
        },
        "available_ollama": [m["name"] for m in mgr.ollama.list_models()],
        "available_gguf": mgr.llama.available_models(),
    }


@app.post("/models/load")
async def load_model(req: LoadRequest):
    try:
        if req.backend == "llama-server" or (
            req.backend == "auto" and req.model.endswith(".gguf")
        ):
            # Resolve to full path if just a filename
            model_path = req.model
            if not model_path.startswith("/"):
                model_path = f"/Users/sam/models/{req.model}"
            mm = mgr.load_llama_model(
                model_path,
                port=req.port,
                ctx_size=req.ctx_size,
                gpu_layers=req.gpu_layers,
            )
        else:
            mm = mgr.load_ollama_model(req.model)

        return {
            "status": "loaded",
            "model": mm.name,
            "backend": mm.backend.value,
            "port": mm.port,
            "memory_mb": mm.memory_mb,
            "lease_id": mm.lease_id,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        log.exception(f"Failed to load {req.model}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/unload")
async def unload_model(req: UnloadRequest):
    result = mgr.unload(req.model, force=req.force)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"{req.model} not found")
    return result


def _resolve_model(model_name: str):
    """Find which managed model matches the request and touch it."""
    # Exact match
    if model_name in mgr.models:
        mgr.touch(model_name)
        return mgr.models[model_name]
    # Partial match (e.g. "qwen3-32b" matches "Qwen3-32B-Q6_K")
    lower = model_name.lower()
    for name, mm in mgr.models.items():
        if lower in name.lower():
            mgr.touch(name)
            return mm
    return None


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    mm = _resolve_model(req.model)
    if not mm:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not loaded")

    if mm.backend == Backend.OLLAMA:
        # Route to Ollama
        if req.stream:
            def stream():
                for chunk in mgr.ollama.chat(mm.name, req.messages):
                    yield json.dumps(chunk) + "\n"
            return StreamingResponse(stream(), media_type="application/x-ndjson")
        else:
            chunks = list(mgr.ollama.chat(mm.name, req.messages))
            return chunks[-1] if chunks else {}

    elif mm.backend == Backend.LLAMA:
        # Forward to llama-server's OpenAI-compatible endpoint
        payload = {
            "model": mm.name,
            "messages": req.messages,
            "stream": req.stream,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens

        if req.stream:
            async def stream():
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"http://127.0.0.1:{mm.port}/v1/chat/completions",
                        json=payload,
                        timeout=None,
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if line:
                                yield line + "\n"
            return StreamingResponse(stream(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{mm.port}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                )
                return resp.json()


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    mm = _resolve_model(req.model)
    if not mm:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not loaded")

    if mm.backend == Backend.OLLAMA:
        if req.stream:
            def stream():
                for chunk in mgr.ollama.generate(mm.name, req.prompt):
                    yield json.dumps(chunk) + "\n"
            return StreamingResponse(stream(), media_type="application/x-ndjson")
        else:
            chunks = list(mgr.ollama.generate(mm.name, req.prompt))
            return chunks[-1] if chunks else {}

    elif mm.backend == Backend.LLAMA:
        payload = {
            "model": mm.name,
            "prompt": req.prompt,
            "stream": req.stream,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens

        if req.stream:
            async def stream():
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"http://127.0.0.1:{mm.port}/v1/completions",
                        json=payload,
                        timeout=None,
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if line:
                                yield line + "\n"
            return StreamingResponse(stream(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{mm.port}/v1/completions",
                    json=payload,
                    timeout=300,
                )
                return resp.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
