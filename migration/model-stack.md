# Model Stack — Migration Reference

**Captured**: 2026-04-19
**Purpose**: Reference for containerizing the model inference stack
**Status**: Services stopped, plists removed, model data wiped. Ready for contained deployment.

---

## Architecture (pre-migration)

```
Satellite agents (containers)
  └─ OPENAI_BASE_URL → models-stg.samtg.xyz:9443 (Caddy)
       └─ model-service (port 8800, uvicorn)
            ├─ ollama backend (port 11434) → qwen3.5, hermes3, glm4, etc.
            ├─ llama-server backend → GGUF imports (qwen2.5-32b-agi)
            └─ mlx-vlm backend (port 8801) → gemma-4 vision
```

## Services (pre-migration)

| Service | Port | Process | Managed by | Source |
|---------|------|---------|------------|--------|
| ollama | 11434 | ollama serve | homebrew plist (KeepAlive) | homebrew |
| model-service | 8800 | uvicorn ollama.server:app | manual | /Users/sam/Documents/services |
| mlx_vlm | 8801 | mlx_vlm.server (gemma-4) | manual | pip: mlx-vlm |
| eval-service | 8802 | uvicorn | manual | /Users/sam/Documents/Projects/eval-service |
| ai-service (legacy) | 8766 | node server.js | plist (KeepAlive) | ~/.local/samcloud/ai-service |
| watchdog | — | zsh script, 300s interval | plist | ~/.local/samcloud/watchdog-start.sh |
| model-gateway | 18008→8450 | autossh tunnel | plist (KeepAlive) | — |

## Ollama Config

- Flash attention: enabled (`OLLAMA_FLASH_ATTENTION=1`)
- KV cache type: q8_0 (`OLLAMA_KV_CACHE_TYPE=q8_0`)
- Auto-reload: model evicts after 5min idle, reloads on next request (~14s)
- Keep-alive override available: `keep_alive=-1` for persistent loading

---

## Model Inventory & Recommendations

### Bring back (proven, worth re-pulling)

| Model | Size | Eval | Tool calling | Speed | Verdict |
|-------|------|------|-------------|-------|---------|
| **qwen3.5:35b-a3b-coding-nvfp4** | 21 GB | 100/100 | Native, reliable | Fast on MLX | **Primary agent model.** Best tool calling of anything tested. Coding variant is more reliable than base for structured output. Always have this available. |
| **hermes3:8b** | 4.7 GB | 90/100 | Native (ollama) | Very fast | **Lightweight fallback.** Good for simple tasks, low memory. Occasionally too cautious — refuses tasks it shouldn't. Good for eval baseline. |
| **qwen2.5:1.5b** | 986 MB | 90/100 | Native (ollama) | Instant | **Smoke test model.** Surprisingly capable for its size. Use for testing pipelines, health checks, quick eval runs. Not for real agent work. |

### Conditional (useful for specific cases)

| Model | Size | Eval | Tool calling | Speed | Verdict |
|-------|------|------|-------------|-------|---------|
| **gemma-4-31b-it-nvfp4** | ~20 GB | 86/100 | Parsed (`<\|tool_call\>` tags) | ~7 tok/s | **Vision model.** Only option for image understanding. Requires mlx-vlm backend + model-service tag parsing. Bring back when vision tasks needed. |
| **qwen3.5:35b-a3b** | 23 GB | — | Native | Fast | Base variant of primary. No advantage over coding variant for agent use. Skip unless you need non-coding behavior. |
| **qwen3.5-notk** | 23 GB | — | Native | Faster (no thinking) | No-thinking variant. Faster responses but less reasoning. Could be useful for high-throughput simple tasks. |

### Don't bother

| Model | Size | Eval | Why not |
|-------|------|------|---------|
| **qwen2.5-32b-agi** | 19 GB | 54/100 | Community fine-tune. Tool calling almost completely broken (22% pass rate). Safety refusal failed — generated destructive bash script instead of refusing. The GGUF import process works (Modelfile + `ollama create`) but this particular model is bad. |
| **glm4** | 5.5 GB | 40/100 | Poor eval, unreliable tool calling. No advantage over hermes3:8b at similar size. |
| **llava:7b** | 4.7 GB | — | Legacy vision model from ai-service (port 8766). Superseded by gemma-4. |
| **qwen3-coder:30b** | 18 GB | — | Older generation. Superseded by qwen3.5 coding variant in every way. |

### Models to watch (not yet tested)

- **Qwen 3.5 72B+ variants**: If memory allows. The 35B-a3b is already excellent but larger models may handle complex multi-step agent tasks better.
- **Gemma 4 updates**: Current vision model works but tool calling required model-service parsing. Future versions may support native tool_calls.
- **Llama 4 (Meta)**: When available in ollama/MLX. Worth eval for comparison.
- **DeepSeek variants**: Active model family, community producing good quants. Worth testing if agent-oriented fine-tunes appear.

---

## Three Unified Backends (model-service)

The model-service at port 8800 is the key piece — it presents a single OpenAI-compatible API to agents and routes to the right backend:

1. **ollama** (port 11434): Native `tool_calls` support. Used for all qwen3.5, hermes3, glm4, qwen2.5 models. This is the primary path — most models go here. Ollama handles quantization, MLX acceleration, and model lifecycle (load/evict).

2. **llama-server** (llama.cpp): OpenAI-compatible pass-through. Used for raw GGUF imports that don't have ollama manifests. The qwen2.5-32b-agi was served this way. Process: download GGUF → write Modelfile with chat template → `ollama create` → serve via ollama. Or run llama-server directly for models ollama can't handle.

3. **mlx-vlm** (port 8801): For vision models (gemma-4). Model outputs raw `<|tool_call>` XML tags in its text response. Model-service parses these into proper OpenAI `tool_calls` format before returning to the agent. This parsing was broken initially and required a fix (ticket #91).

### Backend selection logic (model-service)
- Model config maps model names to backends
- Default: ollama
- Vision models: routed to mlx-vlm
- GGUF imports: either ollama (after `ollama create`) or llama-server directly

---

## Eval Service

- **Source**: /Users/sam/Documents/Projects/eval-service
- **Port**: 8802
- **Trigger**: `POST /eval` with `{"model": "model-name"}`
- **Leaderboard**: `GET /leaderboard`
- **Tests**: Tool calling accuracy, reasoning, speed, safety refusal
- **Scoring**: Out of 100 (composite of subtests)

The eval service is independent from model-service — it calls models via the OpenAI-compatible API and scores their responses. Useful for validating new models before deploying to agents.

### Known eval issues (ticket #92)
- Format sensitivity: some models fail on JSON parsing rather than actual capability
- Speed scoring penalizes larger models unfairly
- Safety test is binary (pass/fail) — qwen2.5-32b-agi generated a destructive script and still got partial credit
- Suggested: separate safety into its own critical-path check

---

## Plists Removed (2026-04-19)

| Plist | What it did | Notes |
|-------|-------------|-------|
| `homebrew.mxcl.ollama.plist` | ollama serve (KeepAlive) | Core model runtime |
| `com.samcloud.ai-service.plist` | Legacy Node.js AI service on 8766 | Has AWS creds (redacted copy in migration/plists/) |
| `com.samcloud.watchdog.plist` | Health monitor every 300s | Checked service availability |
| `com.samcloud.workcloud-forward-5090-model-gateway.plist` | autossh tunnel 18008→8450 | External model access via workcloud |

Model data wiped: ~/.ollama/models (99GB) + ~/.cache/huggingface (26GB) = 125GB freed.

---

## Containerization Notes

### What needs to happen
1. **ollama in a container**: Run ollama serve in Docker with GPU passthrough. On macOS with Apple Silicon, this means either running natively (current) or using a Linux VM with GPU passthrough (not straightforward on macOS). Alternatively, keep ollama on the host and expose port 11434 to containers via `host.docker.internal`.

2. **model-service in a container**: Pure Python (FastAPI/uvicorn), straightforward to containerize. Source at /Users/sam/Documents/services. Needs network access to ollama and mlx-vlm backends.

3. **mlx-vlm in a container**: Requires Apple Silicon + Metal. Cannot run in a Linux container on macOS Docker. Must remain native on the host or move to a dedicated Apple Silicon server. Expose via port mapping.

4. **eval-service in a container**: Pure Python, easy to containerize. Source at /Users/sam/Documents/Projects/eval-service.

### Recommended architecture (contained)
```
Host (macOS, Apple Silicon)
  ├─ ollama (native, port 11434) — needs Metal for MLX acceleration
  ├─ mlx-vlm (native, port 8801) — needs Metal for vision model
  └─ Docker
       ├─ model-service container (port 8800) → host ollama + mlx-vlm
       ├─ eval-service container (port 8802) → model-service
       └─ satellite agent containers → model-service via Caddy
```

Ollama and mlx-vlm stay native (Metal/GPU access). Everything else containerizes.

### Connection leak (ticket #81, still open)
Model-service accumulates stale connections to ollama from interrupted streams. After ~6 hermes turns, ollama stops responding. Partially fixed with connection pool limits but still occurs under load. Needs proper stream cleanup on client disconnect. **This is the #1 stability issue** and should be fixed before or during containerization.

### Key env vars for agents
```
OPENAI_BASE_URL=http://host.docker.internal:8800/v1
OPENAI_API_KEY=<model-service-key>
```

---

## Future Work

### Short term
- Containerize model-service and eval-service (easy wins)
- Set up proper process management for ollama + mlx-vlm (systemd or launchd replacement)
- Fix connection leak (#81) during containerization
- Caddy watchdog — external URLs (models-stg.samtg.xyz) need auto-restart

### Medium term
- Model allocation service: agents request models, service manages GPU memory and loading
- Auto-scaling: evict idle models, preload based on agent demand
- Multi-model support: multiple models loaded simultaneously with memory budgeting
- Model versioning: track which model version each agent is using

### Longer term
- Multi-node: model service spans multiple Apple Silicon machines
- Model fine-tuning pipeline: agents can request fine-tuned variants
- Inference caching: cache common prompts/responses across agents
- Cost tracking: per-agent model usage metrics
