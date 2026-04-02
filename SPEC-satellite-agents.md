# Spec: Satellite Agents

Isolated Hermes agent instances running in Docker containers on the SAMcloud network,
consuming shared model services from slice-test.

**Status**: Draft / staging-test
**Filed**: 2026-04-03
**Author**: claude-services

---

## 1. Overview

We have a working model service (`ollama-manager`) that manages GPU memory via SAMcloud leases
and serves inference through an OpenAI-compatible API. The next step is letting autonomous agents
consume these services from isolated environments.

A **satellite agent** is:
- A Docker container running Hermes (NousResearch/hermes-agent)
- Enrolled on SAMcloud with its own identity
- Consuming model inference via HTTP to ollama-manager
- Isolated: own filesystem, processes, and credentials
- Discoverable: other agents and humans can find it via SAMcloud

This pattern separates **model hosting** (heavy, shared, lease-managed) from **agent runtime**
(lightweight, isolated, many instances).

---

## 2. Architecture

```
                         SAMcloud Registry
                        stg.samtg.xyz:9443
                       ┌───────────────────┐
                       │ services / leases  │
                       │ devices / tickets  │
                       └──┬──────────┬──┬──┘
                          │          │  │
           lease/health   │  enroll  │  │ enroll/health
           (ollama-mgr)   │  /health │  │ (agent-beta)
                          │          │  │
    slice-test            │          │  │
    ┌─────────────────────┤          │  │
    │                     │          │  │
    │ ollama-manager:8800 │          │  │
    │ ┌─────────────────┐ │          │  │
    │ │ /v1/chat/compl. │◄┼──────────┤  │
    │ │ /v1/completions │ │   HTTP   │  │
    │ │ /models/load    │◄┼──────────┼──┤
    │ └────────┬────────┘ │          │  │
    │          │          │          │  │
    │ ┌────────┴────────┐ │    ┌─────┴──┴──────────┐
    │ │ Ollama (MLX)    │ │    │  agent-alpha       │
    │ │ llama-server    │ │    │  ┌──────────────┐  │
    │ └─────────────────┘ │    │  │ Hermes       │  │
    └─────────────────────┘    │  │ OPENAI_BASE  │  │
                               │  │ =host:8800   │  │
                               │  │ terminal:    │  │
                               │  │  local       │  │
                               │  └──────────────┘  │
                               │  Docker container   │
                               └────────────────────┘

                               ┌────────────────────┐
                               │  agent-beta         │
                               │  (same pattern)     │
                               └────────────────────┘
```

---

## 3. Container Image

Base image: `python:3.11-slim` (or `nikolaik/python-nodejs:python3.11-nodejs20` if the agent
needs Node.js tooling for browser/web tools).

Installed on top:
- Hermes agent (from git clone or pip)
- `httpx` (for SAMcloud API calls from bootstrap script)
- `git`, `curl`, basic build tools
- A `samcloud-bootstrap.py` script at `/usr/local/bin/`

The container does **not** need Ollama, llama.cpp, or any model files.
It only talks HTTP to the host model service.

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install Hermes
RUN git clone https://github.com/NousResearch/hermes-agent /opt/hermes-agent \
    && cd /opt/hermes-agent && pip install -e .

# Bootstrap script
COPY samcloud-bootstrap.py /usr/local/bin/
COPY entrypoint.sh /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
```

---

## 4. SAMcloud Enrollment

Each satellite agent gets a pre-provisioned SAMcloud identity. The admin creates it before
the container starts.

### Admin provisions (one-time)

```bash
# Create an agent user on SAMcloud
curl -X POST "$SC_BASE/users/invite" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"username": "agent-alpha", "role": "agent", "scopes": ["device:slice-test"]}'

# Returns an invite token -> activate it to get the permanent agent token
```

### Container startup (automatic)

The entrypoint script:
1. Reads `SC_TOKEN` from environment
2. Registers itself as a service on SAMcloud (optional — agent user is sufficient)
3. Begins health heartbeat (every 60s)
4. Launches Hermes

```bash
#!/bin/bash
# entrypoint.sh

# Register with SAMcloud (optional — for service discovery)
python3 /usr/local/bin/samcloud-bootstrap.py

# Start Hermes
exec hermes --no-interactive
```

---

## 5. Hermes Configuration

Inside the container, Hermes is configured to use ollama-manager as its LLM backend.

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `OPENAI_BASE_URL` | `http://host.docker.internal:8800/v1` | Routes inference to ollama-manager |
| `OPENAI_API_KEY` | `sk-not-needed` | Placeholder (ollama-manager has no auth) |
| `LLM_MODEL` | `qwen3-32b` | Model name (partial match works) |
| `SC_TOKEN` | `sc_agent_<hash>` | SAMcloud agent token |
| `SC_BASE` | `https://stg.samtg.xyz:9443/api/v1` | SAMcloud API endpoint |
| `HERMES_INFERENCE_PROVIDER` | (unset or "auto") | Hermes auto-detects custom endpoint |

### cli-config.yaml

```yaml
model:
  default: "qwen3-32b"
  provider: "auto"

terminal:
  backend: "local"    # local inside the container = already sandboxed
  cwd: "/workspace"
  timeout: 180
```

The terminal backend is `local` because the container itself IS the sandbox.
No nested Docker needed. Commands the agent runs execute inside its own container.

### How Hermes resolves the endpoint

Hermes' `runtime_provider.py` resolution chain:
1. Sees `OPENAI_BASE_URL` is set and is not an openrouter.ai URL
2. Uses `OPENAI_API_KEY` for auth (our placeholder)
3. Makes standard OpenAI API calls to `http://host.docker.internal:8800/v1`
4. ollama-manager's partial model matching resolves `qwen3-32b` to the loaded model

No Hermes code changes required.

---

## 6. Networking

### Container -> Model Service

`host.docker.internal` resolves to the Docker host automatically on macOS.
On Linux, add `--add-host=host.docker.internal:host-gateway` to `docker run`.

ollama-manager binds to `0.0.0.0:8800`, so it's reachable from containers.

### Container -> SAMcloud

Outbound HTTPS to `stg.samtg.xyz:9443`. SAMcloud uses self-signed certs,
so the bootstrap script needs `verify=False` (or install the CA cert in the container).

### Docker run flags

```bash
docker run -d \
  --name agent-alpha \
  -e SC_TOKEN=sc_agent_xxx \
  -e SC_BASE=https://stg.samtg.xyz:9443/api/v1 \
  -e OPENAI_BASE_URL=http://host.docker.internal:8800/v1 \
  -e OPENAI_API_KEY=sk-not-needed \
  -e LLM_MODEL=qwen3-32b \
  --cap-drop=ALL \
  --cap-add=DAC_OVERRIDE --cap-add=CHOWN --cap-add=FOWNER \
  --security-opt=no-new-privileges \
  --pids-limit=256 \
  samcloud/hermes-agent:latest
```

---

## 7. Identity & Credentials

Each agent gets its own SAMcloud user and token. Agents do **not** receive:
- ollama-manager's `SC_TOKEN` (the model service manages its own leases)
- Any other service's credentials

The credential boundary:

```
Agent ──HTTP──> ollama-manager ──lease──> SAMcloud
  │                                          ▲
  └──────────enrollment/health───────────────┘
```

Agents call `/v1/chat/completions` on ollama-manager as unauthenticated HTTP.
ollama-manager handles lease acquisition internally.

Agents use their own `SC_TOKEN` for SAMcloud operations (health reporting,
service discovery, filing tickets).

---

## 8. Multi-Agent Coordination

### Model sharing

Multiple agents share the same model service. ollama-manager tracks
`request_count` and `last_used` per model. The cooldown timer resets on
**any** request from **any** agent.

As long as at least one agent uses a model every 5 minutes, it stays loaded.
When all agents go idle, the model unloads and the lease is released.

### Contention

If two agents need different models and GPU memory is insufficient for both:
- The first model loads successfully
- The second `/models/load` may fail if SAMcloud denies the lease (insufficient memory)
- There is currently no queuing — the request fails with HTTP 409

### No direct lease access

Agents do **not** request leases. They are pure HTTP consumers.
This simplifies the agent and keeps lease management centralized.

---

## 9. Agent Lifecycle

### Create

```bash
docker create --name agent-alpha \
  -e SC_TOKEN=sc_agent_xxx \
  -e OPENAI_BASE_URL=http://host.docker.internal:8800/v1 \
  -e OPENAI_API_KEY=sk-not-needed \
  -e LLM_MODEL=qwen3-32b \
  samcloud/hermes-agent:latest
```

### Start

```bash
docker start agent-alpha
```

Entrypoint: bootstrap registers with SAMcloud, then starts Hermes.

### Stop

```bash
docker stop agent-alpha
```

Hermes handles SIGTERM. Health heartbeat stops. SAMcloud's health-bound
revocation (per ticket #42) will mark the service as stale.

### Destroy

```bash
docker rm agent-alpha
```

Container removed. If persistent volumes were mounted (`-v`), workspace data survives.

### Persistent workspaces

```bash
docker run -d \
  -v hermes-alpha-workspace:/workspace \
  -v hermes-alpha-home:/root/.hermes \
  ...
```

Named volumes persist across container restarts and recreations.
Agent memory, skills, and conversation history survive.

---

## 10. Bootstrap Script

`samcloud-bootstrap.py` runs on container start:

```python
#!/usr/bin/env python3
"""Register this agent with SAMcloud and configure Hermes."""

import os
import httpx

SC_TOKEN = os.environ["SC_TOKEN"]
SC_BASE = os.environ.get("SC_BASE", "https://stg.samtg.xyz:9443/api/v1")
AGENT_NAME = os.environ.get("AGENT_NAME", os.uname().nodename)

client = httpx.Client(
    base_url=SC_BASE,
    headers={"Authorization": f"Bearer {SC_TOKEN}"},
    verify=False,
)

# Report initial health
client.post(f"/services/{AGENT_NAME}/health")

print(f"Bootstrap complete: {AGENT_NAME} registered with SAMcloud")
```

The actual registration (creating the service) is done by the admin during provisioning.
The bootstrap script just confirms connectivity and reports initial health.

---

## 11. Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| No auth on ollama-manager | Any container on the Docker network can infer | Staging only; add auth middleware for production |
| Lease renewal gap | Brief moment during renewal where lease doesn't exist | Under low contention this is fine; add atomic renewal for production |
| No model request queuing | Second model load fails if GPU memory full | Agent gets HTTP 409; can retry after cooldown frees memory |
| Single host only | All containers must reach host.docker.internal | Future: multi-host with SAMcloud service discovery |
| Self-signed certs | Containers need verify=False for SAMcloud | Install CA cert in image for production |

---

## 12. Future Work

### Phase 2: Gateway Integration
Each satellite agent runs Hermes in gateway mode, connecting to its own Telegram/Discord/Slack
bot. Users chat with specific agents through their platform identity.

### Phase 3: Cross-Agent Messaging
Agents discover each other via SAMcloud service registry and communicate through a shared
message bus or SAMcloud's event stream (`GET /api/v1/events`).

### Phase 4: Production Hardening
- Auth middleware on ollama-manager (API key per agent)
- Atomic lease renewal (extend, not release-and-re-request)
- Model request queuing with priority
- Resource quotas per agent (max memory, max concurrent models)
- Multi-host model services with SAMcloud routing

### Phase 5: Agent Orchestration
- Automated provisioning (create agent, enroll, configure, start — single command)
- Agent health monitoring and auto-restart
- Scaling: spin up/down agents based on demand
- Agent-to-agent task delegation
