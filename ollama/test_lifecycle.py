#!/usr/bin/env python3
"""
End-to-end test of the Ollama + SAMcloud resource lease lifecycle.

Flow:
  1. Check resource dashboard (memory state)
  2. Pull a small model (qwen2.5:1.5b) via Ollama
  3. Request a memory lease from SAMcloud
  4. Load the model
  5. Run a test inference
  6. Unload the model
  7. Release the lease
  8. Verify resource state returns to baseline
"""

import sys
import time
import json

from samcloud import SamcloudClient
from ollama_client import OllamaClient, estimate_memory_mb

SC_TOKEN = "sc_agent_448090436817362f5250c0d0f83bef53"
RESOURCE_ID = "slice-test/gpu-0"
SERVICE_ID = "slice-test/ollama-manager"
TEST_MODEL = "qwen2.5:1.5b"


def step(n, msg):
    print(f"\n{'='*60}")
    print(f"  Step {n}: {msg}")
    print(f"{'='*60}")


def main():
    sc = SamcloudClient(token=SC_TOKEN)
    ollama = OllamaClient()

    # 1. Check baseline
    step(1, "Check resource dashboard")
    dash = sc.resource_dashboard()
    for r in dash:
        if r["id"] == RESOURCE_ID:
            print(f"  GPU: {r['label']}")
            print(f"  Memory: {r['memory_pct']}% used")
            print(f"  Compute: {r['compute_pct']}%")
            print(f"  Health: {r['health']}")
            print(f"  Available: {r['available_memory_mb']}MB")
            break

    leases_before = sc.list_leases(resource=RESOURCE_ID)
    print(f"  Active leases: {len(leases_before)}")

    # 2. Pull model
    step(2, f"Pull model: {TEST_MODEL}")
    local = [m["name"] for m in ollama.list_models()]
    if TEST_MODEL in local or f"{TEST_MODEL}:latest" in local:
        print(f"  Already pulled")
    else:
        print(f"  Pulling {TEST_MODEL}...")
        for progress in ollama.pull_model(TEST_MODEL):
            status = progress.get("status", "")
            if "completed" in progress and "total" in progress:
                pct = int(progress["completed"] / max(progress["total"], 1) * 100)
                print(f"  {status}: {pct}%", end="\r")
            elif status:
                print(f"  {status}")
        print(f"\n  Pull complete")

    # 3. Request lease
    step(3, "Request memory lease from SAMcloud")
    memory_mb = estimate_memory_mb(TEST_MODEL)
    print(f"  Estimated memory: {memory_mb}MB")

    lease = sc.request_lease(
        resource_id=RESOURCE_ID,
        service_id=SERVICE_ID,
        memory_mb=memory_mb,
        purpose=f"model:{TEST_MODEL}",
        ttl_seconds=600,
    )
    status_code = lease.pop("status_code", 0)
    lease_id = lease.get("id") or lease.get("lease_id")
    print(f"  Response: {status_code}")
    print(f"  Lease ID: {lease_id}")
    print(f"  Full response: {json.dumps(lease, indent=2, default=str)}")

    # 4. Load model
    step(4, "Load model into Ollama (warm up)")
    t0 = time.time()
    ollama.load_model(TEST_MODEL)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    running = ollama.list_running()
    print(f"  Running models: {[m['name'] for m in running]}")
    for m in running:
        if TEST_MODEL in m.get("name", ""):
            size_mb = m.get("size", 0) / 1024 / 1024
            print(f"  Actual VRAM: {size_mb:.0f}MB")

    # 5. Test inference
    step(5, "Run test inference")
    t0 = time.time()
    response_text = ""
    for chunk in ollama.generate(TEST_MODEL, "What is 2+2? Answer in one word."):
        if "response" in chunk:
            response_text += chunk["response"]
        if chunk.get("done"):
            break
    inference_time = time.time() - t0
    print(f"  Response: {response_text.strip()}")
    print(f"  Time: {inference_time:.1f}s")

    # 6. Unload model
    step(6, "Unload model from Ollama")
    ollama.unload_model(TEST_MODEL)
    time.sleep(1)
    running = ollama.list_running()
    print(f"  Running models after unload: {[m['name'] for m in running]}")

    # 7. Release lease
    step(7, f"Release lease {lease_id}")
    if lease_id:
        result = sc.release_lease(str(lease_id))
        print(f"  Release result: {json.dumps(result, indent=2, default=str)}")
    else:
        print(f"  No lease_id to release - checking response: {lease}")

    # 8. Verify
    step(8, "Verify resource state")
    leases_after = sc.list_leases(resource=RESOURCE_ID)
    print(f"  Active leases now: {len(leases_after)}")
    dash = sc.resource_dashboard()
    for r in dash:
        if r["id"] == RESOURCE_ID:
            print(f"  Memory: {r['memory_pct']}% used")
            print(f"  Health: {r['health']}")

    print(f"\n{'='*60}")
    print(f"  LIFECYCLE TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
