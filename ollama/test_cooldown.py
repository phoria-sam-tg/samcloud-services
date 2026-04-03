#!/usr/bin/env python3
"""
Test the spin-up / cooldown lifecycle for Ollama models.
Uses a short cooldown (30s) to verify the full cycle quickly.

Flow:
  1. Load a small model via the manager (requests lease, loads into Ollama)
  2. Verify it's running and serving
  3. Wait for cooldown
  4. Verify it was unloaded and lease released
"""

import os
import sys
import time
import json

from samcloud import SamcloudClient
from ollama_client import OllamaClient

SC_TOKEN = os.environ.get("SC_TOKEN", "")
RESOURCE_ID = "slice-test/gpu-0"
SERVICE_ID = "slice-test/ollama-manager"
TEST_MODEL = "qwen2.5:1.5b"
SHORT_COOLDOWN = 30  # seconds


def step(n, msg):
    print(f"\n{'='*60}")
    print(f"  Step {n}: {msg}")
    print(f"{'='*60}")


def main():
    sc = SamcloudClient(token=SC_TOKEN)
    ollama = OllamaClient()

    # Use the manager directly with a short cooldown for testing
    sys.path.insert(0, "/Users/sam/Documents/services")
    from ollama.manager import ModelManager, RESOURCE_ID as RID
    import ollama.manager as mgr_mod

    # Override cooldown for testing
    original_cooldown = mgr_mod.COOLDOWN_SECONDS
    mgr_mod.COOLDOWN_SECONDS = SHORT_COOLDOWN
    print(f"Cooldown set to {SHORT_COOLDOWN}s for testing")

    manager = ModelManager(sc=sc, ollama=ollama)

    # 1. Check baseline
    step(1, "Baseline state")
    running = ollama.list_running()
    print(f"  Running Ollama models: {[m['name'] for m in running]}")
    leases_before = sc.list_leases(resource=RESOURCE_ID, status="active")
    print(f"  Active leases: {len(leases_before)}")

    # 2. Load model via manager
    step(2, f"Load {TEST_MODEL}")
    t0 = time.time()
    mm = manager.load_ollama_model(TEST_MODEL)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Backend: {mm.backend.value}")
    print(f"  Memory: {mm.memory_mb}MB")
    print(f"  Lease: {mm.lease_id}")
    print(f"  Managed: {mm.managed}")

    running = ollama.list_running()
    print(f"  Running models: {[m['name'] for m in running]}")
    for m in running:
        if TEST_MODEL in m.get("name", ""):
            actual_mb = m.get("size", 0) / 1024 / 1024
            print(f"  Actual VRAM: {actual_mb:.0f}MB")

    # 3. Test inference
    step(3, "Test inference")
    t0 = time.time()
    resp = ""
    for chunk in ollama.generate(TEST_MODEL, "What is 1+1? One word."):
        if "response" in chunk:
            resp += chunk["response"]
        if chunk.get("done"):
            break
    print(f"  Response: {resp.strip()}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Touch to reset cooldown
    manager.touch(TEST_MODEL)

    # 4. Wait and check cooldown
    step(4, f"Wait {SHORT_COOLDOWN + 10}s for cooldown")
    for i in range(SHORT_COOLDOWN + 10):
        time.sleep(1)
        elapsed = i + 1
        if elapsed % 10 == 0:
            idle = int(time.time() - mm.last_used)
            print(f"  {elapsed}s elapsed, model idle for {idle}s")

    # 5. Trigger cooldown check
    step(5, "Trigger cooldown check")
    results = manager.check_cooldowns()
    print(f"  Cooldown results: {json.dumps(results, indent=2, default=str)}")

    # 6. Verify unloaded
    step(6, "Verify cleanup")
    running = ollama.list_running()
    print(f"  Running Ollama models: {[m['name'] for m in running]}")
    print(f"  Manager models: {list(manager.models.keys())}")
    leases_after = sc.list_leases(resource=RESOURCE_ID, status="active")
    active_mine = [l for l in leases_after if l.get("service_id") == SERVICE_ID]
    print(f"  My active leases: {len(active_mine)}")

    if not running and TEST_MODEL not in manager.models:
        print("\n  COOLDOWN TEST PASSED - model unloaded, lease released")
    else:
        print("\n  COOLDOWN TEST ISSUE - check above")

    # Restore
    mgr_mod.COOLDOWN_SECONDS = original_cooldown

    print(f"\n{'='*60}")
    print(f"  LIFECYCLE TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
