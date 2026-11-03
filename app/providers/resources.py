"""Serialize local inference and release other models on a 16GB shared-memory Mac."""
import gc
import json
import os
import threading
import urllib.request
import weakref
from contextlib import contextmanager

LOCK = threading.RLock()
PROVIDERS = weakref.WeakSet()
MANAGED_MODELS = {"qwen3:4b", "qwen3-vl:4b", "qwen3-vl:4b-instruct", "qwen3.5:4b", "qwen3-embedding:0.6b"}


def register(provider):
    PROVIDERS.add(provider)


def unload_ollama():
    # Only release PhotoRecall's models, never stop the user's Ollama service.
    for name in tuple(MANAGED_MODELS):
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                data=json.dumps({"model": name, "keep_alive": 0}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5):
                pass
        except OSError:
            pass


@contextmanager
def inference(owner):
    with LOCK:
        if os.environ.get("PHOTORECALL_KEEP_MODELS") == "1":
            yield
            return
        for provider in list(PROVIDERS):
            if provider is not owner:
                provider.unload()
        if owner != "ollama" and os.environ.get("PHOTORECALL_DISABLE_OLLAMA") != "1":
            # /api/ps avoids loading an absent model merely to unload it.
            try:
                with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=1) as response:
                    loaded = json.load(response).get("models", [])
                if any(m.get("name") in MANAGED_MODELS for m in loaded):
                    unload_ollama()
            except OSError:
                pass
        yield


def empty_cache():
    gc.collect()
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
