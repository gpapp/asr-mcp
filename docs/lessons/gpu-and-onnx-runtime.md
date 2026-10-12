# GPU, ONNX Runtime and arena discipline

Lessons 5, 6, 27 of `AGENTS.md`. GPU OOM handling, exception types, and arena caps for a 4 GB card.


### 5. ONNX Runtime Error Handling
- `onnxruntime` has NO `ORTRuntimeError` attribute. Catch `Exception` and gate on `is_gpu_oom(e)` — ORT ≥1.22 raises `onnxruntime_pybind11_state.RuntimeException` which inherits `Exception`, NOT `RuntimeError`, so `except RuntimeError` silently never fires.
- `is_gpu_oom` (model_state.py) matches, case-lowered: `failed to allocate memory`, `out of memory`, `out_of_memory`, `available memory of`, `smaller than requested bytes` — the last two exist because arena-cap messages (`Available memory of 0 is smaller than requested bytes of 97517568`) don't contain "Failed to allocate memory".
- Pattern: `except Exception as e: if not is_gpu_oom(e): raise` then recover.
- **Why**: Every OOM was silently re-raised because (a) the except clause itself threw `AttributeError`, then (b) `except RuntimeError` never matched ORT's exception type, then (c) the arena-cap message didn't match the string.

### 6. Embedding GPU OOM → CPU Fallback with Chunking
- ECAPA-TDNN512 embedding on CUDA can OOM after encoder has consumed VRAM.
- `_run_with_cpu_fallback()` catches `Exception` gated by `is_gpu_oom(e)` (NOT `RuntimeError` — see lesson 5) and retries on CPU with cached sessions.
- For long audio: `extract_embedding()` chunks fbank into 60s pieces, embeds each, averages, L2-normalizes.
- `batch_embed_files()` uses `block_sec=60.0` (not 600.0) to prevent huge ONNX calls.


### 27. GPU Arena Discipline: Cap → Shrink Per Run → Reload Clears First
- Caps set at session creation (`_cuda_provider_options`): encoder `gpu_memory_limit_gb×0.625`, embedding `min(÷4, 768 MiB)`, `cudnn_conv_algo_search: HEURISTIC` (EXHAUSTIVE allocates huge workspaces).
- Every GPU run passes `GPU_SHRINK_RUN_OPTIONS` (`memory.enable_memory_arena_shrinkage=gpu:0`) — without it the arena only releases on full session reload.
- Cohere encoder reload (`reload_encoder` in `transcribers/cohere.py`) / embedding reload (`reload_embedding_session` in `model_loader.py`): set the field to `None` + `gc.collect()` BEFORE constructing the replacement — otherwise old arena (alive) + new session (allocating) peak together.
- Encoder OOM escalation: reload fresh arena → retry GPU once → CPU fallback. Embedding OOM: straight to cached CPU session.

### 44. A shared card fails at `cublasCreate()`, and no gate recognised it
Measured 2026-10-01 12:55 on the GTX 1650 (4096 MiB) with two `ollama` models
resident (2316 + 350 MiB). The service could not create a session at all:

```
CUBLAS failure 3: CUBLAS_STATUS_ALLOC_FAILED ; GPU=0 ; expr=cublasCreate(&cublas_handle);
[ERROR] asr_mcp.core.model_state: Failed to reload models: [ONNXRuntimeError] ...
[ERROR] asr_mcp.transcribers.whisper: Whisper transcription failed: cuBLAS failed with status CUBLAS_STATUS_ALLOC_FAILED
```

Two defects, both invisible until the card was not ours alone:

1. **The wording was unknown.** `is_gpu_oom` matched only allocator phrasings,
   and a cuBLAS failure says nothing about an allocation of *ours* — it happens
   at handle creation, before a single weight is placed. Every backend that
   consults the gate (cohere `_run_encoder`, qwen3 chunking, whisper, embedding
   `_run_with_cpu_fallback`) therefore skipped its CPU recovery and returned an
   error instead. Fixed by adding `cublas` (loosely, on purpose — the recovery
   is identical for any cuBLAS failure, so a broad match costs a slower turn and
   never a wrong answer), plus the CUDA driver's own
   `cudaErrorMemoryAllocation`. Lesson 5's rule — one shared predicate, matched
   by string — only works if the marker list is complete.
2. **The load-time escalation had no caller.** `reload_embedding_session(force_cpu=True)`
   existed and was never invoked from anywhere, so a GPU failure during session
   *construction* aborted `reload_models` outright: no embedding session, and
   every diarization/attribution request dead for as long as the other workload
   stayed resident. The runtime path (`_run_with_cpu_fallback`) was already
   fail-safe; only construction was not. `load_embedding_session` now catches,
   gates on `is_gpu_oom`, logs `nvidia-smi`, and builds a CPU session.

**The trap in writing that fix**: the obvious guard
`if "CPUExecutionProvider" in emb_providers: raise` is always true. ORT appends
the CPU provider as a fallback, so the GPU list is
`[("CUDAExecutionProvider", {...}), "CPUExecutionProvider"]`. Ask
`_requests_cuda(providers)` — does an entry *ask* for CUDA — never whether the
CPU EP is mentioned.

**Not fixed, on purpose**: `_cuda_provider_options` derives its arena cap from
`TRANSCRIBE_GPU_MEMORY_LIMIT_GB` (the card's *total*), not from free VRAM. On a
shared card that over-promises, and a cap computed from `nvidia-smi` at session
creation would be the honest number. Left alone because it changes every
backend's memory budget and needs measuring, not guessing.
