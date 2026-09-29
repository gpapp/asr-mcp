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
