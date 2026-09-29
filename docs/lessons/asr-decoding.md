# ASR decoding internals (Cohere ONNX)

Lessons 7, 8, 12, 13, 14, 15, 16 of `AGENTS.md`. Prompt tokens, mel features, KV caches, timestamps and windowing.


### 7. Encoder Chunking for Long Audio
- Mel spectrogram must be split into overlapping windows (MAX_ENCODER_SEC=30s, 25% overlap) for audio >30s.
- Each window encoded independently, outputs concatenated along sequence dimension.
- Decoder receives `encoder_hidden_states` as direct input (not just KV caches).

### 8. Decoder KV Cache Name Mapping
- HuggingFace ONNX outputs use `present.{i}.decoder.key` but decoder inputs expect `past_key_values.{i}.decoder.key`.
- After step 0, map output names: `name.replace("present.", "past_key_values.")`.
- Cross-attention KV caches initialized as empty (seq_len=0).


### 12. Cohere Prompt Tokens: Exact Order via token_to_id (Not tokenizer.encode)
- Build the prompt with **direct `token_to_id` lookups** (`if t in token_to_id`), never `tokenizer.encode()` per token.
- Exact order: `<|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|lang|> <|lang|> <|pnc|> <|noitn|> <|timestamp|> <|nodiarize|>` — note the **duplicate language token** and startofcontext FIRST.
- `eos_id = token_to_id["endoftext"]` — never hardcode (was wrongly `3`).
- **Why**: Missing `<|startofcontext|>`, a single language token, or wrong order made the model emit punctuation-only garbage (`,,`, `e`, `at`) even though the encoder/decoder ran fine.

### 13. Mel Features Must Match the Reference Pipeline Exactly
- Required: `nperseg=512` hann, `noverlap=352` (hop 160), `mode='magnitude'`, `power_to_db(ref=np.max)`, NO dither, pre-emphasis, per-mel-band z-norm. Returns `[T,128]`.
- **Why**: A 400-sample window + `np.log(mel+1e-8)` variant produced off-distribution features → same garbage-text symptom as a bad prompt. Prompt AND features must both be right; matching one hides nothing.
- Magnitude vs power is a constant factor that cancels under z-norm — but window length and log-vs-power_to_db do NOT cancel.

### 14. Decoder attention_mask = Full past + current Length
- `attention_mask = ones(batch, past_seq_len + tokens_this_call + encoder_seq_len)`; `position_ids` offset by `past_seq_len`.
- Masking only the current tokens while position grows desynchronizes RoPE/positions across multi-step decode.

### 15. Cohere Timestamps Are `<|spltokenN|>` Tokens, Not `<|1.23|>`
- `SPLIT_TOKEN_BASE = token_to_id["<|spltoken0|>"]`, 34 bins; segment end = `audio_duration * (token_id - BASE) / 34`.
- The regex `<\|(\d+\.?\d*)\|>` NEVER matches these tokens → without split-token handling every result is a single segment `{start:0, end:full_duration}` (the observed symptom).
- Skip other `<|...|>` specials during decode; map `▁` → space; run `clean_transcript` per flushed segment.

### 16. Decode Long Audio in ≤30s Windows (Encoder Chunking Alone Is Not Enough)
- One decode over a 951s turn ends in early EOS/max_new_tokens → text truncated after the first ~30s of speech even though encoder chunking ran.
- `_transcribe_windowed` + `_plan_window_bounds` (energy-snap cuts, min 300-frame tail) fix this. Streaming/KV-bridge paths set `_no_window=True` (bridging already chunks).
- **Never swallow partial window errors**: if window 2+ fails but window 0 has text, still set `result["error"]` — otherwise output looks like benign truncation. (`TranscribeResult.error` is optional; both text and error may be present.)
