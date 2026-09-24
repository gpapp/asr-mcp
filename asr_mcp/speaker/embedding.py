import hashlib
import logging
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

from asr_mcp.core.model_state import is_gpu_oom, log_gpu_memory, GPU_SHRINK_RUN_OPTIONS

logger = logging.getLogger("asr_mcp.speaker.embedding")

_cpu_embedding_session: Optional[ort.InferenceSession] = None


def _get_cpu_embedding_session() -> ort.InferenceSession:
    global _cpu_embedding_session
    if _cpu_embedding_session is None:
        from asr_mcp.core.model_loader import ensure_embedding_model, get_session_options
        from asr_mcp.config.settings import get_settings
        from asr_mcp.core.model_state import state
        settings = getattr(state, "settings", None) or get_settings()
        emb_path = str(ensure_embedding_model(settings))
        cpu_so = get_session_options(settings)
        logger.warning("Loading CPU embedding fallback session: %s", emb_path)
        _cpu_embedding_session = ort.InferenceSession(
            emb_path, sess_options=cpu_so, providers=["CPUExecutionProvider"],
        )
    return _cpu_embedding_session


def _run_with_cpu_fallback(session, feed, output_names):
    try:
        return session.run(output_names, feed, run_options=GPU_SHRINK_RUN_OPTIONS)
    except Exception as e:
        if is_gpu_oom(e):
            logger.warning("GPU OOM on embedding, falling back to CPU: %s", e)
            log_gpu_memory("embedding OOM fallback")
            cpu_sess = _get_cpu_embedding_session()
            return cpu_sess.run(output_names, feed)
        raise


def extract_embedding(
    waveform: torch.Tensor,
    sample_rate: int,
    embedding_session,
    max_chunk_sec: float = 60.0,
) -> np.ndarray:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    from asr_mcp.speaker.audio import extract_fbank, generate_sliding_windows
    windows, _ = generate_sliding_windows(waveform, sample_rate, window_sec=3.0, stride_sec=1.5)

    if not windows:
        return np.zeros(192, dtype=np.float32)

    all_fbanks = []
    target_length = 4800

    for w in windows:
        chunk_duration = w.shape[-1] / sample_rate
        if chunk_duration >= 0.5:
            if w.shape[-1] < target_length:
                w = torch.nn.functional.pad(w, (0, target_length - w.shape[-1]))
            fb = extract_fbank(w, sample_rate)  # [1, T, 80]
            all_fbanks.append(fb)

    if not all_fbanks:
        # Fallback to direct fbank if audio was too short for 0.5s windows
        if waveform.shape[-1] < target_length:
            w = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[-1]))
        else:
            w = waveform
        all_fbanks.append(extract_fbank(w, sample_rate))

    max_len = max(fb.shape[1] for fb in all_fbanks)
    padded_fbanks = []
    for fb in all_fbanks:
        if fb.shape[1] < max_len:
            padded_fbanks.append(torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1])))
        else:
            padded_fbanks.append(fb)

    batch = torch.stack(padded_fbanks, dim=0)  # [N, 1, max_len, 80]
    cmn_batch = batch - batch.mean(dim=2, keepdim=True)
    batch_np = cmn_batch.squeeze(1).numpy().astype(np.float32)  # [N, max_len, 80]

    input_name = embedding_session.get_inputs()[0].name
    output_name = embedding_session.get_outputs()[0].name

    computed_chunks = []
    batch_size = 16
    for b_start in range(0, len(batch_np), batch_size):
        b_inp = batch_np[b_start:b_start + batch_size]
        out_c = _run_with_cpu_fallback(
            embedding_session, {input_name: b_inp}, [output_name]
        )[0]
        if out_c.ndim == 3:
            out_c = out_c.mean(axis=1)
        computed_chunks.append(out_c)

    embeddings = np.concatenate(computed_chunks, axis=0) if computed_chunks else np.empty((0, 192), dtype=np.float32)

    mean_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    return mean_emb.astype(np.float32)


def batch_embed_files(
    waveforms: list,
    sample_rates: list[int],
    durations: list[float],
    embedding_session,
    block_sec: float = 60.0,
) -> list[Optional[np.ndarray]]:
    from asr_mcp.speaker.audio import extract_fbank, generate_sliding_windows

    target_length = 4800
    min_chunk_sec = 0.5
    input_name = embedding_session.get_inputs()[0].name
    output_name = embedding_session.get_outputs()[0].name

    all_fbanks: list[torch.Tensor] = []
    slot_map: list[int] = []

    for file_idx, (wf, sr, dur) in enumerate(zip(waveforms, sample_rates, durations)):
        if wf is None or dur < 0.3:
            continue
        if wf.dim() == 1:
            wf = wf.unsqueeze(0)
        windows, _ = generate_sliding_windows(wf, sr, window_sec=3.0, stride_sec=1.5)
        for w in windows:
            if w.shape[-1] / sr < min_chunk_sec:
                continue
            if w.shape[-1] < target_length:
                w = torch.nn.functional.pad(w, (0, target_length - w.shape[-1]))
            fb = extract_fbank(w, sr)  # [1, T, 80]
            all_fbanks.append(fb)
            slot_map.append(file_idx)

    if not all_fbanks:
        return [np.zeros(192, dtype=np.float32)] * len(waveforms)

    window_sec = 3.0
    block_size = min(16, max(1, int(block_sec / window_sec)))
    raw_embs_all = []

    for block_start in range(0, len(all_fbanks), block_size):
        block_fbanks = all_fbanks[block_start: block_start + block_size]
        max_len = max(fb.shape[1] for fb in block_fbanks)
        padded = []
        for fb in block_fbanks:
            if fb.shape[1] < max_len:
                padded.append(torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1])))
            else:
                padded.append(fb)
        batch = torch.stack(padded, dim=0)  # [N, 1, max_len, 80]
        cmn_batch = batch - batch.mean(dim=2, keepdim=True)
        batch_np = cmn_batch.squeeze(1).numpy().astype(np.float32)

        block_result = _run_with_cpu_fallback(
            embedding_session, {input_name: batch_np}, [output_name]
        )[0]
        if block_result.ndim == 3:
            block_result = block_result.mean(axis=1)
        raw_embs_all.append(block_result)

    all_embs = np.concatenate(raw_embs_all, axis=0)
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    all_embs = all_embs / np.maximum(norms, 1e-12)

    n_files = len(waveforms)
    file_embs: list[list[np.ndarray]] = [[] for _ in range(n_files)]
    for k, file_idx in enumerate(slot_map):
        file_embs[file_idx].append(all_embs[k])

    results: list[Optional[np.ndarray]] = []
    for file_idx in range(n_files):
        group = file_embs[file_idx]
        if not group:
            results.append(np.zeros(192, dtype=np.float32))
            continue
        mean_emb = np.mean(np.stack(group), axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        results.append(mean_emb.astype(np.float32))

    return results


def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(emb)
    if norm > 0:
        return emb / norm
    return emb


def compute_pitch(waveform: torch.Tensor, sample_rate: int = 16000) -> tuple[float, float]:
    if waveform.dim() > 1:
        waveform = waveform.squeeze()
    audio_np = waveform.numpy().astype(np.float32)

    frame_len = int(0.03 * sample_rate)
    hop_len = int(0.01 * sample_rate)
    min_lag = int(sample_rate / 400)
    max_lag = int(sample_rate / 60)

    pitches = []
    for start in range(0, len(audio_np) - frame_len, hop_len):
        frame = audio_np[start:start + frame_len]
        frame = frame - frame.mean()
        energy = np.sum(frame ** 2)
        if energy < 1e-8:
            continue
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]
        if max_lag > len(corr):
            continue
        search = corr[min_lag:max_lag]
        if len(search) == 0:
            continue
        peak_idx = np.argmax(search) + min_lag
        if corr[peak_idx] > 0.3 * corr[0]:
            freq = sample_rate / peak_idx
            pitches.append(freq)

    if not pitches:
        return (0.0, 0.0)
    return (float(np.median(pitches)), float(np.std(pitches)))


def compute_energy(waveform: torch.Tensor) -> float:
    if waveform.dim() > 1:
        waveform = waveform.squeeze()
    audio_np = waveform.numpy().astype(np.float32)
    return float(np.sqrt(np.mean(audio_np ** 2)))
