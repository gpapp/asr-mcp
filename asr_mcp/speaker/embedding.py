import hashlib
import logging
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

logger = logging.getLogger("asr_mcp.speaker.embedding")

_cpu_embedding_cache: dict[str, ort.InferenceSession] = {}


def _run_with_cpu_fallback(session, feed, output_names):
    try:
        return session.run(output_names, feed)
    except RuntimeError as e:
        if "Failed to allocate memory" in str(e):
            logger.warning("GPU OOM on embedding, falling back to CPU")
            model_path = session.get_modelmeta().model_path
            if model_path not in _cpu_embedding_cache:
                cpu_so = ort.SessionOptions()
                cpu_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                _cpu_embedding_cache[model_path] = ort.InferenceSession(
                    model_path, sess_options=cpu_so, providers=["CPUExecutionProvider"],
                )
            return _cpu_embedding_cache[model_path].run(output_names, feed)
        raise


def extract_embedding(
    waveform: torch.Tensor,
    sample_rate: int,
    embedding_session,
) -> np.ndarray:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    from asr_mcp.speaker.audio import extract_fbank
    fbank = extract_fbank(waveform, sample_rate)

    # CMN (Cepstral Mean Normalization)
    fbank = fbank - fbank.mean(dim=0, keepdim=True)

    fbank_np = fbank.numpy().astype(np.float32)
    if fbank_np.ndim == 2:
        fbank_np = fbank_np[np.newaxis]

    input_name = embedding_session.get_inputs()[0].name
    output_name = embedding_session.get_outputs()[0].name

    embedding = _run_with_cpu_fallback(
        embedding_session, {input_name: fbank_np}, [output_name]
    )[0]

    # Mean pool and L2 normalize
    if embedding.ndim == 3:
        embedding = embedding.mean(axis=1)
    embedding = embedding.reshape(1, -1) if embedding.ndim == 1 else embedding
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / (norm + 1e-8)

    return embedding.squeeze().astype(np.float32)


def batch_embed_files(
    waveforms: list,
    sample_rates: list[int],
    durations: list[float],
    embedding_session,
    block_sec: float = 600.0,
) -> list[Optional[np.ndarray]]:
    from asr_mcp.speaker.audio import extract_fbank

    results = [None] * len(waveforms)
    block_samples = int(block_sec * 16000)

    batch_fbanks = []
    batch_indices = []

    for i, (wf, sr, dur) in enumerate(zip(waveforms, sample_rates, durations)):
        if wf is None or dur < 0.5:
            results[i] = np.zeros(192, dtype=np.float32)
            continue

        if wf.dim() == 1:
            wf = wf.unsqueeze(0)
        fbank = extract_fbank(wf, sr)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)

        batch_fbanks.append(fbank)
        batch_indices.append(i)

        total_samples = sum(f.shape[0] for f in batch_fbanks)
        if total_samples >= block_samples or i == len(waveforms) - 1:
            if batch_fbanks:
                combined = torch.cat(batch_fbanks, dim=0)
                fbank_np = combined.numpy().astype(np.float32)[np.newaxis]

                input_name = embedding_session.get_inputs()[0].name
                output_name = embedding_session.get_outputs()[0].name
                embedding = _run_with_cpu_fallback(
                    embedding_session, {input_name: fbank_np}, [output_name]
                )[0]

                if embedding.ndim == 3:
                    emb_mean = embedding.mean(axis=1)
                else:
                    emb_mean = embedding
                emb_mean = emb_mean.reshape(1, -1) if emb_mean.ndim == 1 else emb_mean
                norm = np.linalg.norm(emb_mean, axis=1, keepdims=True)
                emb_mean = emb_mean / (norm + 1e-8)

                for j, idx in enumerate(batch_indices):
                    results[idx] = emb_mean[j].astype(np.float32)

                batch_fbanks.clear()
                batch_indices.clear()

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
