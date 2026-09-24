import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("asr_mcp.speaker.matcher")

_DEFAULT_CFG = {
    "weights": {
        "embedding": 0.9,
        "pitch": 0.1,
    },
    "matching": {
        "accept_threshold": 0.35,
        "clear_winner_gap": 0.02,
        "embed_only_threshold": 0.16,
    },
    "normalization": {
        "pitch_hz_per_unit": 50,
        "confidence_max_distance": 0.5,
    },
}


def compute_distance(
    cluster_emb: List[float],
    cluster_pitch: float,
    cluster_energy: float,
    voiceprint: Dict,
    cfg: Dict = None,
    cluster_features: Dict = None,
    is_known_speaker: bool = False,
) -> Dict[str, float]:
    cfg = cfg or _DEFAULT_CFG
    weights = cfg.get("weights", _DEFAULT_CFG["weights"])
    norm_cfg = cfg.get("normalization", _DEFAULT_CFG["normalization"])

    vp_emb = np.array(voiceprint.get("embedding", []), dtype=np.float32)
    if len(vp_emb) == 0 or len(cluster_emb) == 0:
        return {
            "total": 1.0,
            "combined": 1.0,
            "embedding": 1.0,
            "emb_dist": 1.0,
            "pitch": 0.0,
            "pitch_dist": 0.0,
            "confidence": 0.0,
        }

    emb_dist = 1.0 - float(np.dot(cluster_emb, vp_emb) /
                           (np.linalg.norm(cluster_emb) * np.linalg.norm(vp_emb) + 1e-8))

    pitch_per = norm_cfg.get("pitch_hz_per_unit", 50)
    vp_pitch = voiceprint.get("pitch_hz", 0.0) or 0.0

    pitch_dist = abs(cluster_pitch - vp_pitch) / pitch_per if (pitch_per and cluster_pitch > 0 and vp_pitch > 0) else 0.0

    w_emb = weights.get("embedding", 0.9)
    w_pitch = weights.get("pitch", 0.1)
    total_w = w_emb + w_pitch
    total = (w_emb * emb_dist + w_pitch * min(pitch_dist, 1.0)) / (total_w if total_w > 0 else 1.0)

    if is_known_speaker:
        bias = cfg.get("matching", {}).get("known_speaker_margin_bias", 0.0)
        if not bias:
            bias = cfg.get("second_pass", {}).get("known_speaker_margin_bias", 0.0)
        if bias > 0:
            total = max(0.0, total - bias)

    conf_max_dist = norm_cfg.get("confidence_max_distance", 0.5)
    confidence = max(0.0, 1.0 - (total / conf_max_dist)) if conf_max_dist else 0.5

    return {
        "total": round(float(total), 4),
        "combined": round(float(total), 4),
        "embedding": round(float(emb_dist), 4),
        "emb_dist": round(float(emb_dist), 4),
        "pitch": round(float(pitch_dist), 4),
        "pitch_dist": round(float(pitch_dist), 4),
        "confidence": round(float(confidence), 4),
    }


def find_best_match(
    cluster_emb: List[float],
    cluster_pitch: float,
    cluster_energy: float,
    voiceprints: Dict[str, Dict],
    cfg: Dict = None,
    cluster_features: Dict = None,
) -> Tuple[Optional[str], float, float, Dict[str, Dict]]:
    distances = {}
    for name, vp in voiceprints.items():
        is_known = not (name.startswith("Speaker ") or name.startswith("SPEAKER ") or name == "OVERLAP")
        dist = compute_distance(
            cluster_emb, cluster_pitch, cluster_energy, vp, cfg, cluster_features, is_known_speaker=is_known
        )
        distances[name] = dist

    matches = [(name, d["combined"], d["confidence"]) for name, d in distances.items()]
    matches.sort(key=lambda x: x[1])

    if not matches:
        return None, float("inf"), 0.0, {}

    best_name, best_dist, best_conf = matches[0]
    second_dist = matches[1][1] if len(matches) > 1 else 1.0

    return best_name, best_dist, second_dist, distances


def is_clear_winner(matches: List[Tuple], voiceprints: Dict, cfg: Dict = None) -> bool:
    cfg = cfg or _DEFAULT_CFG
    gap_threshold = cfg.get("matching", {}).get("clear_winner_gap", 0.02)
    embed_only_thresh = cfg.get("matching", {}).get("embed_only_threshold", 0.16)

    if len(matches) < 2:
        return True

    best = matches[0][1]
    second = matches[1][1]
    best_val = best["combined"] if isinstance(best, dict) else best
    second_val = second["combined"] if isinstance(second, dict) else second
    gap = second_val - best_val

    if gap >= gap_threshold:
        return True

    # Tie-breaking: prefer the candidate with significantly more training data
    best_name = matches[0][0]
    second_name = matches[1][0]
    first_dur = (
        voiceprints.get(best_name, {}).get("segments_sec") or
        voiceprints.get(best_name, {}).get("total_speech_sec", 0)
    )
    second_dur = (
        voiceprints.get(second_name, {}).get("segments_sec") or
        voiceprints.get(second_name, {}).get("total_speech_sec", 0)
    )
    if first_dur > second_dur * 2 and best_val < embed_only_thresh:
        return True

    return False


def match_clusters(
    clusters: Dict[str, Dict],
    voiceprints: Dict[str, Dict],
    cfg: Dict = None,
    all_cluster_features: Dict[str, Dict] = None,
) -> Dict[str, Dict]:
    cfg = cfg or _DEFAULT_CFG
    matching_cfg = cfg.get("matching", _DEFAULT_CFG["matching"])
    accept_thresh = matching_cfg.get("accept_threshold", 0.35)
    embed_only_thresh = matching_cfg.get("embed_only_threshold", 0.16)
    embed_only_accept = matching_cfg.get("embed_only_accept_threshold", 0.22)

    all_cluster_features = all_cluster_features or {}
    results = {}

    for cluster_id, cluster_data in clusters.items():
        emb = cluster_data.get("embedding", [])
        pitch = cluster_data.get("pitch_hz", 0.0) or 0.0
        energy = cluster_data.get("energy_rms", 0.0) or 0.0
        features = all_cluster_features.get(cluster_id, {})

        if not emb:
            continue

        best_name, best_dist, second_dist, all_distances = find_best_match(
            emb, pitch, energy, voiceprints, cfg, features
        )

        matches = [(name, d["combined"], d["confidence"]) for name, d in all_distances.items()]
        matches.sort(key=lambda x: x[1])

        clear_winner = is_clear_winner(matches, voiceprints, cfg)

        is_strong_embed = all_distances.get(best_name, {}).get("emb_dist", 1.0) < embed_only_thresh
        matched = (
            best_name is not None and
            (is_strong_embed or best_dist <= accept_thresh) and
            clear_winner
        )

        results[cluster_id] = {
            "matched": matched,
            "label": best_name if matched else cluster_id,
            "name": best_name if matched else None,
            "distance": best_dist,
            "best_match": best_name,
            "confidence": all_distances.get(best_name, {}).get("confidence", 0.0) if matched else 0.0,
            "distances": all_distances,
            "all_distances": {k: v["combined"] for k, v in all_distances.items()},
            "clear_winner": clear_winner,
        }

    return results


def merge_matched_clusters(results: Dict[str, Dict], clusters: Dict[str, Dict]) -> Dict[str, str]:
    label_map = {}
    for cluster_id, result in results.items():
        if result.get("matched") and result.get("label"):
            label_map[cluster_id] = result["label"]
        else:
            label_map[cluster_id] = cluster_id
    return label_map
