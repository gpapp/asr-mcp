import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("asr_mcp.speaker.matcher")

_DEFAULT_CFG = {
    "weights": {
        "embedding": 0.6,
        "pitch": 0.15,
        "energy": 0.0,
        "spectral": 0.1,
        "mfcc": 0.1,
    },
    "matching": {
        "accept_threshold": 0.35,
        "clear_winner_gap": 0.02,
        "embed_only_threshold": 0.16,
        "embed_only_accept_threshold": 0.22,
    },
    "normalization": {
        "pitch_hz_per_unit": 50,
        "energy_rms_per_unit": 0.05,
        "spectral_centroid_per_unit": 500,
        "spectral_rolloff_per_unit": 1000,
    },
}


def _compute_spectral_distance(cluster: dict, voiceprint: dict, norm_cfg: dict) -> float:
    c_centroid = cluster.get("spectral_centroid", 0.0)
    v_centroid = voiceprint.get("spectral_centroid", 0.0)
    c_rolloff = cluster.get("spectral_rolloff", 0.0)
    v_rolloff = voiceprint.get("spectral_rolloff", 0.0)

    centroid_per = norm_cfg.get("spectral_centroid_per_unit", 500)
    rolloff_per = norm_cfg.get("spectral_rolloff_per_unit", 1000)

    d_centroid = abs(c_centroid - v_centroid) / centroid_per if centroid_per else 0.0
    d_rolloff = abs(c_rolloff - v_rolloff) / rolloff_per if rolloff_per else 0.0

    return (d_centroid + d_rolloff) / 2.0


def _compute_mfcc_distance(cluster: dict, voiceprint: dict, norm_cfg: dict) -> float:
    c_mfcc = cluster.get("mfcc", {})
    v_mfcc = voiceprint.get("mfcc", {})
    if not c_mfcc or not v_mfcc:
        return 0.5

    total_dist = 0.0
    count = 0
    for i in range(13):
        c_mean = c_mfcc.get(f"mfcc{i}_mean", 0.0)
        v_mean = v_mfcc.get(f"mfcc{i}_mean", 0.0)
        c_std = c_mfcc.get(f"mfcc{i}_std", 1.0)
        v_std = v_mfcc.get(f"mfcc{i}_std", 1.0)

        mean_per = norm_cfg.get(f"mfcc{i}_mean_per_unit", 5)
        std_per = norm_cfg.get(f"mfcc{i}_std_per_unit", 3)

        d_mean = abs(c_mean - v_mean) / mean_per if mean_per else 0.0
        d_std = abs(c_std - v_std) / std_per if std_per else 0.0
        total_dist += (d_mean + d_std) / 2.0
        count += 1

    return total_dist / max(count, 1)


def compute_distance(
    cluster_emb: List[float],
    cluster_pitch: float,
    cluster_energy: float,
    voiceprint: Dict,
    cfg: Dict = None,
    cluster_features: Dict = None,
) -> Dict[str, float]:
    cfg = cfg or _DEFAULT_CFG
    weights = cfg.get("weights", _DEFAULT_CFG["weights"])
    norm_cfg = cfg.get("normalization", _DEFAULT_CFG["normalization"])

    vp_emb = np.array(voiceprint.get("embedding", []), dtype=np.float32)
    if len(vp_emb) == 0 or len(cluster_emb) == 0:
        return {"total": 1.0, "embedding": 1.0, "pitch": 0.5, "energy": 0.5, "spectral": 0.5, "mfcc": 0.5}

    emb_dist = 1.0 - float(np.dot(cluster_emb, vp_emb) /
                           (np.linalg.norm(cluster_emb) * np.linalg.norm(vp_emb) + 1e-8))

    pitch_per = norm_cfg.get("pitch_hz_per_unit", 50)
    energy_per = norm_cfg.get("energy_rms_per_unit", 0.05)
    vp_pitch = voiceprint.get("pitch_hz", 0.0)
    vp_energy = voiceprint.get("energy_rms", 0.0)

    pitch_dist = abs(cluster_pitch - vp_pitch) / pitch_per if pitch_per else 0.0
    energy_dist = abs(cluster_energy - vp_energy) / energy_per if energy_per else 0.0

    if cluster_features:
        spectral_dist = _compute_spectral_distance(cluster_features, voiceprint, norm_cfg)
        mfcc_dist = _compute_mfcc_distance(cluster_features, voiceprint, norm_cfg)
    else:
        spectral_dist = 0.0
        mfcc_dist = 0.0

    total = (
        weights.get("embedding", 0.6) * emb_dist
        + weights.get("pitch", 0.15) * pitch_dist
        + weights.get("energy", 0.0) * energy_dist
        + weights.get("spectral", 0.1) * spectral_dist
        + weights.get("mfcc", 0.1) * mfcc_dist
    )

    return {
        "total": total,
        "embedding": emb_dist,
        "pitch": pitch_dist,
        "energy": energy_dist,
        "spectral": spectral_dist,
        "mfcc": mfcc_dist,
    }


def find_best_match(
    cluster_emb: List[float],
    cluster_pitch: float,
    cluster_energy: float,
    voiceprints: Dict[str, Dict],
    cfg: Dict = None,
    cluster_features: Dict = None,
) -> Tuple[Optional[str], float, float, Dict[str, Dict]]:
    matches = {}
    for name, vp in voiceprints.items():
        dist = compute_distance(cluster_emb, cluster_pitch, cluster_energy, vp, cfg, cluster_features)
        matches[name] = dist

    if not matches:
        return None, 1.0, 1.0, matches

    sorted_matches = sorted(matches.items(), key=lambda x: x[1]["total"])
    best_name, best_dist = sorted_matches[0]
    second_dist = sorted_matches[1][1]["total"] if len(sorted_matches) > 1 else 1.0

    return best_name, best_dist["total"], second_dist, matches


def is_clear_winner(matches: List[Tuple], voiceprints: Dict, cfg: Dict = None) -> bool:
    cfg = cfg or _DEFAULT_CFG
    gap_threshold = cfg.get("matching", {}).get("clear_winner_gap", 0.02)

    if len(matches) < 2:
        return True

    best = matches[0][1]
    second = matches[1][1]
    return (second - best) > gap_threshold


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

    results = {}
    for cluster_id, cluster_data in clusters.items():
        emb = cluster_data.get("embedding", [])
        pitch = cluster_data.get("pitch_hz", 0.0)
        energy = cluster_data.get("energy_rms", 0.0)
        features = (all_cluster_features or {}).get(cluster_id)

        best_name, best_dist, second_dist, all_matches = find_best_match(
            emb, pitch, energy, voiceprints, cfg, features
        )

        matched = False
        label = cluster_id

        if best_name and best_dist < accept_thresh:
            if best_dist < embed_only_thresh:
                matched = True
                label = best_name
            elif is_clear_winner(
                sorted(all_matches.items(), key=lambda x: x[1]["total"]),
                voiceprints, cfg,
            ):
                matched = True
                label = best_name

        results[cluster_id] = {
            "matched": matched,
            "label": label,
            "distance": best_dist,
            "best_match": best_name,
            "all_distances": {k: v["total"] for k, v in all_matches.items()},
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
