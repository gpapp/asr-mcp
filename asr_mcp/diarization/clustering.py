import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger("asr_mcp.diarization.clustering")


def cap_clusters(
    raw_embeddings: np.ndarray,
    long_labels: np.ndarray,
    max_clusters: int = 15,
) -> np.ndarray:
    unique_labels = np.unique(long_labels)
    if len(unique_labels) <= max_clusters:
        return long_labels

    logger.info("Capping clusters: %d -> %d", len(unique_labels), max_clusters)
    from sklearn.cluster import KMeans
    n_clusters = min(max_clusters, len(raw_embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    new_labels = kmeans.fit_predict(raw_embeddings)
    return new_labels


def greedy_merge_clusters(
    raw_embeddings: np.ndarray,
    long_labels: np.ndarray,
    merge_threshold: float = 0.25,
) -> Tuple[np.ndarray, dict[int, np.ndarray]]:
    unique_labels = np.unique(long_labels)
    centroids = {}
    for label in unique_labels:
        mask = long_labels == label
        centroids[int(label)] = raw_embeddings[mask].mean(axis=0)

    changed = True
    while changed:
        changed = False
        labels_list = sorted(centroids.keys())
        if len(labels_list) < 2:
            break

        best_i, best_j, best_dist = -1, -1, float("inf")
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                li, lj = labels_list[i], labels_list[j]
                dist = 1.0 - float(np.dot(centroids[li], centroids[lj]) /
                                   (np.linalg.norm(centroids[li]) * np.linalg.norm(centroids[lj]) + 1e-8))
                if dist < best_dist:
                    best_dist = dist
                    best_i, best_j = li, lj

        if best_dist < merge_threshold:
            long_labels[long_labels == best_j] = best_i
            mask = long_labels == best_i
            centroids[best_i] = raw_embeddings[mask].mean(axis=0)
            del centroids[best_j]
            changed = True
            logger.debug("Merged cluster %d -> %d (dist=%.3f)", best_j, best_i, best_dist)

    return long_labels, centroids


def match_known_speakers_simple(
    cluster_centroids: dict[int, np.ndarray],
    known_speakers: dict[str, dict],
    match_thresh: float = 0.03,
    close_match_thresh: float = 0.1,
) -> Tuple[dict[int, str], dict]:
    label_map = {}
    match_info = {}

    for cluster_id, centroid in cluster_centroids.items():
        best_name = None
        best_dist = float("inf")

        for name, vp_data in known_speakers.items():
            vp_emb = np.array(vp_data.get("embedding", []), dtype=np.float32)
            if len(vp_emb) == 0:
                continue
            dist = 1.0 - float(np.dot(centroid, vp_emb) /
                               (np.linalg.norm(centroid) * np.linalg.norm(vp_emb) + 1e-8))
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_name and best_dist < match_thresh:
            label_map[cluster_id] = best_name
            match_info[cluster_id] = {"name": best_name, "distance": best_dist}
        elif best_name and best_dist < close_match_thresh:
            label_map[cluster_id] = best_name
            match_info[cluster_id] = {"name": best_name, "distance": best_dist, "close_match": True}
        else:
            label_map[cluster_id] = f"SPEAKER_{cluster_id:02d}"

    return label_map, match_info


def match_known_speakers_full(
    merged_segments: list,
    all_segments_meta: list,
    embeddable_indices: list,
    raw_embeddings: np.ndarray,
    cluster_centroids: dict,
    profiles: dict,
    known_speakers: dict,
    cfg: dict = None,
) -> Tuple[list[dict], dict]:
    from asr_mcp.speaker.matcher import match_clusters, merge_matched_clusters

    clusters_data = {}
    for cluster_id, centroid in cluster_centroids.items():
        clusters_data[str(cluster_id)] = {
            "embedding": centroid.tolist(),
            "pitch_hz": profiles.get(f"SPEAKER_{cluster_id:02d}", {}).get("pitch_hz", 0.0),
            "energy_rms": profiles.get(f"SPEAKER_{cluster_id:02d}", {}).get("energy_rms", 0.0),
        }

    match_results = match_clusters(clusters_data, known_speakers, cfg)
    label_map = merge_matched_clusters(match_results, clusters_data)

    for seg in merged_segments:
        old_speaker = seg.get("speaker", "")
        if old_speaker in label_map:
            seg["speaker"] = label_map[old_speaker]

    return merged_segments, match_results
