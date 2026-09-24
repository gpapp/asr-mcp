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


def merge_similar_speakers(
    segments: list,
    raw_embeddings: np.ndarray,
    long_labels: np.ndarray,
    cluster_centroids: dict,
    profiles: dict,
    cfg: dict = None,
) -> Tuple[list, dict, dict]:
    from asr_mcp.speaker.matcher import compute_distance

    cfg = cfg or {}
    matching_cfg = cfg.get("matching", {})
    merge_thresh = matching_cfg.get("embed_only_threshold", 0.2)

    unique_labels = sorted(cluster_centroids.keys())
    if len(unique_labels) < 2:
        return segments, cluster_centroids, profiles

    voiceprints = {}
    for label in unique_labels:
        speaker_name = f"Speaker {label + 1}"
        profile = profiles.get(speaker_name, {})
        voiceprints[str(label)] = {
            "embedding": cluster_centroids[label].tolist(),
            "pitch_hz": profile.get("pitch_hz", 0.0),
            "energy_rms": profile.get("energy_rms", 0.0),
            "spectral_centroid": profile.get("spectral_centroid", 0.0),
            "spectral_rolloff": profile.get("spectral_rolloff", 0.0),
            "mfcc": profile.get("mfcc", {}),
            "total_speech_sec": profile.get("total_speech_sec", 0.0),
        }

    merge_map = {}
    for i, li in enumerate(unique_labels):
        for j, lj in enumerate(unique_labels):
            if i >= j or lj in merge_map:
                continue
            emb_i = cluster_centroids[li]
            emb_j = cluster_centroids[lj]
            emb_dist = 1.0 - float(np.dot(emb_i, emb_j) /
                                   (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8))

            if emb_dist < merge_thresh:
                dur_i = voiceprints[str(li)].get("total_speech_sec", 0)
                dur_j = voiceprints[str(lj)].get("total_speech_sec", 0)
                target, source = (li, lj) if dur_i >= dur_j else (lj, li)
                merge_map[source] = target
                logger.info("Merge similar: Speaker %d -> Speaker %d (emb_dist=%.3f)",
                            source + 1, target + 1, emb_dist)

    if not merge_map:
        return segments, cluster_centroids, profiles

    for label in merge_map:
        target = merge_map[label]
        depth = 0
        while target in merge_map and depth < 10:
            target = merge_map[target]
            depth += 1
        merge_map[label] = target

    label_remap = {}
    for label in unique_labels:
        if label in merge_map:
            label_remap[label] = merge_map[label]
        else:
            label_remap[label] = label

    for seg in segments:
        old_speaker = seg.get("speaker", "")
        if old_speaker.startswith("Speaker "):
            try:
                old_num = int(old_speaker.split()[-1]) - 1
                if old_num in label_remap:
                    seg["speaker"] = f"Speaker {label_remap[old_num] + 1}"
            except (ValueError, IndexError):
                pass

    new_centroids = {}
    for label in unique_labels:
        target = label_remap[label]
        if target not in new_centroids:
            new_centroids[target] = []
        new_centroids[target].append(label)

    rebuilt_centroids = {}
    for target, sources in new_centroids.items():
        mask = np.isin(long_labels, sources)
        if np.any(mask):
            rebuilt_centroids[target] = raw_embeddings[mask].mean(axis=0)

    new_profiles = {}
    for target, sources in new_centroids.items():
        speaker_name = f"Speaker {target + 1}"
        merged_profile = {}
        total_dur = 0.0
        for src in sources:
            src_name = f"Speaker {src + 1}"
            if src_name in profiles:
                p = profiles[src_name]
                d = p.get("total_speech_sec", 0)
                total_dur += d
                for field in ["pitch_hz", "pitch_std", "energy_rms", "spectral_centroid", "spectral_rolloff"]:
                    merged_profile[field] = merged_profile.get(field, 0) + p.get(field, 0) * d
                if "mfcc" in p and not merged_profile.get("mfcc"):
                    merged_profile["mfcc"] = p["mfcc"]
        if total_dur > 0:
            for field in ["pitch_hz", "pitch_std", "energy_rms", "spectral_centroid", "spectral_rolloff"]:
                if field in merged_profile:
                    merged_profile[field] /= total_dur
            merged_profile["total_speech_sec"] = total_dur
        new_profiles[speaker_name] = merged_profile

    num_merged = len(unique_labels) - len(rebuilt_centroids)
    logger.info("Merged %d similar speakers: %d -> %d",
                num_merged, len(unique_labels), len(rebuilt_centroids))

    return segments, rebuilt_centroids, new_profiles


def match_known_speakers_full(
    merged_segments: list,
    all_segments_meta: list,
    embeddable_indices: list,
    raw_embeddings: np.ndarray,
    cluster_centroids: dict,
    profiles: dict,
    known_speakers: dict,
    cfg: dict = None,
    renumber: bool = False,
) -> Tuple[list[dict], dict]:
    """Match cluster centroids against known speaker voiceprints.

    Improvements over the original:
    - Extracts full acoustic features (spectral, MFCCs) from profiles and passes
      them to match_clusters so the complete 20% spectral+MFCC weight is used.
    - Stores top-3 alternatives in each segment for ghost-elimination fallback.
    - Updates profiles keys from "Speaker N" to the matched voiceprint name.
    - Merges duplicate clusters that matched to the same known speaker.
    - Merges clusters with near-identical distance profiles (max_diff < 0.05).
    """
    from asr_mcp.speaker.matcher import match_clusters, merge_matched_clusters

    if not known_speakers:
        return merged_segments, profiles

    cfg = cfg or {}
    matching_cfg = cfg.get("matching", {})
    accept_thresh = matching_cfg.get("accept_threshold", 0.35)
    embed_only_thresh = matching_cfg.get("embed_only_threshold", 0.16)
    conf_thresh = matching_cfg.get("confidence_threshold", 0.3)
    conf_max_dist = cfg.get("normalization", {}).get("confidence_max_distance", 0.5)
    gap_threshold = matching_cfg.get("clear_winner_gap", 0.02)

    # Build clusters_data with full acoustic features from profiles
    clusters_data = {}
    all_cluster_features = {}
    for cluster_id, centroid in cluster_centroids.items():
        label = f"Speaker {cluster_id + 1}"
        profile = profiles.get(label, {})
        clusters_data[label] = {
            "embedding": centroid.tolist() if isinstance(centroid, np.ndarray) else centroid,
            "pitch_hz": profile.get("pitch_hz", 0.0),
            "energy_rms": profile.get("energy_rms", 0.0),
        }
        # Extract spectral + MFCC features for the full distance model
        features: dict = {}
        for feat in ("spectral_centroid", "spectral_rolloff"):
            if feat in profile:
                features[feat] = profile[feat]
        mfcc = profile.get("mfcc")
        if isinstance(mfcc, dict):
            for name, val in mfcc.items():
                if isinstance(val, (int, float)):
                    features[name] = float(val)
        elif mfcc is not None:
            for j, val in enumerate(mfcc):
                features[f"mfcc{j}_mean"] = float(val)
            # std not stored in profile but contribute if available
        if features:
            all_cluster_features[label] = features

    # Run full multi-feature matching
    match_results = match_clusters(
        clusters_data, known_speakers, cfg,
        all_cluster_features=all_cluster_features if all_cluster_features else None,
    )

    # Collect all_matches for alternatives and post-match merging
    all_matches: dict[str, list] = {}
    for spk, result in match_results.items():
        distances = result.get("all_distances", {})
        matches_list = []
        for name, dist in distances.items():
            conf = max(0.0, 1.0 - dist / conf_max_dist)
            matches_list.append((name, dist, conf))
        matches_list.sort(key=lambda x: x[1])
        all_matches[spk] = matches_list

    # Apply best match: remap segments, store alternatives, update profiles
    label_map = merge_matched_clusters(match_results, clusters_data)

    # Track which original "Speaker N" mapped to which known name
    matched_known: dict[str, str] = {}  # old_label -> new_name
    for spk, result in match_results.items():
        if result.get("matched") and result.get("label") != spk:
            matched_known[spk] = result["label"]

    for seg in merged_segments:
        old_speaker = seg.get("speaker", "")
        if old_speaker in label_map:
            seg["speaker"] = label_map[old_speaker]
        # Store top-3 alternatives for ghost elimination fallback
        matches = all_matches.get(old_speaker, [])
        alternatives = []
        best_dist = match_results.get(old_speaker, {}).get("distance", 1.0)
        for name, dist, conf in matches[:4]:
            # skip the winner itself if already applied
            if name == seg.get("speaker"):
                continue
            if conf >= conf_thresh:
                alternatives.append({"speaker": name, "confidence": round(conf, 2)})
            if len(alternatives) >= 3:
                break
        if alternatives:
            seg["alternatives"] = alternatives

    # Update profiles: remap keys from "Speaker N" to matched voiceprint name
    for old_label, new_name in matched_known.items():
        if old_label in profiles and new_name not in profiles:
            profiles[new_name] = profiles.pop(old_label)
            profiles[new_name]["matched_from"] = old_label

    # Post-match merging: combine segments of clusters that both matched same speaker
    speaker_to_clusters: dict[str, list] = {}
    for spk, result in match_results.items():
        if result.get("matched"):
            best_name = result.get("label", spk)
            speaker_to_clusters.setdefault(best_name, []).append(spk)

    for speaker, cluster_list in speaker_to_clusters.items():
        if len(cluster_list) > 1:
            for extra in cluster_list[1:]:
                for seg in merged_segments:
                    if seg["speaker"] == extra:
                        seg["speaker"] = speaker

    # Additional merge: clusters with near-identical distance profiles
    cluster_ids = list(all_matches.keys())
    merged_already: set = set()
    for i in range(len(cluster_ids)):
        if cluster_ids[i] in merged_already:
            continue
        for j in range(i + 1, len(cluster_ids)):
            if cluster_ids[j] in merged_already:
                continue
            matches_i = all_matches[cluster_ids[i]]
            matches_j = all_matches[cluster_ids[j]]
            dist_vec_i = {m[0]: m[1] for m in matches_i}
            dist_vec_j = {m[0]: m[1] for m in matches_j}
            common = set(dist_vec_i.keys()) & set(dist_vec_j.keys())
            if len(common) >= 3:
                max_diff = max(abs(dist_vec_i[s] - dist_vec_j[s]) for s in common)
                if max_diff < 0.05:
                    target = cluster_ids[i]
                    for seg in merged_segments:
                        if seg.get("speaker") == cluster_ids[j]:
                            seg["speaker"] = target
                    merged_already.add(cluster_ids[j])

    return merged_segments, profiles
