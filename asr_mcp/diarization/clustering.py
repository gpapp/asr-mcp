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
    n_clusters = len(set(int(l) for l in long_labels))
    if n_clusters > max_clusters:
        logger.info("Capping clusters: %d -> %d", n_clusters, max_clusters)
        clusterer = AgglomerativeClustering(
            n_clusters=max_clusters,
            metric="cosine",
            linkage="average",
        )
        if len(raw_embeddings) > 1:
            long_labels = clusterer.fit_predict(raw_embeddings)
    return long_labels


def greedy_merge_clusters(
    raw_embeddings: np.ndarray,
    long_labels: np.ndarray,
    merge_threshold: float = 0.25,
) -> Tuple[np.ndarray, dict[int, np.ndarray]]:
    cluster_ids = sorted(set(int(l) for l in long_labels))
    cluster_avgs = {}
    for cid in cluster_ids:
        mask = long_labels == cid
        cluster_avgs[cid] = np.mean(raw_embeddings[mask], axis=0)

    changed = True
    while changed:
        changed = False
        ids = sorted(cluster_avgs.keys())
        for i_idx in range(len(ids)):
            for j_idx in range(i_idx + 1, len(ids)):
                id_i, id_j = ids[i_idx], ids[j_idx]
                if id_i not in cluster_avgs or id_j not in cluster_avgs:
                    continue
                vi = cluster_avgs[id_i]
                vj = cluster_avgs[id_j]
                norm_i = np.linalg.norm(vi)
                norm_j = np.linalg.norm(vj)
                if norm_i < 1e-8 or norm_j < 1e-8:
                    continue
                dist = 1.0 - float(np.dot(vi, vj) / (norm_i * norm_j))
                if dist < merge_threshold:
                    long_labels[long_labels == id_j] = id_i
                    mask_i = long_labels == id_i
                    cluster_avgs[id_i] = np.mean(raw_embeddings[mask_i], axis=0)
                    del cluster_avgs[id_j]
                    changed = True
                    break
            if changed:
                break

    cluster_centroids = {}
    for cluster_id in set(long_labels):
        mask = (long_labels == cluster_id)
        mean_emb = raw_embeddings[mask].mean(axis=0)
        norm_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
        cluster_centroids[int(cluster_id)] = norm_emb

    return long_labels, cluster_centroids


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
    merged_segments: list[dict],
    all_segments_meta: list[dict],
    embeddable_indices: list[int],
    raw_embeddings: np.ndarray,
    cluster_centroids: dict,
    profiles: dict,
    known_speakers: dict[str, dict],
    cfg: dict = None,
    renumber: bool = False,
) -> Tuple[list[dict], dict]:
    """Match cluster centroids against known speaker voiceprints.

    Matches cohere-diarization's match_known_speakers_full algorithm:
    1. Computes centroid embedding for each final speaker from their segment embeddings.
    2. Collects full acoustic features (pitch, energy, spectral, MFCCs) from profiles.
    3. Runs multi-feature distance matching against known voiceprints.
    4. Evaluates clear winner / data tie-breaking.
    5. Replaces matched speaker labels in segments, populates alternatives, updates profiles.
    6. Merges multiple clusters that matched to the same known speaker (updating both segments and profiles).
    7. Merges clusters with near-identical distance profiles (<0.05 max diff).
    """
    from asr_mcp.speaker.matcher import match_clusters
    from asr_mcp.diarization.segment_ops import merge_profiles

    if not known_speakers:
        return merged_segments, profiles

    cfg = cfg or {}
    matching_cfg = cfg.get("matching", {})
    match_thresh = matching_cfg.get("accept_threshold", 0.35)
    gap_threshold = matching_cfg.get("clear_winner_gap", 0.02)
    embed_only_thresh = matching_cfg.get("embed_only_threshold", 0.16)
    conf_thresh = matching_cfg.get("confidence_threshold", 0.3)
    conf_max_dist = cfg.get("normalization", {}).get("confidence_max_distance", 0.5)

    # 1. Compute centroid for each final speaker from segments & meta
    speaker_centroids: dict[str, list] = {}
    for seg in merged_segments:
        spk = seg["speaker"]
        if spk not in speaker_centroids:
            speaker_centroids[spk] = []

        seg_start = seg["start"]
        seg_end = seg["end"]
        for i, meta in enumerate(all_segments_meta):
            if meta["start"] >= seg_start - 0.1 and meta["end"] <= seg_end + 0.1:
                if "speaker_raw" in meta:
                    raw_id = meta["speaker_raw"]
                    if raw_id in cluster_centroids:
                        speaker_centroids[spk].append(np.array(cluster_centroids[raw_id]))

    # 2. Build clusters dict with full acoustic features
    clusters = {}
    all_cluster_features = {}
    for spk, emb_list in speaker_centroids.items():
        if not emb_list:
            continue
        centroid = np.mean(emb_list, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        prof = profiles.get(spk, {})
        clusters[spk] = {
            "embedding": centroid.tolist(),
            "pitch_hz": prof.get("pitch_hz", 0.0) or 0.0,
            "energy_rms": prof.get("energy_rms", 0.0) or 0.0,
        }

        features = {
            "spectral_centroid": prof.get("spectral_centroid", 0.0) or 0.0,
            "spectral_rolloff": prof.get("spectral_rolloff", 0.0) or 0.0,
        }
        mfcc = prof.get("mfcc")
        if isinstance(mfcc, dict):
            for name, val in mfcc.items():
                if isinstance(val, (int, float)):
                    features[name] = float(val)
        for i in range(13):
            if f"mfcc{i}_mean" not in features:
                features[f"mfcc{i}_mean"] = prof.get(f"mfcc{i}_mean", 0.0) or 0.0
            if f"mfcc{i}_std" not in features:
                features[f"mfcc{i}_std"] = prof.get(f"mfcc{i}_std", 0.0) or 0.0
        all_cluster_features[spk] = features

    # 3. Match clusters
    match_results = match_clusters(
        clusters, known_speakers, cfg,
        all_cluster_features=all_cluster_features,
    )

    # 4. Build all_matches for post-processing
    all_matches: dict[str, list] = {}
    for spk, result in match_results.items():
        matches_list = []
        for name, dist_info in result.get("distances", {}).items():
            conf = dist_info.get("confidence", 0.0)
            combined_dist = dist_info.get("combined", dist_info.get("total", 1.0))
            matches_list.append((name, combined_dist, conf))
        matches_list.sort(key=lambda x: x[1])
        all_matches[spk] = matches_list

    # 5. Apply best match if distance below threshold, store alternatives
    for spk, emb_list in speaker_centroids.items():
        if not emb_list or spk not in all_matches:
            continue
        matches = all_matches[spk]
        if not matches:
            continue
        best_match, best_dist, best_conf = matches[0]

        clear_winner = True
        if len(matches) > 1:
            second_dist = matches[1][1]
            gap = second_dist - best_dist
            if gap < gap_threshold:
                first_dur = (
                    known_speakers.get(matches[0][0], {}).get("segments_sec") or
                    known_speakers.get(matches[0][0], {}).get("total_speech_sec", 0)
                )
                second_dur = (
                    known_speakers.get(matches[1][0], {}).get("segments_sec") or
                    known_speakers.get(matches[1][0], {}).get("total_speech_sec", 0)
                )
                if first_dur > second_dur * 2 and best_dist < embed_only_thresh:
                    clear_winner = True
                else:
                    clear_winner = False

        alternatives = []
        for name, dist, conf in matches[1:4]:
            if conf >= conf_thresh:
                alternatives.append({"speaker": name, "confidence": round(conf, 2)})

        if best_match and best_dist <= match_thresh and clear_winner:
            for seg in merged_segments:
                if seg["speaker"] == spk:
                    seg["speaker"] = best_match
                    if alternatives:
                        seg["alternatives"] = alternatives
            if spk in profiles:
                profiles[best_match] = profiles.pop(spk)
                profiles[best_match]["matched_from"] = spk
                profiles[best_match]["match_confidence"] = best_conf

    # 6. Post-match merging: combine clusters that both matched to the same speaker
    speaker_to_clusters: dict[str, list] = {}
    for spk, matches_list in all_matches.items():
        if not matches_list:
            continue
        best_match, best_dist, _ = matches_list[0]
        if best_match and best_dist <= match_thresh:
            speaker_to_clusters.setdefault(best_match, []).append(spk)

    for speaker, cluster_list in speaker_to_clusters.items():
        if len(cluster_list) > 1:
            primary = cluster_list[0]
            for extra in cluster_list[1:]:
                for seg in merged_segments:
                    if seg["speaker"] == extra or seg["speaker"] == primary:
                        seg["speaker"] = speaker
                merge_profiles(profiles, speaker, extra)

    # 7. Additional merge: clusters with near-identical distance profiles
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
            if not matches_i or not matches_j:
                continue
            dist_vec_i = {m[0]: m[1] for m in matches_i}
            dist_vec_j = {m[0]: m[1] for m in matches_j}
            common = set(dist_vec_i.keys()) & set(dist_vec_j.keys())
            if len(common) >= 3:
                max_diff = max(abs(dist_vec_i[s] - dist_vec_j[s]) for s in common)
                if max_diff < 0.05:
                    target_spk = cluster_ids[i]
                    source_spk = cluster_ids[j]
                    for seg in merged_segments:
                        if seg["speaker"] == source_spk:
                            seg["speaker"] = target_spk
                    merge_profiles(profiles, target_spk, source_spk)
                    merged_already.add(source_spk)

    return merged_segments, profiles


def collapse_unknown_speakers_second_pass(
    segments: list[dict],
    audio_np: np.ndarray,
    sample_rate: int,
    known_speakers: dict[str, dict],
    profiles: dict,
    state=None,
    cfg: dict = None,
) -> Tuple[list[dict], dict]:
    """Second-pass re-identification and consolidation of unknown speakers.

    1. Gathers all audio across the entire call for each speaker.
    2. Computes aggregated high-SNR embeddings.
    3. Blends in-call references with DB voiceprints for confirmed known speakers.
    4. Matches remaining unknown 'Speaker N' clusters against known targets with larger fit margin.
    5. Cross-matches and merges duplicate unknown speakers.
    6. Fuses adjacent same-speaker segments.
    """
    from asr_mcp.speaker.embedding import extract_embedding
    from asr_mcp.speaker.matcher import find_best_match, is_clear_winner
    from asr_mcp.diarization.segment_ops import merge_profiles, collapse_same_speaker_segments

    if not segments or audio_np is None or len(audio_np) == 0:
        return segments, profiles

    cfg = cfg or {}
    sec_cfg = cfg.get("second_pass", {})
    if not sec_cfg.get("enabled", True):
        return segments, profiles

    accept_thresh = sec_cfg.get("accept_threshold", 0.38)
    unknown_merge_thresh = sec_cfg.get("unknown_merge_threshold", 0.25)
    min_speaker_dur = sec_cfg.get("min_speaker_duration_sec", 1.0)

    all_speakers = set(seg.get("speaker") for seg in segments if seg.get("speaker"))
    unknown_speakers = [
        s for s in all_speakers
        if (s.startswith("Speaker ") or s.startswith("SPEAKER ")) and s != "OVERLAP"
    ]
    if not unknown_speakers:
        return segments, profiles

    # 1. Gather audio and compute aggregated embeddings for each speaker
    spk_embeddings: dict[str, np.ndarray] = {}
    for spk in all_speakers:
        if spk == "OVERLAP":
            continue
        audio_chunks = []
        for seg in segments:
            if seg.get("speaker") != spk:
                continue
            s_sec = seg.get("start", 0.0)
            e_sec = seg.get("end", 0.0)
            if e_sec - s_sec < 0.2:
                continue
            s_idx = max(0, int(s_sec * sample_rate))
            e_idx = min(len(audio_np), int(e_sec * sample_rate))
            if e_idx > s_idx:
                audio_chunks.append(audio_np[s_idx:e_idx])

        if not audio_chunks:
            continue

        spk_audio = np.concatenate(audio_chunks)
        dur = len(spk_audio) / sample_rate
        if dur >= min_speaker_dur:
            try:
                emb = extract_embedding(spk_audio, sample_rate, state=state)
                if emb is not None and len(emb) > 0:
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        spk_embeddings[spk] = emb / norm
            except Exception as e:
                logger.warning("Second pass embedding failed for %s: %s", spk, e)

    # 2. Build reference voiceprints (blending DB voiceprints + in-call centroids)
    reference_targets = {}
    if known_speakers:
        for name, vp in known_speakers.items():
            ref_vp = dict(vp)
            db_emb = np.array(vp.get("embedding", []), dtype=np.float32)
            if name in spk_embeddings and len(db_emb) > 0:
                in_call_emb = spk_embeddings[name]
                blended = 0.5 * db_emb + 0.5 * in_call_emb
                norm = np.linalg.norm(blended)
                if norm > 0:
                    blended = blended / norm
                ref_vp["embedding"] = blended.tolist()
            reference_targets[name] = ref_vp

    # 3. Match unknown clusters against reference targets
    resolved_unknowns: set[str] = set()
    if reference_targets:
        for spk in list(unknown_speakers):
            if spk not in spk_embeddings:
                continue
            emb = spk_embeddings[spk]
            prof = profiles.get(spk, {})
            pitch = prof.get("pitch_hz", 0.0) or 0.0
            energy = prof.get("energy_rms", 0.0) or 0.0
            features = {
                k: v for k, v in prof.items()
                if k.startswith("mfcc") or k.startswith("spectral")
            }

            best_name, best_dist, second_dist, all_dists = find_best_match(
                emb.tolist(), pitch, energy, reference_targets, cfg, features
            )

            if not best_name or best_dist > accept_thresh:
                continue

            matches = [(n, d["combined"], d["confidence"]) for n, d in all_dists.items()]
            matches.sort(key=lambda x: x[1])
            if not is_clear_winner(matches, reference_targets, cfg):
                continue

            # Remap segments
            for seg in segments:
                if seg.get("speaker") == spk:
                    seg["speaker"] = best_name
            merge_profiles(profiles, best_name, spk)
            resolved_unknowns.add(spk)
            logger.info(
                "Second pass: collapsed unknown %s -> %s (dist=%.3f, conf=%.2f)",
                spk, best_name, best_dist, all_dists[best_name].get("confidence", 0.0)
            )

    # 4. Cross-match and merge duplicate unknown speakers
    remaining_unknowns = [s for s in unknown_speakers if s not in resolved_unknowns]
    merged_unknowns: set[str] = set()
    for i in range(len(remaining_unknowns)):
        u1 = remaining_unknowns[i]
        if u1 in merged_unknowns or u1 not in spk_embeddings:
            continue
        for j in range(i + 1, len(remaining_unknowns)):
            u2 = remaining_unknowns[j]
            if u2 in merged_unknowns or u2 not in spk_embeddings:
                continue

            cos_dist = 1.0 - float(np.dot(spk_embeddings[u1], spk_embeddings[u2]))
            if cos_dist < unknown_merge_thresh:
                # Keep the one with longer total speech duration as primary
                dur1 = profiles.get(u1, {}).get("total_speech_sec", 0.0)
                dur2 = profiles.get(u2, {}).get("total_speech_sec", 0.0)
                primary, secondary = (u1, u2) if dur1 >= dur2 else (u2, u1)

                for seg in segments:
                    if seg.get("speaker") == secondary:
                        seg["speaker"] = primary
                merge_profiles(profiles, primary, secondary)
                merged_unknowns.add(secondary)
                logger.info(
                    "Second pass: merged duplicate unknowns %s -> %s (dist=%.3f)",
                    secondary, primary, cos_dist
                )

    segments = collapse_same_speaker_segments(segments, max_gap=0.5)
    return segments, profiles
