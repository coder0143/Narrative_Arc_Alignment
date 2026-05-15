"""
Core pipeline for Narrative Arc Alignment.

For each video:
  1. Extract frames at target FPS
  2. Embed frames with Qwen3-VL-Embedding
  3. Compute embedding velocity (semantic change rate between frames)
  4. Build arc signature = 1D curve of semantic velocity over time

Cross-video:
  - DTW alignment of arc signatures to find structurally equivalent moments
  - Text query → cosine similarity → ranked frame retrieval across all videos
  - Scene DNA: k-means cluster all frames from all videos together
"""

import os
import json
import time
import base64
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine
from scipy.signal import savgol_filter

# Optional: fastdtw for fast DTW, fallback to scipy
try:
    from fastdtw import fastdtw
    HAS_FASTDTW = True
except ImportError:
    HAS_FASTDTW = False
    print("[pipeline] fastdtw not found, using scipy DTW (slower for long videos)")


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, target_fps: float = 1.0, max_frames: int = 120) -> Tuple[List[Image.Image], List[float], float]:
    """
    Extract frames from video at target_fps.
    Returns (frames, timestamps_in_seconds, video_duration).
    Caps at max_frames to stay within VRAM budget.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps

    # Compute frame step
    step = max(1, int(native_fps / target_fps))

    frames, timestamps = [], []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).resize((336, 336), Image.LANCZOS)
            frames.append(pil)
            timestamps.append(frame_idx / native_fps)
        frame_idx += 1

    cap.release()

    # Subsample if over max_frames
    if len(frames) > max_frames:
        idxs = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idxs]
        timestamps = [timestamps[i] for i in idxs]

    return frames, timestamps, duration


def frame_to_base64(frame: Image.Image, quality: int = 60) -> str:
    buf = BytesIO()
    frame.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ── Arc computation ───────────────────────────────────────────────────────────

def compute_velocity(embeddings: np.ndarray) -> np.ndarray:
    """
    Frame-to-frame cosine distance in embedding space.
    Returns velocity array of length len(embeddings)-1.
    Padded to same length with leading 0.
    """
    N = len(embeddings)
    velocity = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        velocity[i] = cosine(embeddings[i - 1], embeddings[i])
    return velocity


def smooth_arc(velocity: np.ndarray, window: int = 5) -> np.ndarray:
    """Savitzky-Golay smoothing of velocity curve."""
    if len(velocity) < window + 2:
        return velocity
    wl = min(window, len(velocity) if len(velocity) % 2 == 1 else len(velocity) - 1)
    wl = max(3, wl)
    if wl % 2 == 0:
        wl += 1
    try:
        return savgol_filter(velocity, window_length=wl, polyorder=2).astype(np.float32)
    except Exception:
        return velocity


def normalize_arc(arc: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx = arc.min(), arc.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arc)
    return (arc - mn) / (mx - mn)


# ── DTW alignment ─────────────────────────────────────────────────────────────

def dtw_align(arc_a: np.ndarray, arc_b: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Align two 1D arc signatures using DTW.
    Returns (distance, path) where path is list of (idx_a, idx_b) pairs.
    """
    if HAS_FASTDTW:
        dist, path = fastdtw(arc_a.reshape(-1, 1), arc_b.reshape(-1, 1))
        return float(dist), path
    else:
        # Pure numpy DTW
        N, M = len(arc_a), len(arc_b)
        dtw_mat = np.full((N + 1, M + 1), np.inf)
        dtw_mat[0, 0] = 0.0
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                cost = abs(float(arc_a[i - 1]) - float(arc_b[j - 1]))
                dtw_mat[i, j] = cost + min(dtw_mat[i - 1, j], dtw_mat[i, j - 1], dtw_mat[i - 1, j - 1])
        # Backtrack
        path = []
        i, j = N, M
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            moves = [(dtw_mat[i - 1, j], i - 1, j), (dtw_mat[i, j - 1], i, j - 1), (dtw_mat[i - 1, j - 1], i - 1, j - 1)]
            _, i, j = min(moves)
        path.reverse()
        return float(dtw_mat[N, M]), path


def find_equivalent_moment(
    query_frame_idx: int,
    arc_a: np.ndarray,
    arc_b: np.ndarray,
    path: List[Tuple[int, int]],
) -> int:
    """
    Given a frame index in video A, find structurally equivalent frame in video B via DTW path.
    """
    # Find path pairs where first element matches query_frame_idx (closest)
    best_pair = min(path, key=lambda p: abs(p[0] - query_frame_idx))
    return best_pair[1]


# ── Text query retrieval ───────────────────────────────────────────────────────

def query_frames(
    query_embedding: np.ndarray,
    all_embeddings: Dict[str, np.ndarray],
    all_timestamps: Dict[str, List[float]],
    top_k: int = 3,
) -> List[Dict]:
    """
    Given a query embedding, retrieve top_k frames from each video.
    Returns list of {video_id, frame_idx, timestamp, score}.
    """
    results = []
    for vid_id, embeddings in all_embeddings.items():
        scores = embeddings @ query_embedding  # cosine sim (both L2-normalized)
        top_idxs = np.argsort(scores)[::-1][:top_k]
        for idx in top_idxs:
            results.append({
                "video_id": vid_id,
                "frame_idx": int(idx),
                "timestamp": float(all_timestamps[vid_id][idx]),
                "score": float(scores[idx]),
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ── Scene DNA clustering ───────────────────────────────────────────────────────

def scene_dna_cluster(
    all_embeddings: Dict[str, np.ndarray],
    n_clusters: int = 6,
) -> Dict[str, Any]:
    """
    Pool all frame embeddings from all videos, k-means cluster them.
    Returns cluster assignments per video frame.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    video_ids = list(all_embeddings.keys())
    all_embs = np.vstack([all_embeddings[v] for v in video_ids])
    counts = [len(all_embeddings[v]) for v in video_ids]

    # PCA to 64 dims before clustering for speed
    n_components = min(64, all_embs.shape[0] - 1, all_embs.shape[1])
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(all_embs)

    n_clusters = min(n_clusters, len(all_embs))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(reduced)

    # Split labels back per video
    result = {"clusters": {}, "n_clusters": n_clusters}
    offset = 0
    for vid_id, count in zip(video_ids, counts):
        result["clusters"][vid_id] = labels[offset: offset + count].tolist()
        offset += count

    # 2D PCA for visualization
    pca2 = PCA(n_components=2)
    coords = pca2.fit_transform(all_embs)
    result["pca_coords"] = coords.tolist()
    result["labels"] = labels.tolist()

    # Track which video each point came from
    video_tags = []
    for vid_id, count in zip(video_ids, counts):
        video_tags.extend([vid_id] * count)
    result["video_tags"] = video_tags

    return result


# ── VideoStore: holds all processed video data in memory ──────────────────────

class VideoStore:
    """In-memory store for processed videos during a session."""

    def __init__(self):
        self.videos: Dict[str, Dict] = {}  # video_id → metadata + data
        self._tmp_dir = tempfile.mkdtemp(prefix="narcarc_")

    def save_upload(self, filename: str, data: bytes) -> str:
        """Save uploaded video bytes to temp dir, return path."""
        safe_name = Path(filename).name
        path = os.path.join(self._tmp_dir, safe_name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def process_video(
        self,
        video_id: str,
        video_path: str,
        filename: str,
        target_fps: float,
        model_name: str,
        embedder_module,
    ) -> Dict:
        """Full pipeline: extract → embed → velocity → arc."""
        print(f"[pipeline] Processing {filename} ...")
        t0 = time.time()

        frames, timestamps, duration = extract_frames(video_path, target_fps=target_fps)
        print(f"[pipeline] Extracted {len(frames)} frames in {time.time()-t0:.1f}s")

        t1 = time.time()
        embeddings = embedder_module.embed_frames(frames, model_name=model_name)
        print(f"[pipeline] Embedded {len(frames)} frames in {time.time()-t1:.1f}s")

        velocity = compute_velocity(embeddings)
        arc = normalize_arc(smooth_arc(velocity))

        # Thumbnail base64 for first frame
        thumb = frame_to_base64(frames[0], quality=50)

        # Store frame thumbnails (for retrieval display)
        frame_b64s = [frame_to_base64(f, quality=55) for f in frames]

        record = {
            "video_id": video_id,
            "filename": filename,
            "path": video_path,
            "duration": duration,
            "n_frames": len(frames),
            "target_fps": target_fps,
            "timestamps": timestamps,
            "embeddings": embeddings,      # np.ndarray (N, D)
            "velocity": velocity.tolist(),
            "arc": arc.tolist(),
            "thumbnail": thumb,
            "frame_b64s": frame_b64s,
        }
        self.videos[video_id] = record
        print(f"[pipeline] Done with {filename} in {time.time()-t0:.1f}s total")
        return self._safe_record(record)

    def _safe_record(self, record: Dict) -> Dict:
        """Return record without heavy numpy arrays for JSON serialization."""
        return {k: v for k, v in record.items() if k not in ("embeddings",)}

    def get_safe(self, video_id: str) -> Optional[Dict]:
        if video_id not in self.videos:
            return None
        return self._safe_record(self.videos[video_id])

    def list_videos(self) -> List[Dict]:
        return [
            {
                "video_id": v["video_id"],
                "filename": v["filename"],
                "duration": v["duration"],
                "n_frames": v["n_frames"],
                "thumbnail": v["thumbnail"],
            }
            for v in self.videos.values()
        ]

    def cleanup(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
