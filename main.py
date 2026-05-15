"""
Narrative Arc Alignment — FastAPI Backend
=========================================
Routes:
  POST /api/upload           — upload a video, process it
  GET  /api/videos           — list processed videos
  GET  /api/video/{id}       — metadata + arc for one video
  POST /api/query/text       — NL query → top frames across videos
  POST /api/query/frame      — click a frame → similar frames in other videos
  POST /api/align            — DTW align two videos, return alignment map
  POST /api/scene-dna        — cluster all frames across all videos
  DELETE /api/video/{id}     — remove a video from session
  GET  /                     — serve frontend
"""

import os
import uuid
import asyncio
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import embedder as emb_module
import pipeline as pipe_module
from pipeline import VideoStore, dtw_align, find_equivalent_moment, query_frames, scene_dna_cluster

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
DEFAULT_FPS = float(os.environ.get("EXTRACT_FPS", "1.0"))

app = FastAPI(title="Narrative Arc Alignment", version="1.0")
store = VideoStore()

# ── Pydantic models ───────────────────────────────────────────────────────────

class TextQueryRequest(BaseModel):
    query: str
    top_k: int = 3
    video_ids: Optional[List[str]] = None  # None = all videos

class FrameQueryRequest(BaseModel):
    source_video_id: str
    frame_idx: int
    top_k: int = 3
    target_video_ids: Optional[List[str]] = None

class AlignRequest(BaseModel):
    video_id_a: str
    video_id_b: str
    query_frame_idx: Optional[int] = None  # if set, find equivalent in B

class SceneDNARequest(BaseModel):
    video_ids: Optional[List[str]] = None
    n_clusters: int = 6


# ── Upload & process ──────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fps: float = Form(DEFAULT_FPS),
):
    if not file.filename:
        raise HTTPException(400, "No filename")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        raise HTTPException(400, f"Unsupported format: {ext}")

    data = await file.read()
    if len(data) > 500 * 1024 * 1024:  # 500MB cap
        raise HTTPException(400, "File too large (max 500MB)")

    video_id = str(uuid.uuid4())[:8]
    video_path = store.save_upload(file.filename, data)

    # Run processing synchronously (vLLM is not async-friendly)
    # In production you'd offload to a worker queue
    try:
        record = store.process_video(
            video_id=video_id,
            video_path=video_path,
            filename=file.filename,
            target_fps=fps,
            model_name=MODEL_NAME,
            embedder_module=emb_module,
        )
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")

    return JSONResponse({"status": "ok", "video_id": video_id, "record": record})


# ── Video listing ─────────────────────────────────────────────────────────────

@app.get("/api/videos")
def list_videos():
    return {"videos": store.list_videos()}


@app.get("/api/video/{video_id}")
def get_video(video_id: str):
    rec = store.get_safe(video_id)
    if rec is None:
        raise HTTPException(404, "Video not found")
    return rec


@app.delete("/api/video/{video_id}")
def delete_video(video_id: str):
    if video_id not in store.videos:
        raise HTTPException(404, "Video not found")
    del store.videos[video_id]
    return {"status": "deleted"}


# ── Text query ────────────────────────────────────────────────────────────────

@app.post("/api/query/text")
def text_query(req: TextQueryRequest):
    if not store.videos:
        raise HTTPException(400, "No videos loaded")

    query_emb = emb_module.embed_text(
        req.query,
        instruction="Retrieve video frames relevant to the user's query.",
        model_name=MODEL_NAME,
    )

    target_ids = req.video_ids or list(store.videos.keys())
    all_embeddings = {vid: store.videos[vid]["embeddings"] for vid in target_ids if vid in store.videos}
    all_timestamps = {vid: store.videos[vid]["timestamps"] for vid in target_ids if vid in store.videos}

    results = query_frames(query_emb, all_embeddings, all_timestamps, top_k=req.top_k)

    # Attach frame thumbnails
    for r in results:
        vid = store.videos[r["video_id"]]
        r["frame_b64"] = vid["frame_b64s"][r["frame_idx"]]
        r["filename"] = vid["filename"]

    return {"query": req.query, "results": results}


# ── Frame-to-frame cross-video query ──────────────────────────────────────────

@app.post("/api/query/frame")
def frame_query(req: FrameQueryRequest):
    if req.source_video_id not in store.videos:
        raise HTTPException(404, "Source video not found")

    src = store.videos[req.source_video_id]
    if req.frame_idx >= src["n_frames"]:
        raise HTTPException(400, "frame_idx out of range")

    query_emb = src["embeddings"][req.frame_idx]

    target_ids = req.target_video_ids or [
        v for v in store.videos if v != req.source_video_id
    ]
    all_embeddings = {vid: store.videos[vid]["embeddings"] for vid in target_ids if vid in store.videos}
    all_timestamps = {vid: store.videos[vid]["timestamps"] for vid in target_ids if vid in store.videos}

    results = query_frames(query_emb, all_embeddings, all_timestamps, top_k=req.top_k)

    for r in results:
        vid = store.videos[r["video_id"]]
        r["frame_b64"] = vid["frame_b64s"][r["frame_idx"]]
        r["filename"] = vid["filename"]

    # Also return source frame for display
    source_frame = {
        "frame_b64": src["frame_b64s"][req.frame_idx],
        "timestamp": src["timestamps"][req.frame_idx],
        "filename": src["filename"],
    }

    return {"source": source_frame, "results": results}


# ── DTW Arc Alignment ─────────────────────────────────────────────────────────

@app.post("/api/align")
def align_videos(req: AlignRequest):
    if req.video_id_a not in store.videos:
        raise HTTPException(404, f"Video {req.video_id_a} not found")
    if req.video_id_b not in store.videos:
        raise HTTPException(404, f"Video {req.video_id_b} not found")

    vid_a = store.videos[req.video_id_a]
    vid_b = store.videos[req.video_id_b]

    arc_a = np.array(vid_a["arc"], dtype=np.float32)
    arc_b = np.array(vid_b["arc"], dtype=np.float32)

    dist, path = dtw_align(arc_a, arc_b)

    response = {
        "dtw_distance": dist,
        "arc_a": vid_a["arc"],
        "arc_b": vid_b["arc"],
        "timestamps_a": vid_a["timestamps"],
        "timestamps_b": vid_b["timestamps"],
        "filename_a": vid_a["filename"],
        "filename_b": vid_b["filename"],
        "n_frames_a": vid_a["n_frames"],
        "n_frames_b": vid_b["n_frames"],
        # Sampled path (every 5th point to keep JSON small)
        "path_sampled": path[::5],
    }

    # If a specific frame in A was queried, find equivalent in B
    if req.query_frame_idx is not None:
        eq_idx = find_equivalent_moment(req.query_frame_idx, arc_a, arc_b, path)
        response["equivalent"] = {
            "frame_idx_a": req.query_frame_idx,
            "frame_idx_b": eq_idx,
            "timestamp_a": vid_a["timestamps"][req.query_frame_idx],
            "timestamp_b": vid_b["timestamps"][eq_idx],
            "frame_b64_a": vid_a["frame_b64s"][req.query_frame_idx],
            "frame_b64_b": vid_b["frame_b64s"][eq_idx],
        }

    return response


# ── Scene DNA ─────────────────────────────────────────────────────────────────

@app.post("/api/scene-dna")
def scene_dna(req: SceneDNARequest):
    if not store.videos:
        raise HTTPException(400, "No videos loaded")

    target_ids = req.video_ids or list(store.videos.keys())
    if len(target_ids) < 2:
        raise HTTPException(400, "Scene DNA requires at least 2 videos")

    all_embeddings = {vid: store.videos[vid]["embeddings"] for vid in target_ids if vid in store.videos}

    result = scene_dna_cluster(all_embeddings, n_clusters=req.n_clusters)

    # Attach frame thumbnails per video per cluster
    frame_info = {}
    for vid_id in target_ids:
        if vid_id not in store.videos:
            continue
        vid = store.videos[vid_id]
        cluster_labels = result["clusters"][vid_id]
        frame_info[vid_id] = {
            "filename": vid["filename"],
            "frames": [
                {
                    "frame_idx": i,
                    "timestamp": vid["timestamps"][i],
                    "cluster": cluster_labels[i],
                    "frame_b64": vid["frame_b64s"][i],
                }
                for i in range(vid["n_frames"])
            ],
        }

    result["frame_info"] = frame_info
    return result


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r") as f:
        return f.read()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    print(f"[server] Starting on {host}:{port}")
    print(f"[server] Model: {MODEL_NAME}")
    uvicorn.run(app, host=host, port=port, timeout_keep_alive=300)
