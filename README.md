# Narrative Arc Alignment
### Cross-video semantic alignment · Qwen3-VL-Embedding · vLLM

---

## What it does

Three modes, one backend:

**1. Semantic Search**
Natural language → retrieve matching frames across all uploaded videos simultaneously.
Click any result frame → find visually/semantically similar frames in other videos.
Click anywhere on the arc chart → same cross-video frame query.

**2. DTW Arc Alignment**
Select two videos. The system computes embedding velocity (semantic change rate) for each,
builds a 1D arc signature, and aligns them with Dynamic Time Warping.
Click a frame on Video A's arc → instantly see the structurally equivalent moment in Video B,
even if content is completely different (e.g. a film trailer vs a sports highlight).

**3. Scene DNA**
Pools ALL frames from ALL videos, clusters them in joint Qwen3-VL embedding space.
Frames that land in the same cluster share semantic structure across different videos.
No query needed — pure unsupervised discovery.

---

## Setup

```bash
cd narrative-arc

# Install vLLM inrespective environment, here for cuda installation:
uv pip install vllm --torch-backend=auto 

# Install deps (vLLM already on HPC)
pip install fastapi uvicorn python-multipart scipy scikit-learn fastdtw \
            opencv-python-headless Pillow numpy --break-system-packages

# Or:
pip install -U -r requirements.txt
```

---

## Run on HPC

```bash
# Option 1: default (Qwen/Qwen3-VL-Embedding-2B from HuggingFace)
python main.py

# Option 2: local model path (recommended if HF is slow)
EMBEDDING_MODEL=/path/to/Qwen3-VL-Embedding-2B python main.py

# Option 3: custom FPS and port
EMBEDDING_MODEL=/path/to/model EXTRACT_FPS=1.5 PORT=8000 python main.py
```

Server starts on `0.0.0.0:8000`.

---

## Access in your local browser (SSH tunnel)

Run this on YOUR local machine (not the HPC):

```bash
ssh -L 8000:localhost:8000 <your_user>@<hpc_host>
```

Then open: http://localhost:8000

If the HPC uses a jump host:
```bash
ssh -L 8000:localhost:8000 -J <jumphost> <your_user>@<hpc_node>
```

---

## Model notes

Default: `Qwen/Qwen3-VL-Embedding-2B`
- 2B params, 2048-dim embeddings, image + text natively
- ~5GB VRAM, L4 24GB is very comfortable
- vLLM pooling runner handles batching automatically

To swap model, set `EMBEDDING_MODEL` env var to any HF model ID or local path.

---

## Performance (L4 24GB)

| Step | Time (approx) |
|------|--------------|
| Load model | ~30s (once) |
| Extract frames from 5min video @ 1fps | ~5s |
| Embed 300 frames | ~25s |
| Text query | ~0.5s |
| DTW align two 120-frame videos | ~0.1s |
| Scene DNA (3 videos, 360 frames) | ~2s |

---

## File structure

```
narrative-arc/
├── main.py          FastAPI routes
├── embedder.py      vLLM + Qwen3-VL-Embedding wrapper
├── pipeline.py      Frame extraction, velocity, DTW, clustering
├── static/
│   └── index.html   Full frontend (HTML/CSS/JS, no npm)
└── requirements.txt
```
