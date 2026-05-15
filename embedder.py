"""
Qwen3-VL-Embedding wrapper using vLLM pooling runner.
Handles both text and image (frame) inputs.
"""

import numpy as np
import os
from typing import List, Dict, Any, Optional, Union
from PIL import Image

# Lazy load — vLLM init is slow, only done once
_llm = None


def get_llm(model_name: str = "Qwen/Qwen3-VL-Embedding-2B"):
    global _llm
    if _llm is None:
        from vllm import LLM
        print(f"[embedder] Loading {model_name} with vLLM...")
        _llm = LLM(
            model=model_name,
            runner="pooling",
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=8192,  # enough for frames; saves VRAM vs 262k default
        )
        print("[embedder] Model loaded.")
    return _llm


def _format_conversation(
    input_dict: Dict[str, Any],
    default_instruction: str = "Represent the visual and semantic content.",
) -> List[Dict]:
    content = []
    instruction = input_dict.get("instruction") or default_instruction
    text = input_dict.get("text")
    image = input_dict.get("image")  # PIL.Image or file path or URL

    if image is not None:
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                image_content = image
            else:
                image_content = "file://" + os.path.abspath(image)
            content.append({"type": "image", "image": image_content})
        elif isinstance(image, Image.Image):
            content.append({"type": "image", "image": image})
        else:
            content.append({"type": "image", "image": image})

    if text:
        content.append({"type": "text", "text": text})

    if not content:
        content.append({"type": "text", "text": ""})

    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]


def _prepare_vllm_input(input_dict: Dict[str, Any], llm) -> Dict[str, Any]:
    conversation = _format_conversation(input_dict)
    prompt_text = llm.llm_engine.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    multi_modal_data = None
    image = input_dict.get("image")
    if image is not None:
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                from vllm.multimodal.utils import fetch_image
                try:
                    img_obj = fetch_image(image)
                    multi_modal_data = {"image": img_obj}
                except Exception as e:
                    print(f"[embedder] Warning: could not fetch image {image}: {e}")
            else:
                abs_path = os.path.abspath(image)
                if os.path.exists(abs_path):
                    img_obj = Image.open(abs_path).convert("RGB")
                    multi_modal_data = {"image": img_obj}
        elif isinstance(image, Image.Image):
            multi_modal_data = {"image": image.convert("RGB")}

    return {"prompt": prompt_text, "multi_modal_data": multi_modal_data}


def embed_batch(
    inputs: List[Dict[str, Any]],
    model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
) -> np.ndarray:
    """
    Embed a batch of inputs. Each input is a dict with optional keys:
      - 'text': str
      - 'image': PIL.Image | file path | URL
      - 'instruction': str (optional, overrides default)
    Returns numpy array of shape (N, D).
    """
    llm = get_llm(model_name)
    vllm_inputs = [_prepare_vllm_input(inp, llm) for inp in inputs]
    outputs = llm.embed(vllm_inputs)
    embeddings = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    return embeddings


def embed_text(text: str, instruction: Optional[str] = None, model_name: str = "Qwen/Qwen3-VL-Embedding-2B") -> np.ndarray:
    inp = {"text": text}
    if instruction:
        inp["instruction"] = instruction
    return embed_batch([inp], model_name=model_name)[0]


def embed_frames(frames: List[Image.Image], model_name: str = "Qwen/Qwen3-VL-Embedding-2B") -> np.ndarray:
    """Embed a list of PIL frames. Returns (N, D)."""
    inputs = [{"image": frame, "instruction": "Represent the visual content of this video frame."} for frame in frames]
    return embed_batch(inputs, model_name=model_name)
