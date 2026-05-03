import torch
import json
import numpy as np
import csv
import time
import os
import torch.nn.functional as F

from PIL import Image
from transformers import XCLIPProcessor, XCLIPModel
from datasets import load_dataset

# --- 1. Setup & Model Loading ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

MODEL_NAME = "microsoft/xclip-large-patch14"

processor = XCLIPProcessor.from_pretrained(MODEL_NAME)
model = XCLIPModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
model.eval()

print(f"X-CLIP loaded on: {device}")

# --- 2. Feature Extraction ---

# X-CLIP expects exactly 8 frames per clip (standard across all xclip variants)
NUM_FRAMES = 8

def sample_frames(frames: list, num_frames: int = NUM_FRAMES) -> list:
    """
    Uniformly samples `num_frames` from a list of PIL Images.
    If fewer frames than needed, repeats the last frame.
    """
    n = len(frames)
    if n == 0:
        raise ValueError("frames list is empty")
    if n >= num_frames:
        indices = np.linspace(0, n - 1, num_frames, dtype=int)
        return [frames[i] for i in indices]
    else:
        # Pad by repeating the last frame
        return frames + [frames[-1]] * (num_frames - n)


def debug_video_output(frames: list = None):
    """
    Diagnostic helper — call this from the notebook to inspect exactly what
    model.get_video_features() returns in your installed transformers version.
    Pass a list of PIL Images, or leave None to use a blank dummy clip.
    """
    if frames is None:
        dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        frames = [dummy] * NUM_FRAMES

    sampled = sample_frames(frames, NUM_FRAMES)
    inputs = processor(videos=[sampled], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out = model.get_video_features(pixel_values=inputs["pixel_values"])

    print("=== get_video_features output ===")
    print("type         :", type(out))
    print("is Tensor    :", isinstance(out, torch.Tensor))
    if isinstance(out, torch.Tensor):
        print("shape        :", out.shape)
    else:
        # It's a ModelOutput dataclass — inspect every attribute
        for attr in vars(out):
            val = getattr(out, attr)
            if isinstance(val, torch.Tensor):
                print(f"  .{attr:30s} shape={val.shape}  dtype={val.dtype}")
            elif val is None:
                print(f"  .{attr:30s} None")
            elif isinstance(val, (tuple, list)):
                print(f"  .{attr:30s} {type(val).__name__} len={len(val)}")
    return out


def extract_xclip_video_features(frames: list) -> torch.Tensor:
    """
    Given a list of PIL Images (one clip), samples NUM_FRAMES uniformly,
    processes them through X-CLIP, and returns a normalized video embedding [1, Dim].
    """
    sampled = sample_frames(frames, NUM_FRAMES)

    inputs = processor(
        videos=[sampled],      # batch of 1 clip
        return_tensors="pt",
        padding=True
    ).to(device)
    # pixel_values: [1, T, C, H, W]

    with torch.no_grad():
        video_out = model.get_video_features(**inputs)

        # In this transformers version, get_video_features returns BaseModelOutputWithPooling.
        # pooler_output is [1, 512] — already temporally aggregated and projected.
        # Do NOT re-project; just unwrap the tensor.
        if not isinstance(video_out, torch.Tensor):
            video_out = video_out.pooler_output  # [1, 512]

        video_features = video_out / video_out.norm(dim=-1, keepdim=True)

    return video_features  # [1, Dim]


def extract_text_features(text: str) -> torch.Tensor:
    """Extracts and normalizes X-CLIP text features. Returns [1, Dim]."""
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_out = model.get_text_features(**inputs)

        # Same pattern: unwrap pooler_output directly if not already a tensor.
        if not isinstance(text_out, torch.Tensor):
            text_out = text_out.pooler_output  # [1, 512]

        text_features = text_out / text_out.norm(dim=-1, keepdim=True)

    return text_features  # [1, Dim]


# --- 3. Dictionary & Matrix Management ---


def precompute_sp_embeddings(prompt_path: str) -> dict:
    """
    Loads SP prompts from JSON and precomputes X-CLIP text embeddings
    for each action class's intent, motion, and object descriptions.

    Expected JSON structure (same as SP-CLIP):
    {
        "ClassName": {
            "intent": "...",
            "motion": "...",
            "object": "..."
        },
        ...
    }
    """
    with open(prompt_path, 'r') as f:
        prompts = json.load(f)

    sp_dictionary = {}
    print(f"Precomputing X-CLIP embeddings for {len(prompts)} classes...")

    for action_class, levels in prompts.items():
        sp_dictionary[action_class] = {
            "intent": extract_text_features(levels["intent"]),
            "motion": extract_text_features(levels["motion"]),
            "object": extract_text_features(levels["object"])
        }

    return sp_dictionary


def stack_sp_embeddings(sp_dictionary: dict):
    """
    Stacks per-class SP embeddings into fixed matrices for fast batch similarity.

    Returns:
        intent_mat  [NumClasses, Dim]
        motion_mat  [NumClasses, Dim]
        object_mat  [NumClasses, Dim]
        action_classes (list of str)
    """
    action_classes = list(sp_dictionary.keys())

    intent_list = [sp_dictionary[cls]["intent"] for cls in action_classes]
    motion_list = [sp_dictionary[cls]["motion"] for cls in action_classes]
    object_list = [sp_dictionary[cls]["object"] for cls in action_classes]

    return (
        torch.cat(intent_list, dim=0),   # [NumClasses, Dim]
        torch.cat(motion_list, dim=0),
        torch.cat(object_list, dim=0),
        action_classes
    )


def precompute_action_clip(prompt_path: str) -> dict:
    """
    Loads ActionCLIP-style prompts from JSON and precomputes
    X-CLIP text embeddings. If multiple templates are provided per class,
    their embeddings are averaged.
    """
    with open(prompt_path, 'r') as f:
        prompts = json.load(f)

    dictionary = {}
    print(f"Precomputing X-CLIP embeddings for {len(prompts)} classes (ActionCLIP Style)...")

    for action_class, prompt_list in prompts.items():
        if isinstance(prompt_list, str):
            prompt_list = [prompt_list]

        feat_list = []
        for p in prompt_list:
            feat_list.append(extract_text_features(p))

        # Average and re-normalize
        avg_feat = torch.mean(torch.stack(feat_list), dim=0)
        avg_feat = avg_feat / avg_feat.norm(dim=-1, keepdim=True)

        dictionary[action_class] = avg_feat

    return dictionary


def stack_action_clip(dictionary: dict):
    """
    Stacks per-class embeddings into a single matrix.
    """
    action_classes = list(dictionary.keys())
    feat_list = [dictionary[cls] for cls in action_classes]
    return torch.cat(feat_list, dim=0), action_classes


# --- 4. Main Inference Engine ---

def run_sp_xclip_inference(video_frames: list, sp_matrices) -> dict:
    """
    Runs SP-style inference with X-CLIP:
      1. Encode video frames -> video vector [1, Dim]
      2. Compute similarity against intent/motion/object text matrices
      3. Weighted aggregation (same weights as SP-CLIP)
      4. Return predicted class and per-level scores
    """
    intent_mat, motion_mat, object_mat, action_classes = sp_matrices

    # 1. Get video embedding
    video_tensor = extract_xclip_video_features(video_frames)  # [1, Dim]

    # 2. Similarity: [1, Dim] @ [Dim, NumClasses] -> [1, NumClasses]
    intent_scores = video_tensor @ intent_mat.T
    motion_scores = video_tensor @ motion_mat.T
    object_scores = video_tensor @ object_mat.T

    # 3. Weighted aggregation (SP-CLIP logic)
    final_scores = (0.4 * intent_scores) + (0.4 * motion_scores) + (0.2 * object_scores)

    # 4. Predict
    predicted_idx = torch.argmax(final_scores).item()
    predicted_class = action_classes[predicted_idx]

    return {
        "predicted_class": predicted_class,
        "intent_score":    intent_scores[0, predicted_idx].item(),
        "motion_score":    motion_scores[0, predicted_idx].item(),
        "object_score":    object_scores[0, predicted_idx].item(),
        "final_score":     final_scores[0, predicted_idx].item()
    }


def run_action_clip_inference(video_frames: list, matrices) -> dict:
    """
    Runs ActionCLIP-style inference with X-CLIP.
    """
    text_mat, action_classes = matrices

    # 1. Get video embedding
    video_tensor = extract_xclip_video_features(video_frames)  # [1, Dim]

    # 2. Similarity: [1, Dim] @ [Dim, NumClasses] -> [1, NumClasses]
    scores = video_tensor @ text_mat.T

    # 3. Predict
    predicted_idx = torch.argmax(scores).item()
    predicted_class = action_classes[predicted_idx]

    return {
        "predicted_class": predicted_class,
        "final_score":     scores[0, predicted_idx].item()
    }


# --- 5. Video Utilities ---

def downsize_clip(frames: list, fps: int, stream_fps: int = 30) -> list:
    """
    Uniformly thins a list of frames to the target fps
    (same helper as in sp-clip for compatibility).
    """
    step = int(stream_fps / fps)
    return frames[::step]


def get_clip_generator(dataset_split, max_clips: int = 10):
    """
    Yields one full clip at a time from a HuggingFace dataset split.
    Identical to the sp-clip version for drop-in compatibility.

    Yields dicts:
        {
            "clip_id":    str,
            "label_name": str,
            "images":     list[PIL.Image]
        }
    """
    current_row = 0
    total_rows = len(dataset_split)
    clips_yielded = 0

    while clips_yielded < max_clips and current_row < total_rows:
        first_row = dataset_split[current_row]
        clip_id   = first_row.get("clip_id", f"clip_{clips_yielded}")
        label_id  = first_row["label"]

        label_name = dataset_split.features["label"].int2str(label_id)

        clip_images = []

        while current_row < total_rows:
            row = dataset_split[current_row]
            if row.get("clip_id") == clip_id:
                clip_images.append(row["image"])
                current_row += 1
            else:
                break

        clips_yielded += 1
        yield {
            "clip_id":    clip_id,
            "label_name": label_name,
            "images":     clip_images
        }
