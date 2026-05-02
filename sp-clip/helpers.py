import torch
import open_clip
import cv2
from PIL import Image
import json
import numpy as np
import csv
import time
import os

from datasets import load_dataset

# --- 1. Setup & Model Loading ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:  
    device = torch.device("cpu")

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14', 
    pretrained='openai'  
)
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer('ViT-L-14')

# --- 2. Optimized Extraction Functions ---

def extract_sp_clip_features_list(frames: list, batch_size=16):
    """
    Processes frames in batches to avoid OOM while still utilizing GPU speed.
    """
    # 1. Preprocess all frames on CPU and stack
    # [TotalFrames, C, H, W]
    all_frames_tensor = torch.stack([preprocess(f) for f in frames])
    
    num_frames = all_frames_tensor.shape[0]
    all_features = []

    with torch.no_grad():
        # 2. Loop through the tensor in chunks of batch_size
        for i in range(0, num_frames, batch_size):
            # Slice the batch and move only this chunk to GPU
            batch = all_frames_tensor[i : i + batch_size].to(device)
            
            # Encode
            batch_features = model.encode_image(batch)
            
            # Normalize individual frame vectors
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            
            # Move back to CPU or keep on GPU (keeping on GPU is faster for the final mean)
            all_features.append(batch_features)
            
        # 3. Concatenate all batches back together
        # Result shape: [TotalFrames, Dim]
        video_features = torch.cat(all_features, dim=0)
        
        # 4. Calculate mean across the total frame dimension
        mean_features = video_features.mean(dim=0, keepdim=True)
        
        # 5. Re-normalize the final averaged vector
        mean_features /= mean_features.norm(dim=-1, keepdim=True)
        
    return mean_features

def extract_text_features(text: str):
    """Extracts text features and keeps them on the GPU."""
    tokens = tokenizer(text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features 

# --- 3. Dictionary & Matrix Management ---

def precompute_sp_embeddings(prompt_path):
    """Loads prompts and stores them as a dictionary of GPU Tensors."""
    with open(prompt_path, 'r') as f:
        prompts = json.load(f)
    
    sp_dictionary = {}
    print(f"Precomputing embeddings for {len(prompts)} classes...")
    
    for action_class, levels in prompts.items():
        sp_dictionary[action_class] = {
            "intent": extract_text_features(levels['intent']),
            "motion": extract_text_features(levels['motion']),
            "object": extract_text_features(levels['object'])
        }
            
    return sp_dictionary

def stack_sp_embeddings(sp_dictionary):
    """Stacks dictionary items into fixed matrices on the GPU."""
    action_classes = list(sp_dictionary.keys())
    
    intent_list = [sp_dictionary[cls]['intent'] for cls in action_classes]
    motion_list = [sp_dictionary[cls]['motion'] for cls in action_classes]
    object_list = [sp_dictionary[cls]['object'] for cls in action_classes]

    # Use torch.cat to keep everything on device as [NumClasses, Dim]
    return (
        torch.cat(intent_list, dim=0), 
        torch.cat(motion_list, dim=0), 
        torch.cat(object_list, dim=0),
        action_classes
    )

# --- 4. Main Inference Engine ---

def run_sp_clip_inference(video_frames: list, sp_matrices):
    """Runs inference without ever leaving the GPU until the final result."""
    intent_mat, motion_mat, object_mat, action_classes = sp_matrices
    
    # 1. Get video vector 
    video_tensor = extract_sp_clip_features_list(video_frames)

    # 2. Similarity (Matrix Multiplication)
    # video_tensor: [1, Dim] @ matrices: [Dim, NumClasses]
    intent_scores = video_tensor @ intent_mat.T
    motion_scores = video_tensor @ motion_mat.T
    object_scores = video_tensor @ object_mat.T

    # 3. Weighted Aggregation (SP-CLIP Logic)
    final_scores = (0.4 * intent_scores) + (0.4 * motion_scores) + (0.2 * object_scores)

    # 4. Get the highest score prediction
    predicted_idx = torch.argmax(final_scores).item()
    predicted_class = action_classes[predicted_idx]


    return { 
        "predicted_class": predicted_class,
        "intent_score": intent_scores[0, predicted_idx].item(), 
        "motion_score": motion_scores[0, predicted_idx].item(), 
        "object_score": object_scores[0, predicted_idx].item(), 
        "final_score": final_scores[0, predicted_idx].item()
    }


def downsize_clip(frames, fps, stream_fps=30):
    """
    Downsize a video clip to the specified frames per second (fps).
    
    Parameters:
    frames (list): A list of frames from the video clip.
    fps (int): The desired frames per second for the downsized clip.
    """
    step = int(stream_fps / fps)  
    clip_frames = frames[::step]   
    return clip_frames  

def get_clip_generator(dataset_split, max_clips=10):
    """
    Yields one full clip at a time. 
    max_clips determines the total number of full actions to process.
    """
    current_row = 0
    total_rows = len(dataset_split)
    clips_yielded = 0
    
    while clips_yielded < max_clips and current_row < total_rows:
        first_row = dataset_split[current_row]
        clip_id = first_row.get("clip_id", f"clip_{clips_yielded}")
        label_id = first_row["label"]
        
        # Resolve action name
        label_name = dataset_split.features["label"].int2str(label_id)
        
        clip_images = []
        
        # Group all rows that belong to the same clip_id
        while current_row < total_rows:
            row = dataset_split[current_row]
            # Check if we are still in the same clip
            if row.get("clip_id") == clip_id:
                clip_images.append(row["image"])
                current_row += 1
            else:
                # We hit a new clip, break inner loop but don't increment current_row
                # so the next iteration of the outer loop picks it up
                break
        
        clips_yielded += 1
        yield {
            "clip_id": clip_id,
            "label_name": label_name,
            "images": clip_images
        }
        




    
    
