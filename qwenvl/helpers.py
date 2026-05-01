import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


## --- 1. SETUP FUNCTIONS --- ##

def downsize_clip(frames, fps):
    """
    Downsize a video clip to the specified frames per second (fps).
    
    Parameters:
    frames (list): A list of frames from the video clip.
    fps (int): The desired frames per second for the downsized clip.
    """
    step = int(30 / fps)  
    clip_frames = frames[::step]   
    return clip_frames  


def load_qwen_model(model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
    """Initialize the model and processor."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

## --- 2. DATA STRATEGY FUNCTIONS --- ##

def get_clip_generator(dataset_split, max_rows=1000):
    """
    Efficiently yields one full clip at a time using contiguous indexing.
    """
    current_row = 0
    total_len = len(dataset_split)
    
    while current_row < max_rows and current_row < total_len:
        first_row = dataset_split[current_row]
        clip_id = first_row["clip_id"]
        label_id = first_row["label"]
        
        # Resolve action name from metadata
        label_name = dataset_split.features["label"].int2str(label_id)
        
        clip_images = []
        # Fast inner loop for contiguous frames
        while current_row < total_len:
            row = dataset_split[current_row]
            if row["clip_id"] == clip_id:
                clip_images.append(row["image"])
                current_row += 1
            else:
                break
        
        yield {
            "clip_id": clip_id,
            "label_name": label_name,
            "images": clip_images
        }

## --- 3. INFERENCE ENGINE --- ##

def run_inference(model, processor, images, prompt="Describe the action in this video."):
    """Handles the Qwen-VL conversation and generation logic."""
    
    # 1. Downsize using your helper
    
    
    # 2. Format Messages
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": images, "fps": 1.0},
            {"type": "text", "text": prompt},
        ],
    }]

    # 3. Process Inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # 4. Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        
    # Trim the prompt from the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]


# --- 4. VISUALIZATION FUNCTION --- #



def visualize_clip(clip_frames, fps=5):
    """
    Plays a list of frames as a video in an OpenCV window.
    """
    print("Press 'q' to close the video window.")
    
    # Calculate delay in milliseconds
    delay = int(1000 / fps)

    for idx, frame in enumerate(clip_frames):
        # 1. Convert PIL image to NumPy array if needed
        frame_np = np.array(frame)

        # 2. Convert RGB to BGR (OpenCV uses BGR)
        # Only do this if your images are RGB (Standard for PIL/HuggingFace)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        # 3. Add a label so you know which frame you are seeing
        cv2.putText(frame_bgr, f"Frame: {idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 4. Show the frame
        cv2.imshow("UCF-Crime Clip Visualization", frame_bgr)

        # 5. Wait and check for 'q' key to quit
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    # A small extra wait to ensure the window closes on macOS
    cv2.waitKey(1) 