# 🚀 Zero-Shot Action Recognition

This repository explores state-of-the-art vision-language approaches for **Zero-Shot Action Recognition** on the UCF101 dataset. We evaluate and compare three distinct methodologies: **Semantic Prompting with CLIP (SP-CLIP)**, **X-CLIP (Temporal-Aware CLIP)**, and **Direct Inference with Qwen-VL**.

---

## 📊 Performance Comparison

| Method | Accuracy | Latency (Avg/Frame) | Approach Type |
| :--- | :--- | :--- | :--- |
| **X-CLIP (Regular)** | **83.66%** | **0.022s** | Temporal-Aware Prompting |
| **X-CLIP (SP)** | 74.68% | 0.024s | Semantic Decomposition + Temporal-Awareness |
| **SP-CLIP** | 73.22% | 0.03s | Semantic Decomposition |
| **Qwen-VL** | 65.90% | 0.23s | VLM Image Captioning |

> [!NOTE]
> X-CLIP models utilize the microsoft/xclip-large-patch14 architecture, while standard CLIP models utilize the openai/clip-vit-large-patch14 architecture.
> Qwen/Qwen2.5-VL-3B-Instruct was used for Qwen

---

## 🛠️ Implementation Details

### 1. X-CLIP (Temporal-Aware CLIP)
This method utilizes **X-CLIP**, a Microsoft adaptation of CLIP designed specifically for video. Unlike standard CLIP which treats frames as independent images, X-CLIP incorporates temporal modeling to capture the progression of actions over time.

- **Regular Prompting**: Uses ActionCLIP-style multi-template prompts (e.g., *"a person is [LABEL]"*, *"doing [LABEL] in the gym"*). Averaging embeddings across multiple templates significantly boosts robustness.
- **Semantic Prompting (SP)**: Applies the same semantic decomposition logic used in SP-CLIP (Intent, Motion, Object) but leverages X-CLIP's video encoder for better temporal feature extraction.
- **Efficiency**: Highly optimized for 8-frame sampling, providing the best trade-off between speed and accuracy.

### 2. SP-CLIP (Semantic Prompting)
This baseline method leverages the standard **CLIP** (ViT-L/14) model by decomposing complex action labels into more descriptive semantic components.

- **Implementation**: We break down each of the 101 action classes into three semantic layers: **Intent**, **Motion**, and **Object**. 
- **Scoring**: Embeddings for these descriptions are precomputed. During inference, we calculate the cosine similarity between the video frames and these three layers, aggregating the scores (0.4/0.4/0.2 weights).
- **📝 Example (Archery)**: 
  - *Intent*: "A person aiming to hit a distant target with precision."
  - *Motion*: "A slow tensioning of the arms followed by a sudden release."
  - *Object*: "A bow, an arrow, and a circular target board."

### 3. Qwen-VL (Vision-Language Model)
This approach utilizes **Qwen-VL**, a large-scale vision-language model capable of understanding images and text in a unified space.

- **Implementation**: The model is prompted directly with a video clip (sampled at 1 FPS). We provide the entire list of 101 possible labels and ask it to select the most appropriate one.
- **Prompting**: The model is constrained by a closed-set instruction: "Identify the action in this video from the following list: {labels_str}. Output only the exact label" name.

- **Latency**: Higher latency compared to CLIP-based methods due to the large parameter count and long prompt context.

---

## ⚙️ Setup & Usage

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Tests
Each method has its own directory containing a `test.ipynb` notebook and `helpers.py` for core logic.
- `xclip/regular-prompting-test.ipynb`: Evaluate the top-performing X-CLIP approach.
- `xclip/sp-test.ipynb`: Evaluate X-CLIP with Semantic Prompting.
- `sp-clip/test.ipynb`: Evaluate the CLIP baseline with Semantic Prompting.
- `qwenvl/test.ipynb`: Evaluate the Qwen-VL VLM approach.
