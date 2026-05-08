# 🚀 Zero-Shot Action Recognition

This repository explores state-of-the-art vision-language approaches for **Zero-Shot Action Recognition** on the UCF101 dataset. We evaluate and compare three distinct methodologies: **Semantic Prompting with CLIP (SP-CLIP)**, **X-CLIP (Temporal-Aware CLIP)**, and **Direct Inference with Qwen-VL**.

---

## 📊 Performance Comparison

| Method | Accuracy | Latency (Avg/Frame) | Approach Type |
| :--- | :--- | :--- | :--- |
| **X-CLIP (Regular)** | **83.66%** | **0.023s** | Temporal-Aware Prompting |
| **EVA CLIP (Regular)** | 79.46% | 0.022s* | Averaged Template Prompting |
| **CLIP (Regular)** | 77.35% | 0.030s | Averaged Template Prompting |
| **X-CLIP (SP)** | 74.68% | 0.024s | Semantic Decomposition + Temporal-Awareness |
| **SP-CLIP** | 73.22% | 0.030s | Semantic Decomposition |
| **Qwen-VL** | 65.90% | 0.226s | VLM Image Captioning |

> [!NOTE]
> X-CLIP models utilize the microsoft/xclip-large-patch14 architecture, standard CLIP models utilize the openai/clip-vit-large-patch14 architecture, and EVA CLIP utilizes the EVA02-L-14 architecture.
> Qwen/Qwen2.5-VL-3B-Instruct was used for Qwen.
> *EVA CLIP latency is significantly higher because the model was forced to run on the CPU (it encounters infinite hangs when compiling on Apple Silicon MPS).

---

## 🛠️ Implementation Details

### 1. X-CLIP (Temporal-Aware CLIP)
This method utilizes **X-CLIP**, a Microsoft adaptation of CLIP designed specifically for video. Unlike standard CLIP which treats frames as independent images, X-CLIP incorporates temporal modeling to capture the progression of actions over time.

- **Regular Prompting**: Uses ActionCLIP-style multi-template prompts (e.g., *"a person is [LABEL]"*, *"doing [LABEL] in the gym"*). Averaging embeddings across multiple templates significantly boosts robustness.
- **Semantic Prompting (SP)**: Applies the same semantic decomposition logic used in SP-CLIP (Intent, Motion, Object) but leverages X-CLIP's video encoder for better temporal feature extraction.
- **Efficiency**: Highly optimized for 8-frame sampling, providing the best trade-off between speed and accuracy.

### 2. CLIP/EVA-02 (Regular)
This method utilizes the standard **CLIP** (ViT-L/14) and **EVA-02-L-14** models by decomposing complex action labels into more descriptive semantic components.

- **Implementation**: We aggregate embeddings from multiple prompts for each class and then find the class with the highest similarity to the video frames.

- **Example** Action "Archery":
    - **Prompt Templates**: 
        - "a person is [LABEL]"
        - "an example of [LABEL]"
        - "a demonstration of [LABEL]"


### 3. SP-CLIP (Semantic Prompting)
This baseline method leverages the standard **CLIP** (ViT-L/14) model by decomposing complex action labels into more descriptive semantic components.

- **Implementation**: We break down each of the 101 action classes into three semantic layers: **Intent**, **Motion**, and **Object**. 
- **Scoring**: Embeddings for these descriptions are precomputed. During inference, we calculate the cosine similarity between the video frames and these three layers, aggregating the scores (0.4/0.4/0.2 weights).
- **📝 Example (Archery)**: 
  - *Intent*: "A person aiming to hit a distant target with precision."
  - *Motion*: "A slow tensioning of the arms followed by a sudden release."
  - *Object*: "A bow, an arrow, and a circular target board."

### 4. Qwen-VL (Vision-Language Model)
This approach utilizes **Qwen-VL**, a large-scale vision-language model capable of understanding images and text in a unified space.

- **Implementation**: The model is prompted directly with a video clip (sampled at 1 FPS). We provide the entire list of 101 possible labels and ask it to select the most appropriate one.
- **Prompting**: The model is constrained by a closed-set instruction: "Identify the action in this video from the following list: {labels_str}. Output only the exact label".

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
