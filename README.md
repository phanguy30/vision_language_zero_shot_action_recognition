# 🚀 Zero-Shot Action Recognition

This repository explores state-of-the-art vision-language approaches for **Zero-Shot Action Recognition** on the UCF101 dataset. We evaluate and compare two distinct methodologies: **Semantic Prompting with CLIP (SP-CLIP)** and **Direct Inference with Qwen-VL**.

---

## 📊 Performance Comparison

| Method | Accuracy | Inference Time (Avg) | Approach Type |
| :--- | :--- | :--- | :--- |
| **SP-CLIP** | **73.22%** | **0.03s** / frame | Semantic Decomposition |
| **Qwen-VL** | 65.90% | 0.23s / frame | VLM Direct Prompting |

---

## 🛠️ Implementation Details

### 1. SP-CLIP (Semantic Prompting)
This method leverages the **CLIP** (Contrastive Language-Image Pre-training) model by decomposing complex action labels into more descriptive semantic components. Instead of matching a single word like "Archery," we match against a richer set of descriptions.

- **Implementation**: We break down each of the 101 action classes into three semantic layers: **Intent**, **Motion**, and **Object**. 
- **Scoring**: Embeddings for these descriptions are precomputed. During inference, we calculate the cosine similarity between the video frames and these three layers, aggregating the scores to produce the final prediction.
- **Efficiency**: Extremely fast due to precomputed text embeddings and the lightweight nature of CLIP's vision encoder.

#### 📝 Example Prompt (Archery)
- **Intent**: *"A person aiming to hit a distant target with precision."*
- **Motion**: *"A slow tensioning of the arms followed by a sudden release."*
- **Object**: *"A bow, an arrow, and a circular target board."*

---

### 2. Qwen-VL (Vision-Language Model)
This approach utilizes **Qwen-VL**, a large-scale vision-language model capable of understanding images and text in a unified space.

- **Implementation**: The model is prompted directly with a video clip (sampled at 1 FPS). We provide the model with the entire list of 101 possible labels and ask it to select the most appropriate one.
- **Flexibility**: Highly flexible as it requires no manual feature engineering or label decomposition. It relies on the model's inherent reasoning capabilities.
- **Latency**: Higher latency compared to SP-CLIP due to the large parameter count and the need to process the full list of labels in the prompt.

#### 📝 Example Prompt
> *"Analyze this video clip. From the following list, pick the action that best describes what is happening: ApplyEyeMakeup, ApplyLipstick, Archery, ..., YoYo. Provide only the label name and nothing else."*

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
- `sp-clip/test.ipynb`: Evaluate the Semantic Prompting approach.
- `qwenvl/test.ipynb`: Evaluate the Qwen-VL approach.


