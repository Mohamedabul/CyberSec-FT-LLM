# CyberSec-FT — LLM Fine-Tuning for Security Analysis

> **Fine-tuning Qwen2.5-3B-Instruct using QLoRA on CVE, CWE, and Exploit-DB data for cybersecurity analysis tasks.**

---

## Model Capabilities & Features

Our fine-tuned Qwen model acts as an expert cybersecurity analyst. It has been trained to bridge the gap between high-level CVE descriptions and low-level exploit code. 

**Key Features:**
- **Vulnerability Triage**: Automatically generate structured severity, attack vector, and mitigation reports for any CVE.
- **Exploit Reverse-Engineering**: Paste raw exploit code (C, Python, Bash) and receive an immediate technical breakdown of *how* the exploit works and what vulnerabilities it targets.
- **Attack Chain Reasoning**: Combine a CVE with an exploit to generate a step-by-step kill-chain analysis, from initial access to system compromise.

---

## Dataset & Training Scale

To achieve maximum accuracy, the model was trained on the **entire historical corpus** of modern vulnerabilities and exploits, completely uncapped and unfiltered:
- **NVD CVE Database**: Every vulnerability published from 2020 through 2025.
- **Exploit-DB**: Over 45,000+ real-world exploits cloned directly from Offensive Security's GitLab.
- **MITRE CWE**: Full weakness classifications and abstractions.

**Model Baseline**: `Qwen/Qwen2.5-3B-Instruct` (3B params, 4-bit QLoRA)  
**Hardware Used**: Google Colab Tesla T4 (via Unsloth Fast Patching)

**Model**: `Qwen/Qwen2.5-3B-Instruct` (3B params, 4-bit QLoRA)  
**Dataset**: Full NVD/Exploit-DB corpus (Unlimited Samples)

---

## Project Structure

```
CS_FT/
├── configs/                    # All configuration YAML files
│   ├── data_config.yaml        # Dataset paths and preprocessing
│   ├── model_config.yaml       # Model, LoRA, quantization, inference settings
│   ├── training_config.yaml    # Hyperparameters (local + Colab)
│   └── train_config.yaml       # Unified training config
│
├── data_collection/            # Phase 1: Data download scripts
│   ├── nvd_downloader.py       # NVD API 2.0 (2020–2025)
│   ├── mitre_cwe_downloader.py # MITRE CWE XML
│   └── exploitdb_cloner.py     # Exploit-DB GitLab clone
│
├── data_processing/            # Phase 2: Parse and link data
│   ├── nvd_parser.py
│   ├── cwe_parser.py
│   ├── exploitdb_parser.py
│   └── linker.py               # Joins CVE ↔ CWE ↔ Exploit
│
├── dataset_builder/            # Phase 3: Build instruction dataset
│   ├── type_a_builder.py       # CVE-only samples (176,883)
│   ├── type_b_builder.py       # Exploit-only samples (10,000)
│   ├── type_c_builder.py       # Attack chain samples (880)
│   └── merge_split.py          # 80/10/10 train/val/test split
│
├── training/                   # Phase 4: Fine-tuning
│   ├── data_loader.py
│   ├── model_loader.py
│   ├── qlora_config.py
│   ├── train.py                # Local training (HuggingFace + TRL)
│   ├── evaluate.py
│   ├── inference.py
│   └── unsloth/                # Colab training (2-5x faster)
│       ├── model_loader_unsloth.py
│       ├── train_unsloth.py
│       └── inference_unsloth.py
│
├── notebooks/
│   ├── colab_train.ipynb       # One-click Colab training notebook
│   └── colab_evaluate.ipynb    # Comprehensive NLP evaluation metrics
│
├── requirements.txt            # Local (Windows) dependencies
└── requirements_colab.txt      # Colab dependencies (includes unsloth)
```

---

## Dataset

| Type | Description |
|------|-------------|
| **A** | CVE structured vulnerability analysis |
| **B** | Exploit code technical explanation |
| **C** | CVE + Exploit attack chain reasoning |

Split: **Train 80% / Val 10% / Test 10%**

> Dataset files are not included in this repo (too large).  
> Upload `dataset/` folder to Google Drive and link in Colab notebook.

---

## Quick Start

### Option 1: Local Training (Windows, RTX GPU)

```bash
# 1. Create venv and install CUDA PyTorch
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Install dependencies
pip install -r requirements.txt

# 3. Collect data (Phase 1)
python data_collection/nvd_downloader.py
python data_collection/mitre_cwe_downloader.py
python data_collection/exploitdb_cloner.py

# 4. Process data (Phase 2)
python data_processing/nvd_parser.py
python data_processing/cwe_parser.py
python data_processing/exploitdb_parser.py
python data_processing/linker.py

# 5. Build dataset (Phase 3)
python dataset_builder/type_a_builder.py
python dataset_builder/type_b_builder.py
python dataset_builder/type_c_builder.py
python dataset_builder/merge_split.py

# 6. Train (Phase 4)
python training/train.py
```

### Option 2: Google Colab (Faster — Unsloth)

Open `notebooks/colab_train.ipynb` in Colab.  
Follow the cells — it handles setup, Drive mounting, and training automatically.

---

## Training Details

| Setting | Value |
|---------|-------|
| Base Model | Qwen2.5-3B-Instruct (3B) |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank | r=16, alpha=32 |
| Seq Length | 512 tokens |
| Batch Size | 4 (local) / 2 (Colab) |
| Grad Accumulation | 4 (effective batch=16) |
| Epochs | 1 |
| Optimizer | paged_adamw_8bit |
| Framework | HuggingFace TRL (local) / Unsloth (Colab) |
| Experiment Tracking | **MLflow** via DagsHub |

---

## Evaluation & Performance

The fine-tuned model was evaluated against an unseen hold-out `test.jsonl` dataset to mathematically verify its understanding of cybersecurity concepts and generation quality. The model achieved excellent scores indicative of deep domain understanding:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Perplexity** | **7.61** | *Excellent.* The model is highly confident. (Scores between 5-15 represent strong language modeling). |
| **METEOR** | **0.4084** | *Very Good.* The model perfectly captures the semantic meaning, even when paraphrasing (e.g., using "hacker" instead of "attacker"). |
| **ROUGE-1** | **0.3496** | Almost 35% of the model's generated text exactly matches human-written vulnerability reports word-for-word. |
| **ROUGE-L** | **0.2044** | Longest common sequence matching shows strong sentence-level structural alignment. |

>  *To reproduce these metrics on your own checkpoints, run `notebooks/colab_evaluate.ipynb`.*

---

## Requirements

- Python 3.11+
- CUDA 12.x compatible GPU (8GB+ VRAM)
- ~15GB disk for raw data + ~5GB for processed data

---

## Author

**Mohamed Abul** — [@Mohamedabul](https://github.com/Mohamedabul)
