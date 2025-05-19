
# Multimodal Visual Question Answering with Amazon Berkeley Objects Dataset

![Visual Question Answering](https://img.shields.io/badge/Project-VQA-blueviolet) ![Dataset-ABO](https://img.shields.io/badge/Dataset-Amazon_Berkeley_Objects-green) ![Framework-HuggingFace](https://img.shields.io/badge/Framework-HuggingFace-orange) ![Environment-Kaggle](https://img.shields.io/badge/Environment-Kaggle-blue)

Welcome to the **VR_Project** repository, a comprehensive implementation of a Multimodal Visual Question Answering (VQA) system using the Amazon Berkeley Objects (ABO) dataset. This project, developed as part of the AIM 825 - Visual Recognition course (May 2025), focuses on curating a VQA dataset with single-word answers, evaluating pre-trained models (BLIP, BLIP-2, ViLT, CLIP), and fine-tuning them using Low-Rank Adaptation (LoRA) on Kaggle's free GPU infrastructure.

**Authors**:
- Ayush Kashyap (IMT2022129)
- Rishit Mane (IMT2022564)
- Rohan Rajesh (IMT2022575)

---

## 📖 Project Overview

This project aims to develop a robust VQA system tailored for e-commerce applications, leveraging the ABO dataset's rich multimodal data (147,702 product entries, 398,212 images). Key objectives include:

- **Dataset Curation**: Created a VQA dataset with 159,150 question-answer pairs, each with single-word answers, using the `abo-images-small` (3GB, 256x256 pixels) and `abo-listings` (83MB) datasets. The Gemini 2.0 API was used to generate diverse, visually answerable queries.
- **Baseline Evaluation**: Evaluated pre-trained models (BLIP, BLIP-2, ViLT, CLIP) using metrics like Accuracy, F1 Score, BERTScore, METEOR, and ROUGE-L.
- **Fine-Tuning**: Improved model performance using LoRA, achieving up to 62.5% test accuracy for BLIP (r=32) and 54.46% for ViLT (r=32).
- **Resource Constraints**: All experiments were conducted on Kaggle's dual 16GB GPU environment, adhering to a 7-billion-parameter limit.

The curated dataset, code, and results are fully available in this repository.

---

## 🚀 Features

- **Custom VQA Dataset**: 159,150 question-answer pairs, with each image linked to five diverse queries (color, shape, quantity, etc.).
- **Model Evaluation**: Comprehensive baseline and fine-tuned performance metrics for BLIP, BLIP-2, ViLT, and CLIP.
- **LoRA Fine-Tuning**: Parameter-efficient fine-tuning with LoRA ranks (r=8, 16, 32) for BLIP and ViLT.
- **Kaggle-Compatible**: Optimized for Kaggle's free GPU environment with mixed-precision training (fp16).
- **Reproducible Code**: Jupyter notebooks for data curation, baseline evaluation, fine-tuning, and inference.

---

## 📂 Project Structure

The repository is organized as follows:

```
VR_Project/
├── Baseline/                          # Notebooks for baseline model evaluation
│   ├── baseline_BLIP.ipynb            # BLIP model evaluation
│   ├── baseline_BLIP2.ipynb           # BLIP-2 model evaluation
│   └── baseline_ViLT.ipynb            # ViLT model evaluation
├── Dataset/                           # Curated VQA dataset
│   ├── train_combined.csv             # Training dataset (80%, 56,000 entries)
│   └── val_combined.csv               # Validation dataset (20%, 14,000 entries)
├── Finetune_models/                   # Fine-tuning experiments
│   ├── model_v1/                      # BLIP fine-tuning (Version 1, r=8)
│   │   ├── results/                   # Fine-tuning outputs
│   │   │   ├── fine_tuned_blip_vqa_lora/  # Fine-tuned model weights
│   │   │   ├── resultCSV/             # Prediction results
│   │   │   │   └── predictions.csv    # Validation predictions
│   │   │   └── __huggingface_repos__.json  # Hugging Face metadata
│   │   ├── Blip_version1.ipynb        # Fine-tuning notebook
│   │   └── .DS_Store                  # macOS system file
│   ├── model_v2/                      # BLIP fine-tuning (Version 2, r=16)
│   ├── model_v3/                      # BLIP fine-tuning (Version 3, r=32)
│   ├── ViLT_Lora32_finetuning.ipynb   # ViLT fine-tuning (r=32)
│   └── .DS_Store                      # macOS system file
├── sample-submission/                 # Submission files
│   └── IMT2022564_129_575/           # Submission directory
│       ├── inference.py               # Inference script
│       └── requirements.txt           # Dependencies
├── VR_Project_IMT2022_564_575_129/    # Project report directory
├── VR_Project_IMT2022_564_575_129.pdf # Project report PDF
├── curation.ipynb                     # Dataset curation notebook
├── eval.ipynb                         # Evaluation notebook
└── .DS_Store                          # macOS system file
```

### Key File Descriptions
- **Baseline/**: Contains Jupyter notebooks for evaluating pre-trained BLIP, BLIP-2, and ViLT models.
- **Dataset/**: Includes CSV files for training and validation splits of the curated VQA dataset.
- **Finetune_models/**: Houses fine-tuning experiments for BLIP (three versions) and ViLT, with model weights and predictions.
- **sample-submission/**: Scripts and requirements for generating model inferences.
- **curation.ipynb**: Implements dataset creation using Gemini 2.0 API and ABO data.
- **eval.ipynb**: Evaluates model performance using metrics like Accuracy and BERTScore.
- **VR_Project_IMT2022_564_575_129.pdf**: Detailed project report.

---

## 🛠️ Installation

To set up the project locally or on Kaggle, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rish710/VR_Project.git
   cd VR_Project
   ```

2. **Install Dependencies**:
   Install the required packages listed in `sample-submission/IMT2022564_129_575/requirements.txt`:
   ```bash
   pip install -r sample-submission/IMT2022564_129_575/requirements.txt
   ```

   Key dependencies include:
   - `torch` (PyTorch for model training)
   - `transformers` (Hugging Face for BLIP, ViLT)
   - `peft` (LoRA fine-tuning)
   - `pandas`, `numpy` (data processing)
   - `Pillow` (image handling)

3. **Download the ABO Dataset**:
   - Download the `abo-images-small` (3GB) and `abo-listings` (83MB) datasets from the [Amazon Berkeley Objects dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html).
   - Place them in a directory accessible to the notebooks (update paths in `curation.ipynb`).

4. **Set Up Kaggle Environment** (Optional):
   - Import the repository into a Kaggle notebook.
   - Ensure the Kaggle GPU is enabled (dual 16GB NVIDIA GPUs recommended).
   - Upload the ABO dataset to Kaggle's input directory.

---

## 🚀 Usage

### 1. Dataset Curation
Run the `curation.ipynb` notebook to generate the VQA dataset:
- Processes `abo-images-small` and `abo-listings` using Gemini 2.0 API.
- Outputs `train_combined.csv` and `val_combined.csv` in the `Dataset/` folder.

### 2. Baseline Evaluation
Execute the notebooks in the `Baseline/` directory:
- `baseline_BLIP.ipynb`: Evaluates BLIP (46.60% accuracy).
- `baseline_BLIP2.ipynb`: Evaluates BLIP-2 (24.90% accuracy).
- `baseline_ViLT.ipynb`: Evaluates ViLT (26.11% accuracy).

### 3. Fine-Tuning
Run the fine-tuning notebooks in `Finetune_models/`:
- `model_v1/Blip_version1.ipynb`: Fine-tunes BLIP with LoRA (r=8).
- `model_v2/`: Fine-tunes BLIP with LoRA (r=16).
- `model_v3/`: Fine-tunes BLIP with LoRA (r=32, 62.5% accuracy).
- `ViLT_Lora32_finetuning.ipynb`: Fine-tunes ViLT with LoRA (r=32, 54.46% accuracy).

### 4. Evaluation
Use `eval.ipynb` to compute performance metrics (Accuracy, F1, BERTScore, etc.) for baseline and fine-tuned models.

### 5. Inference
Run `sample-submission/IMT2022564_129_575/inference.py` to generate predictions:
```bash
python sample-submission/IMT2022564_129_575/inference.py
```

---

## 📊 Results

### Baseline Performance
| Model  | Test Accuracy (%) | BERTScore (Precision) | ROUGE Score | BERT Cosine Similarity | Levenshtein Distance |
|--------|-------------------|-----------------------|-------------|------------------------|---------------------|
| CLIP   | 2.60              | 0.1364                | 0.0299      | 0.2732                 | 6.16                |
| BLIP   | 46.60             | 0.9143                | 0.4756      | 0.7437                 | 2.73                |
| BLIP-2 | 24.90             | 0.7608                | 0.2587      | 0.5522                 | 4.14                |
| ViLT   | 26.11             | 0.7812                | 0.2734      | 0.5721                 | 3.98                |

### Fine-Tuned Performance
- **BLIP (LoRA, r=32)**:
  - Test Accuracy: 62.5%
  - ROUGE Score: 0.6375
  - BERT Cosine Similarity: 0.8163
- **ViLT (LoRA, r=32)**:
  - Test Accuracy: 54.46%
  - Macro F1 Score: 0.0841
  - BERT F1 Score: 0.9883

---

## 🔍 Challenges

- **Limited GPU Resources**: Kaggle's 16GB GPU constraints required careful optimization (e.g., mixed-precision training, LoRA).
- **Kaggle Instability**: Notebook timeouts necessitated modular code and shorter training loops.
- **LoRA Integration**: Initial compatibility issues with BLIP's architecture were resolved by modifying the forward method.
- **Quantization Issues**: Inability to use `bitsandbytes` for 4-bit/8-bit quantization due to Kaggle installation errors.

---

### Notes
- **Artifact ID**: A new UUID (`f2b7a4d8-9c3e-4f7b-9f8e-3a5b7f1c2d6a`) was generated as this is a new README, distinct from the previous project structure artifact.
- **Formatting**: The README uses Markdown features like badges, emojis, tables, and headings for visual appeal and clarity.
- **Project Structure**: The provided structure was directly incorporated, with descriptions for each directory and key file.
- **Content**: The README summarizes the report’s key points (dataset curation, model evaluation, fine-tuning, results) while keeping it concise and user-friendly.
- **Customization**: If you want to add specific details (e.g., actual email addresses, additional sections, or a logo), let me know, and I can update the artifact with the same `artifact_id` for continuity.
- **Report Integration**: The README reflects the report’s content (e.g., dataset size, model performance, challenges) and aligns with the project’s academic context.

Let me know if you need further refinements or additional sections!
