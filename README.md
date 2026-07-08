<div align="center">

# 🫁 Pneumonia Detection & Explainable Medical AI Assistant

**A production-oriented deep learning system for chest X-ray pneumonia screening — combining CNN-based classification, Grad-CAM interpretability, and a PDF-grounded Retrieval-Augmented Generation (RAG) pipeline for evidence-based clinical explanations.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#)

[Demo](#-streamlit-application) · [Architecture](#-system-architecture) · [Results](#-model-performance) · [Installation](#️-installation--setup) · [Roadmap](#-roadmap)

</div>

---

## 📌 Overview

Most academic pneumonia-detection projects stop at "train a CNN, report accuracy." This project goes further — it's built as an **end-to-end, explainable, evidence-grounded diagnostic assistant** that mirrors how AI is actually deployed in regulated, high-stakes domains like healthcare.

The system integrates three engineering disciplines that rarely appear together in a single portfolio project:

| Layer | What it demonstrates |
|---|---|
| **Computer Vision** | Transfer learning, class-imbalance handling, calibrated inference |
| **Explainable AI (XAI)** | Grad-CAM saliency mapping for model trust & auditability |
| **Applied NLP / RAG** | Retrieval-grounded LLM outputs — no hallucinated medical claims |

The result is a working prototype of what a **clinical decision-support tool** looks like when accuracy, interpretability, and trustworthy communication are treated as first-class requirements — not afterthoughts.

---

## 🚀 Key Features

- 🩻 **Pneumonia classification** from chest X-rays using a fine-tuned CNN
- 🧠 **MobileNetV2 transfer learning** with a custom classifier head
- 🔥 **Grad-CAM explainability** — visual heatmaps showing *why* the model decided what it decided
- ⚖️ **Class-imbalance-aware training** via weighted loss
- 📚 **PDF-grounded RAG pipeline** — medical explanations retrieved from indexed clinical documents, not generated from parametric memory
- 💬 **Evidence-based chatbot** answering pneumonia-related questions with cited source context
- 🖥️ **Interactive Streamlit dashboard** unifying prediction, explainability, and chat in one interface
- 📄 **Optional clinical-style PDF report generation**
- 🧩 Modular, extensible architecture designed for batch inference and future multimodal expansion

---

## 🏗️ System Architecture

```
                 ┌─────────────────────┐
   Chest X-ray → │   MobileNetV2 CNN    │ → Prediction + Confidence
                 │  (Transfer Learning) │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      Grad-CAM        │ → Attention Heatmap
                 │  Explainability Layer│
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌─────────────────────┐
   Medical PDFs →│   RAG Pipeline       │ → Evidence-Grounded
   (Knowledge   │ (FAISS + Embeddings) │    Medical Explanation
    Base)        └──────────┬───────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Streamlit Dashboard│ → Unified Diagnostic UI
                 └─────────────────────┘
```

**RAG Retrieval Flow:**
`PDF Corpus → Chunking → Sentence-Transformer Embeddings → FAISS Vector Index → Semantic Search → Grounded LLM Response`

---

## 🧰 Tech Stack

| Domain | Technologies |
|---|---|
| **Deep Learning** | PyTorch, TorchVision, MobileNetV2 |
| **Explainability** | Grad-CAM |
| **Retrieval-Augmented Generation** | LangChain, FAISS, Sentence-Transformers |
| **Application Layer** | Streamlit |
| **Data** | Chest X-Ray Pneumonia Dataset (Kaggle) |

---

## 📊 Model Details

| Parameter | Value |
|---|---|
| Backbone | MobileNetV2 |
| Training Strategy | Transfer learning (last block + classifier head fine-tuned) |
| Loss Function | Weighted Cross-Entropy (handles class imbalance) |
| Input Resolution | 224 × 224 |
| Task | Binary classification — `NORMAL` vs. `PNEUMONIA` |

---

## 📈 Model Performance

> *Populate this table with your test-set metrics before publishing — recruiters and hiring managers weight quantified results heavily.*

| Metric | Score |
|---|---|
| Accuracy | `TBD` |
| Precision | `TBD` |
| Recall (Sensitivity) | `TBD` |
| F1-Score | `TBD` |
| AUC-ROC | `TBD` |

---

## 🔍 Explainability Layer

Black-box predictions are not acceptable in medical AI. This project uses **Grad-CAM** to generate class-discriminative localization maps that highlight the lung regions driving each prediction, producing:

- Heatmap overlays on the original X-ray
- Visual confirmation that the model attends to clinically relevant regions (not background artifacts)
- A trust signal that supports responsible-AI and interpretability requirements in imaging pipelines

---

## 📚 PDF-Grounded RAG System

Rather than generating free-form text from an LLM's parametric knowledge, all medical explanations are **retrieved from a curated, indexed set of medical PDFs**, including:

- Pneumonia fact sheets
- Prevention guidelines
- Treatment references
- Antibiotic usage notes

**Pipeline:** `PDF → Text Chunking → Embeddings → FAISS Vector Index → Semantic Search → Grounded Answer`

This design choice directly addresses one of the biggest risks in applied LLM systems — **hallucination** — by constraining the model's outputs to retrieved, verifiable source material.

---

## 🖥️ Streamlit Application

A unified diagnostic dashboard combining every component of the pipeline:

- X-ray upload & viewer
- Grad-CAM overlay visualization
- Prediction label + confidence score
- Risk indicator
- Retrieval-grounded medical explanation
- Sidebar chatbot for interactive Q&A over the medical knowledge base

```bash
streamlit run app.py
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/sreenugopireddy/Pneumonia-detection-Using-CNN.git
cd Pneumonia-detection-Using-CNN
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install torch torchvision streamlit
pip install grad-cam
pip install langchain langchain-community langchain-text-splitters
pip install sentence-transformers faiss-cpu
pip install pypdf
```

---

## ⚠️ Limitations

- Training data is pediatric-sourced — domain shift may affect generalization to adult/external X-rays
- Class imbalance can elevate false-positive rate
- Designed as a **screening aid**, not a diagnostic replacement
- Softmax confidence scores are not calibrated clinical probabilities
- Performance may degrade on X-rays from different acquisition equipment/styles

---

## 🗺️ Roadmap

- [ ] Expand to a larger, demographically balanced dataset
- [ ] Multi-class lung disease detection (beyond binary pneumonia)
- [ ] Native DICOM support
- [ ] Multimodal RAG — joint image + text retrieval
- [ ] Similar-case retrieval / case-based reasoning
- [ ] Automated clinical report export
- [ ] Model calibration curves (reliability diagrams)

---

## 👤 Author

**Sreenu GopiReddy**
AI/ML Engineer — Medical Imaging & Retrieval-Augmented Generation Systems

[GitHub](https://github.com/sreenugopireddy) · [LinkedIn](https://linkedin.com/in/sreenugopireddy)

---

## 📌 Disclaimer

This system is developed strictly for **educational and research purposes**. It is **not a certified medical diagnostic device** and must not be used for real clinical decision-making.

---

<div align="center">

### ⭐ If this project was useful, consider starring the repo!

</div>
