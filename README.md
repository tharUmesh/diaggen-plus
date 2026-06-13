# DiagGen+ 🏥
### Medical QA Diagnostic Assistant

---

> ⚠️ **EDUCATIONAL SIMULATION ONLY.** This tool is not approved for clinical use and does not constitute medical advice. Always consult a licensed physician.

---

## Overview

DiagGen+ is an interactive symptom-checker that reasons through diagnoses the way a doctor does — starting with a classification, then asking targeted follow-up questions until it reaches confident, explainable conclusions. It demonstrates all five required AI techniques in one cohesive system:

| Technique | Implementation |
|-----------|---------------|
| **NLP** | spaCy preprocessing + GloVe/FastText embeddings |
| **Transformers / LLMs** | BioBERT fine-tuning for symptom-to-disease classification |
| **Generative AI** | Conditional VAE for rare-disease training data augmentation |
| **Model Training Approaches** | Transfer learning + few-shot learning (SetFit) |
| **Prompt Engineering** | Chain-of-thought + in-context prompting for QA loop and explanations |

---

## Project Structure

```
diaggen-plus/
├── config/
│   ├── config.yaml          # all tunable parameters (thresholds, model IDs, API settings)
│   └── prompts.yaml         # all LLM prompt templates
├── data/
│   ├── raw/                 # downloaded datasets (not committed to Git)
│   ├── processed/           # cleaned & split data (not committed to Git)
│   ├── synthetic/           # VAE-generated augmentation samples (not committed to Git)
│   └── README.md            # dataset download instructions
├── notebooks/
│   ├── phase0_eda.ipynb              # Phase 0: EDA + preprocessing
│   ├── phase1_bert_finetuning.ipynb  # Phase 1: BERT fine-tuning
│   ├── phase2_vae_augmentation.ipynb # Phase 2: VAE training + augmentation
│   ├── phase3_qa_loop.ipynb          # Phase 3: QA loop development
│   └── evaluation/
│       └── model_comparison.ipynb    # baseline vs augmented comparison charts
├── src/
│   ├── preprocessing/   # cleaner.py, embeddings.py
│   ├── models/          # classifier.py (BERT), vae.py (Conditional VAE)
│   ├── training/        # trainer.py, augment.py
│   ├── qa/              # engine.py (QA loop), llm_client.py, question_gen.py
│   ├── evaluation/      # metrics.py
│   └── utils/           # config_loader.py, logger.py
├── app/
│   ├── main.py          # Streamlit entry point
│   ├── pages/           # diagnosis.py, about.py
│   └── components/      # confidence_bar.py, chat_ui.py
├── models/
│   ├── bert/            # saved BERT checkpoints (not committed to Git)
│   └── vae/             # saved VAE weights (not committed to Git)
├── reports/
│   └── figures/         # charts and plots for the report
├── tests/               # pytest test suite
├── .env.example         # API key template (copy to .env)
├── .gitignore
└── environment.yml      # conda environment definition
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-group/diaggen-plus.git
cd diaggen-plus
```

### 2. Create the conda environment
```bash
conda env create -f environment.yml
conda activate diaggen-plus
```

### 3. Download the spaCy language model
```bash
python -m spacy download en_core_web_sm
```

### 4. Set up API keys
```bash
cp .env.example .env
# Edit .env and add your API key for the active LLM provider
```

### 5. Download the dataset
See [`data/README.md`](data/README.md) for full instructions. Quick start:
```bash
# Option B — Hugging Face (fastest, no Kaggle account needed)
python -c "
from datasets import load_dataset, concatenate_datasets
ds = load_dataset('gretelai/symptom_to_diagnosis') 
combined_ds = concatenate_datasets(list(ds.values()))
combined_ds.to_csv(
    'data/raw/symptom_to_diagnosis.csv', 
    index=False
)
"
```

### 6. Register the kernel for Jupyter
```bash
python -m ipykernel install --user --name diaggen-plus --display-name "diaggen-plus"
jupyter notebook
```

---

## Running the Notebooks (in order)

| Order | Notebook | Phase | Purpose |
|-------|----------|-------|---------|
| 1 | `notebooks/phase0_eda.ipynb` | 0 | EDA, preprocessing, data split |
| 2 | `notebooks/phase1_bert_finetuning.ipynb` | 1 | BERT fine-tuning → Gate 1 MVP |
| 3 | `notebooks/phase2_vae_augmentation.ipynb` | 2 | VAE augmentation → Gate 2 |
| 4 | `notebooks/phase3_qa_loop.ipynb` | 3 | QA loop → Gate 3 (DiagGen+) |
| 5 | `notebooks/evaluation/model_comparison.ipynb` | 4 | Final evaluation charts |

---

## Running the Streamlit App

```bash
streamlit run app/main.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

All parameters are in `config/config.yaml`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `bert.confidence_threshold` | `0.80` | Below this → trigger QA loop |
| `qa.max_turns` | `4` | Max follow-up questions before forced decision |
| `llm.active_provider` | `"anthropic"` | Switch to `"openai"` or `"ollama"` |
| `bert.base_model` | `"dmis-lab/biobert-base-cased-v1.2"` | Swap to `"bert-base-uncased"` if BioBERT is slow |
| `data.rare_class_threshold` | `100` | Classes below this count targeted for VAE augmentation |

---

## Development Phases & Gates

```
Phase 0 (Week 1)  → Dataset + preprocessing pipeline
Phase 1 (Week 2)  → BERT classifier + basic app  ← GATE 1: submittable MVP (3/5 techniques)
Phase 2 (Week 3)  → VAE augmentation + few-shot  ← GATE 2: full DiagGen (5/5 techniques)
Phase 3 (Week 4-5)→ QA loop                      ← GATE 3: full DiagGen+
Phase 4 (Week 5-6)→ Polish + report + video
```

---

## Team

| Member | Responsibility |
|--------|---------------|
| Levindu | NLP Lead — preprocessing, embeddings |
| Kanishka | Model Lead — BERT fine-tuning, evaluation |
| Tharun | Generative Lead — VAE, augmentation |
| Nethangi | QA / Prompt Lead — QA loop, prompt engineering |

---

## Ethical Statement

This project uses only publicly available, non-PHI datasets. Synthetic records generated by the VAE are clearly labelled and used for training augmentation only, never presented as real patient data. All outputs carry a disclaimer that the tool is for educational purposes only.
