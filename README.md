# DSA4213_Fine-tune_BRRT
# Toxic Chat Classification

This project fine-tunes a pre-trained BERT model for toxic chat classification using two approaches: LoRA and full fine-tuning. The project includes hyperparameter tuning using Optuna and uses class weighting to handle the imbalanced dataset.

## Setup and Installation

1.  **Clone the repository (if applicable) or ensure you have the `main.py` file.**

2.  **Install the required dependencies:**

   ### Core Libraries

| Package | Version (Recommended) | Description |
|----------|----------------------|-------------|
| `torch` | >=2.0.0 | Deep learning framework for training models |
| `transformers` | >=4.40.0 | Hugging Face library for pretrained Transformer models |
| `peft` | >=0.7.0 | Parameter-Efficient Fine-Tuning (LoRA and others) |
| `optuna` | >=3.4.0 | Automated hyperparameter tuning |
| `pandas` | >=2.2.0 | Data manipulation and processing |
| `numpy` | >=1.26.0 | Numerical operations |
| `scikit-learn` | >=1.4.0 | Evaluation metrics and preprocessing utilities |
| `wandb` | >=0.17.0 | Experiment tracking and visualization |

---

