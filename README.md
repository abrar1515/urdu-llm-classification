# Performance Comparison of LoRA-Fine-Tuned LLMs for Urdu News Topic Classification

Code, predictions and analysis for the paper *"Performance Comparison of LoRA-Fine-Tuned Large Language Models for Urdu News Topic Classification"* (Muhammad Abrar Saddique). Paper: _[link will be added]_.

Three open-weight LLMs (Qwen2.5-3B, Llama-3.2-3B, Gemma-2-2B) are fine-tuned with LoRA under identical settings for four-class Urdu news topic classification and compared with a TF-IDF + linear SVM baseline, using bootstrap confidence intervals, paired McNemar tests with Holm correction, tokenizer-fertility analysis and a manual audit of shared errors.

## Main results (test set, n = 800)

| Model | Macro-F1 [95% CI] | Training time |
|---|---|---|
| TF-IDF + LinearSVC | 0.935 [0.917, 0.952] | ~4 s (CPU) |
| Qwen2.5-3B (LoRA) | 0.939 [0.921, 0.955] | 85.4 min |
| Llama-3.2-3B (LoRA) | 0.955 [0.940, 0.969] | 80.5 min |
| Gemma-2-2B (LoRA) | **0.960** [0.946, 0.973] | 66.6 min |

After Holm correction Gemma-2-2B is significantly better than Qwen2.5-3B (p = 0.038) and the TF-IDF baseline (p = 0.007), but not significantly different from Llama-3.2-3B. Each configuration was trained once (seed 42) on a single NVIDIA T4 GPU. See the paper for details and limitations.

## Repository structure

```
.
├── src/
│   ├── prepare_inputs.py   builds the text-free inputs (TF-IDF predictions, test labels, audit CSV)
│   └── analysis.py         tables, statistics and figures (PNG + PDF, 600 dpi)
├── notebooks/
│   └── kaggle_train.ipynb  training notebook (Kaggle, outputs cleared)
├── results/                text-free inputs for the analysis (committed)
│   ├── result.json         metrics, per-class reports, training history, tokenizer fertility
│   ├── preds_*.npy         test-set predictions of the three LLMs and of TF-IDF
│   ├── test_labels.csv     gold test labels (no text)
│   └── label_audit.csv     manual audit of the 18 shared errors (no article text)
├── figures/                figures of the paper and extras (PNG + PDF)
├── outputs/                tables/*.csv and summary.json written by analysis.py
└── data/                   NOT committed: train.csv and test.csv with the news text
```

## Reproduce

### 0. Data (not included)
The news text of the **Urdu News Dataset 1M** (Mendeley Data, DOI [10.17632/834vsxnb99.4](https://doi.org/10.17632/834vsxnb99.4)) is not redistributed here: the dataset may only be used for non-commercial research with credit to the news sources. Download it from Mendeley Data and follow the cleaning, sampling and splitting procedure (seed 42) described in Section 3.1 of the paper.

### 1. Training (Kaggle)
Run `notebooks/kaggle_train.ipynb` on Kaggle (GPU T4, Internet on). Put your Hugging Face token in Kaggle Secrets; Llama 3.2 and Gemma 2 are gated, so accept their licenses on Hugging Face first. Set `DRY_RUN = False` for the full experiment (about 4 hours for the three LoRA runs). Download `results.json`, `preds_*.npy`, `train.csv` and `test.csv` from the commit output.

### 2. Analysis (any computer, no GPU)
Copy the Kaggle files `result(s).json` and `preds_*.npy` into `results/` and `train.csv` / `test.csv` into `data/`, then:

```bash
pip install -r requirements.txt

# once: re-creates the TF-IDF predictions and writes text-free label files
python src/prepare_inputs.py --audit-xlsx path/to/label_audit.xlsx   # --audit-xlsx is optional

# tables, statistics and figures
python src/analysis.py
```

`prepare_inputs.py` should report TF-IDF accuracy 0.9350 / macro-F1 0.9351, which confirms the split. `analysis.py` writes the figures to `figures/` (PNG and PDF, 600 dpi by default; `--dpi 300` or `--formats png` to change) and the tables and `summary.json` to `outputs/`.

## Environment
- Training: Kaggle GPU T4, container image `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`, with `transformers==4.56.2`, `peft==0.15.2`, `bitsandbytes==0.45.5`, `accelerate==1.6.0`.
- Analysis: Python 3.10+ with the packages in `requirements.txt`.

## License and citation
Code: MIT License (see `LICENSE`). The dataset and the pretrained models keep their own licenses. If you use this work, please cite the paper (citation will be added after publication).
