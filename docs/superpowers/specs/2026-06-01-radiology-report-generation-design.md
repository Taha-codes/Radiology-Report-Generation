# Radiology Report Generation Pipeline — Design Spec
Date: 2026-06-01

## Overview

A baseline vision-language pipeline that takes a chest X-ray image and generates a radiology report. The architecture couples a frozen CheXagent visual encoder with a trainable BioGPT language model via a learned linear projection.

---

## Architecture

### Components

| Module | Class | Trainable |
|---|---|---|
| CheXagent Q-Former encoder | `CheXagentEncoder` | No (frozen) |
| Linear bridge | `LinearProjection` | Yes |
| BioGPT causal LM | `BioGPTGenerator` (via `BioGptForCausalLM`) | Yes |

### Visual Prefix Injection

1. `CheXagentEncoder` loads `StanfordAIMI/CheXagent` via `InstructBlipForConditionalGeneration`. The LM head is discarded after loading to free memory. Runs `vision_model` → Q-Former to produce **32 visual query tokens** of dim 768. All weights frozen.
2. `LinearProjection`: `nn.Linear(768, 1024)` + `nn.LayerNorm(1024)` maps visual space → BioGPT hidden space.
3. `RadiologyReportModel.forward()`: concatenates projected visual prefix `[B, 32, 1024]` with BioGPT text embeddings `[B, T, 1024]`, then calls BioGPT with `inputs_embeds`. Labels are prefixed with 32 × `-100` so CrossEntropyLoss ignores visual positions.
4. `generate()`: visual prefix passed as `inputs_embeds`; if device is MPS, computation falls back to CPU before beam search.

---

## Data Pipeline

### Sources
- `MIMIC-CXR dataset /mimic_cxr_aug_train.csv` — 64,586 rows
- `MIMIC-CXR dataset /mimic_cxr_aug_validate.csv` — 500 rows
- Combined: 65,086 rows → random 80/10/10 split (seed=42)
  - Train: ~52,069 | Val: ~6,509 | Test: ~6,508

### Image Selection (PA-preferential)
Parse `PA` column → if non-empty take first PA path; else first from `AP`; else first from full `image` list. Paths are relative to `MIMIC-CXR dataset /official_data_iccv_final/`.

### MONAI Transform Chain
1. `LoadImage(image_only=True)` — load JPG as numpy
2. `EnsureChannelFirst()` — (C, H, W)
3. `RepeatChannel(3)` — grayscale → 3-channel RGB
4. `Resize((224, 224))` — spatial resize
5. `ScaleIntensity(minv=0.0, maxv=1.0)` — normalize to [0, 1]
6. `Lambda` — CLIP-style normalization (mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276])
7. `ToTensor()` — final torch tensor

### Report Text Selection
`ast.literal_eval(row['text'])` → `next((t for t in texts if t.strip()), None)`. Rows with no valid text are skipped.

### Tokenization
BioGPT tokenizer, `max_length=256`, truncate+pad, `return_tensors='pt'`.

---

## Training (`src/train.py`)

- **Optimizer**: Adam on projection + BioGPT params only, lr=1e-4
- **Scheduler**: `ReduceLROnPlateau(mode='min', patience=2, factor=0.5)` on val loss
- **Gradient clipping**: `clip_grad_norm_(params, max_norm=1.0)`
- **Epochs**: 5
- **Batch size**: 8
- **DataLoader**: `num_workers=0` (macOS), `pin_memory=torch.cuda.is_available()`
- **Checkpoint**: saves `{model_state, optimizer_state, epoch, val_loss}` to `checkpoints/best_model.pt` when val loss improves
- **Logging**: per-epoch train and val loss printed to stdout

---

## Evaluation (`src/evaluate.py`)

- Loads best checkpoint; runs on test set (up to `--max_test_samples 500` by default)
- Inference at batch_size=1 with greedy/beam decode
- **Metrics**:
  - BLEU-1, BLEU-2, BLEU-3, BLEU-4 (nltk, SmoothingFunction.method1)
  - ROUGE-L (rouge-score, mean F1)
  - BERTScore (bert-score, mean F1, `rescale_with_baseline=True`)
  - CIDEr (pycocoevalcap)
- Prints metric table + 5 randomly selected sample rows (reference vs generated, truncated)
- Optionally saves full results to `results.json`

---

## File Structure

```
src/
  dataset.py      — MIMICCXRDataset, MONAI transforms, split logic
  model.py        — CheXagentEncoder, LinearProjection, RadiologyReportModel
  train.py        — training + validation loop, checkpointing
  evaluate.py     — evaluation loop, all metrics, results table
config.py         — paths, hyperparameters
requirements.txt
README.md
checkpoints/      — created at runtime
docs/superpowers/specs/  — this file
```

---

## Dependencies

```
torch, torchvision
monai
transformers
accelerate
nltk
rouge-score
bert-score
pycocoevalcap
numpy
pandas
pillow
```
