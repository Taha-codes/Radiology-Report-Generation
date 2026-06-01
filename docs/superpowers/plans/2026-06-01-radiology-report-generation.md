# Radiology Report Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a baseline vision-language pipeline that takes a chest X-ray and generates a radiology report using a frozen CheXagent visual encoder bridged to BioGPT via a learned linear projection.

**Architecture:** CheXagent's Q-Former produces 32 visual query tokens (dim 768); a LinearProjection maps these to BioGPT's 1024-dim input space; the projected prefix is prepended to text embeddings and BioGPT generates autoregressively with cross-entropy loss masked to text positions only.

**Tech Stack:** PyTorch, MONAI, HuggingFace Transformers (InstructBLIP + BioGPT), nltk, rouge-score, bert-score, pycocoevalcap, pandas, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `config.py` | All paths and hyperparameters |
| `src/__init__.py` | Package marker |
| `src/dataset.py` | MONAI transforms, image/text selection, `MIMICCXRDataset`, `load_datasets` |
| `src/model.py` | `LinearProjection`, `CheXagentEncoder`, `RadiologyReportModel` |
| `src/train.py` | Training + validation loop, checkpointing |
| `src/evaluate.py` | Metric helpers, evaluation loop, results table |
| `tests/test_config.py` | Config sanity checks |
| `tests/test_dataset.py` | Dataset utilities and `__getitem__` shapes |
| `tests/test_model.py` | Model component shapes and forward pass |
| `tests/test_evaluate.py` | Metric helper correctness |
| `requirements.txt` | Pinned dependencies |
| `README.md` | Setup and usage instructions |

---

### Task 1: Project scaffold and config

**Files:**
- Create: `config.py`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `checkpoints/.gitkeep`
- Create: `tests/test_config.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src tests checkpoints
touch src/__init__.py tests/__init__.py checkpoints/.gitkeep
```

- [ ] **Step 2: Write config.py**

```python
# config.py
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "MIMIC-CXR dataset ")
IMAGES_ROOT = os.path.join(DATASET_ROOT, "official_data_iccv_final")
TRAIN_CSV = os.path.join(DATASET_ROOT, "mimic_cxr_aug_train.csv")
VAL_CSV = os.path.join(DATASET_ROOT, "mimic_cxr_aug_validate.csv")

CHEXAGENT_MODEL = "StanfordAIMI/CheXagent"
BIOGPT_MODEL = "microsoft/biogpt"

BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
MAX_TEXT_LEN = 256
NUM_VISUAL_TOKENS = 32
VISUAL_HIDDEN_DIM = 768
BIOGPT_HIDDEN_DIM = 1024
GRAD_CLIP_NORM = 1.0
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
SEED = 42
```

- [ ] **Step 3: Write tests/test_config.py**

```python
# tests/test_config.py
import config


def test_paths_resolve():
    assert config.PROJECT_ROOT.endswith("Radiology-Report-Generation")
    assert config.DATASET_ROOT.endswith("MIMIC-CXR dataset ")


def test_hyperparameters():
    assert config.NUM_VISUAL_TOKENS == 32
    assert config.VISUAL_HIDDEN_DIM == 768
    assert config.BIOGPT_HIDDEN_DIM == 1024
    assert config.GRAD_CLIP_NORM == 1.0
    assert config.SCHEDULER_PATIENCE == 2
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_config.py -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add config.py src/__init__.py tests/__init__.py tests/test_config.py checkpoints/.gitkeep
git commit -m "feat: add project scaffold and config"
```

---

### Task 2: Dataset utilities and MONAI transforms

**Files:**
- Create: `src/dataset.py` (parse/select helpers + transforms only — no Dataset class yet)
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dataset.py
import numpy as np
import torch
import tempfile
import os
from PIL import Image


def test_parse_list_valid():
    from src.dataset import _parse_list
    assert _parse_list("['a.jpg', 'b.jpg']") == ['a.jpg', 'b.jpg']


def test_parse_list_empty_and_bad():
    from src.dataset import _parse_list
    assert _parse_list("[]") == []
    assert _parse_list("not a list") == []


def test_select_image_pa_preferred():
    from src.dataset import _select_image_path
    row = {
        "PA": "['files/p10/img_pa.jpg']",
        "AP": "['files/p10/img_ap.jpg']",
        "image": "['files/p10/img_pa.jpg', 'files/p10/img_ap.jpg']",
    }
    assert _select_image_path(row) == 'files/p10/img_pa.jpg'


def test_select_image_ap_fallback():
    from src.dataset import _select_image_path
    row = {"PA": "[]", "AP": "['files/p10/img_ap.jpg']", "image": "['files/p10/img_ap.jpg']"}
    assert _select_image_path(row) == 'files/p10/img_ap.jpg'


def test_select_image_fallback_to_any():
    from src.dataset import _select_image_path
    row = {"PA": "[]", "AP": "[]", "image": "['files/p10/img_lat.jpg']"}
    assert _select_image_path(row) == 'files/p10/img_lat.jpg'


def test_select_image_all_empty_returns_none():
    from src.dataset import _select_image_path
    row = {"PA": "[]", "AP": "[]", "image": "[]"}
    assert _select_image_path(row) is None


def test_select_report_first_non_empty():
    from src.dataset import _select_report_text
    row = {"text": "['', '  ', 'Findings: Normal chest.']"}
    assert _select_report_text(row) == 'Findings: Normal chest.'


def test_select_report_all_empty_returns_none():
    from src.dataset import _select_report_text
    row = {"text": "['', '   ']"}
    assert _select_report_text(row) is None


def test_to_rgb_grayscale():
    from src.dataset import _to_rgb
    x = np.zeros((1, 224, 224), dtype=np.float32)
    assert _to_rgb(x).shape == (3, 224, 224)


def test_to_rgb_already_rgb():
    from src.dataset import _to_rgb
    x = np.zeros((3, 224, 224), dtype=np.float32)
    assert _to_rgb(x).shape == (3, 224, 224)


def test_clip_normalize_changes_values():
    from src.dataset import _clip_normalize
    x = torch.ones(3, 224, 224)
    out = _clip_normalize(x)
    assert out.shape == (3, 224, 224)
    assert not torch.allclose(out, x)


def test_build_transforms_output_shape():
    from src.dataset import build_transforms
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(tmp.name)
    tmp.close()
    result = build_transforms()(tmp.name)
    os.unlink(tmp.name)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_dataset.py -v
```
Expected: ImportError (src/dataset.py does not exist)

- [ ] **Step 3: Implement src/dataset.py (utilities + transforms)**

```python
# src/dataset.py
import ast
import os
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
from monai.transforms import (
    Compose, EnsureChannelFirst, Lambda, LoadImage,
    Resize, ScaleIntensity, ToTensor,
)
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import config

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _parse_list(s: str) -> list:
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def _select_image_path(row) -> Optional[str]:
    pa = _parse_list(row.get("PA", "[]"))
    if pa:
        return pa[0]
    ap = _parse_list(row.get("AP", "[]"))
    if ap:
        return ap[0]
    imgs = _parse_list(row.get("image", "[]"))
    return imgs[0] if imgs else None


def _select_report_text(row) -> Optional[str]:
    texts = _parse_list(row.get("text", "[]"))
    return next((t for t in texts if t.strip()), None)


def _to_rgb(x: np.ndarray) -> np.ndarray:
    """Ensure 3 channels; CXRs often load as single-channel grayscale."""
    if x.shape[0] == 1:
        return np.repeat(x, 3, axis=0)
    return x[:3]


def _clip_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CLIP_MEAN, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=x.dtype).view(3, 1, 1)
    return (x - mean) / std


def build_transforms() -> Compose:
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Lambda(_to_rgb),
        Resize((224, 224)),
        ScaleIntensity(minv=0.0, maxv=1.0),
        ToTensor(),
        Lambda(_clip_normalize),
    ])
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_dataset.py -v
```
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: add dataset utilities and MONAI transforms"
```

---

### Task 3: MIMICCXRDataset class and 80/10/10 split

**Files:**
- Modify: `src/dataset.py` — append `MIMICCXRDataset` class and `load_datasets` function
- Modify: `tests/test_dataset.py` — append `__getitem__` shape test

- [ ] **Step 1: Write failing test**

Append to `tests/test_dataset.py`:

```python
def test_dataset_getitem_shapes(tmp_path):
    from src.dataset import MIMICCXRDataset, build_transforms
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    img_path = str(tmp_path / "test.jpg")
    Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(img_path)

    records = [{"img_path": img_path, "text": "Findings: No acute process."}]
    ds = MIMICCXRDataset(records, tokenizer, build_transforms())

    assert len(ds) == 1
    item = ds[0]
    assert item["pixel_values"].shape == (3, 224, 224)
    assert item["input_ids"].shape == (256,)
    assert item["attention_mask"].shape == (256,)
    assert item["labels"].shape == (256,)
    pad_mask = item["attention_mask"] == 0
    assert (item["labels"][pad_mask] == -100).all()
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_dataset.py::test_dataset_getitem_shapes -v
```
Expected: AttributeError (MIMICCXRDataset not defined)

- [ ] **Step 3: Append MIMICCXRDataset and load_datasets to src/dataset.py**

```python
class MIMICCXRDataset(Dataset):
    def __init__(self, records: list, tokenizer, transforms: Compose):
        self.records = records
        self.tokenizer = tokenizer
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        pixel_values = self.transforms(record["img_path"])

        encoding = self.tokenizer(
            record["text"],
            max_length=config.MAX_TEXT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_datasets(tokenizer) -> tuple:
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    df = pd.concat([train_df, val_df], ignore_index=True)

    records = []
    for _, row in df.iterrows():
        rel_path = _select_image_path(row)
        text = _select_report_text(row)
        if rel_path is None or text is None:
            continue
        abs_path = os.path.join(config.IMAGES_ROOT, rel_path)
        if not os.path.exists(abs_path):
            continue
        records.append({"img_path": abs_path, "text": text})

    random.seed(config.SEED)
    random.shuffle(records)
    n = len(records)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    transforms = build_transforms()
    return (
        MIMICCXRDataset(records[:train_end], tokenizer, transforms),
        MIMICCXRDataset(records[train_end:val_end], tokenizer, transforms),
        MIMICCXRDataset(records[val_end:], tokenizer, transforms),
    )
```

- [ ] **Step 4: Run all dataset tests**

```bash
pytest tests/test_dataset.py -v
```
Expected: All PASSED (BioGPT tokenizer downloads ~1 MB on first run)

- [ ] **Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: add MIMICCXRDataset and 80/10/10 split logic"
```

---

### Task 4: LinearProjection

**Files:**
- Create: `src/model.py` (projection only)
- Create: `tests/test_model.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model.py
import torch


def test_linear_projection_shape():
    from src.model import LinearProjection
    proj = LinearProjection(in_dim=768, out_dim=1024)
    x = torch.randn(2, 32, 768)
    out = proj(x)
    assert out.shape == (2, 32, 1024)


def test_linear_projection_trainable():
    from src.model import LinearProjection
    proj = LinearProjection()
    assert all(p.requires_grad for p in proj.parameters())
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_model.py -v
```
Expected: ImportError

- [ ] **Step 3: Write src/model.py with LinearProjection**

```python
# src/model.py
import torch
import torch.nn as nn

import config


class LinearProjection(nn.Module):
    def __init__(
        self,
        in_dim: int = config.VISUAL_HIDDEN_DIM,
        out_dim: int = config.BIOGPT_HIDDEN_DIM,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_model.py -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: add LinearProjection module"
```

---

### Task 5: CheXagentEncoder and RadiologyReportModel

**Files:**
- Modify: `src/model.py` — append `CheXagentEncoder` and `RadiologyReportModel`
- Modify: `tests/test_model.py` — append forward and generate tests using mocked encoder

- [ ] **Step 1: Write failing tests**

Append to `tests/test_model.py`:

```python
from unittest.mock import MagicMock


def _make_model_with_mock_encoder(B: int):
    """Build RadiologyReportModel with encoder replaced by a mock returning [B, 32, 768]."""
    from src.model import LinearProjection, RadiologyReportModel
    from transformers import BioGptForCausalLM

    mock_vis = torch.randn(B, 32, 768)
    model = RadiologyReportModel.__new__(RadiologyReportModel)
    # Call nn.Module.__init__ so self.parameters() works
    torch.nn.Module.__init__(model)
    model.encoder = MagicMock(return_value=mock_vis)
    model.projection = LinearProjection()
    model.biogpt = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    return model


def test_radiology_model_forward_returns_scalar_loss():
    B, T = 2, 64
    model = _make_model_with_mock_encoder(B)

    input_ids = torch.randint(0, 42000, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    labels = input_ids.clone()

    loss = model.forward(
        pixel_values=torch.randn(B, 3, 224, 224),
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    assert loss.ndim == 0
    assert loss.item() > 0


def test_radiology_model_generate_returns_token_ids():
    B = 1
    model = _make_model_with_mock_encoder(B)

    output_ids = model.generate(
        pixel_values=torch.randn(B, 3, 224, 224),
        max_new_tokens=10,
        num_beams=1,
    )
    assert output_ids.shape[0] == B
    assert output_ids.shape[1] > 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_model.py::test_radiology_model_forward_returns_scalar_loss tests/test_model.py::test_radiology_model_generate_returns_token_ids -v
```
Expected: AttributeError (RadiologyReportModel not defined)

- [ ] **Step 3: Append CheXagentEncoder and RadiologyReportModel to src/model.py**

```python
from transformers import BioGptForCausalLM, InstructBlipForConditionalGeneration


class CheXagentEncoder(nn.Module):
    def __init__(self, model_name: str = config.CHEXAGENT_MODEL):
        super().__init__()
        full = InstructBlipForConditionalGeneration.from_pretrained(model_name)
        self.vision_model = full.vision_model
        self.qformer = full.qformer
        self.query_tokens = full.query_tokens  # nn.Parameter [1, 32, 768]
        del full
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.shape[0]
        image_embeds = self.vision_model(pixel_values=pixel_values).last_hidden_state
        image_attn = torch.ones(
            image_embeds.shape[:2], dtype=torch.long, device=pixel_values.device
        )
        query_tokens = self.query_tokens.expand(B, -1, -1)
        qformer_out = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attn,
            return_dict=True,
        )
        # Slice to confirmed 32 query token positions (matches num_query_tokens=32 in config)
        return qformer_out.last_hidden_state[:, : config.NUM_VISUAL_TOKENS, :]


class RadiologyReportModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = config.CHEXAGENT_MODEL,
        generator_name: str = config.BIOGPT_MODEL,
    ):
        super().__init__()
        self.encoder = CheXagentEncoder(encoder_name)
        self.projection = LinearProjection()
        self.biogpt = BioGptForCausalLM.from_pretrained(generator_name)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        B = pixel_values.shape[0]
        vis = self.projection(self.encoder(pixel_values))           # [B, 32, 1024]
        txt = self.biogpt.get_input_embeddings()(input_ids)         # [B, T, 1024]
        inputs_embeds = torch.cat([vis, txt], dim=1)                # [B, 32+T, 1024]

        prefix_labels = torch.full(
            (B, config.NUM_VISUAL_TOKENS), -100,
            dtype=labels.dtype, device=labels.device,
        )
        full_labels = torch.cat([prefix_labels, labels], dim=1)

        vis_attn = torch.ones(
            B, config.NUM_VISUAL_TOKENS,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        full_attn = torch.cat([vis_attn, attention_mask], dim=1)

        return self.biogpt(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn,
            labels=full_labels,
        ).loss

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ) -> torch.Tensor:
        orig_device = next(self.parameters()).device
        # MPS does not support all beam-search ops; fall back to CPU
        gen_device = torch.device("cpu") if orig_device.type == "mps" else orig_device

        if gen_device != orig_device:
            self.to(gen_device)

        pv = pixel_values.to(gen_device)
        vis = self.projection(self.encoder(pv))                     # [B, 32, 1024]
        vis_attn = torch.ones(vis.shape[:2], dtype=torch.long, device=gen_device)

        output_ids = self.biogpt.generate(
            inputs_embeds=vis,
            attention_mask=vis_attn,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

        if gen_device != orig_device:
            self.to(orig_device)

        return output_ids
```

- [ ] **Step 4: Run all model tests**

```bash
pytest tests/test_model.py -v
```
Expected: All PASSED (BioGPT ~1.4 GB downloads on first run; CheXagent not loaded in tests)

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: add CheXagentEncoder and RadiologyReportModel"
```

---

### Task 6: Training loop

**Files:**
- Create: `src/train.py`

- [ ] **Step 1: Write src/train.py**

```python
# src/train.py
import argparse
import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
from src.dataset import load_datasets
from src.model import RadiologyReportModel


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.BIOGPT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds, _ = load_datasets(tokenizer)
    pin_memory = torch.cuda.is_available()  # False for MPS and CPU

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=pin_memory,
    )

    model = RadiologyReportModel().to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.SCHEDULER_PATIENCE, factor=config.SCHEDULER_FACTOR,
    )

    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'LR':>10}")
    print("-" * 46)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(pixel_values, input_ids, attention_mask, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.GRAD_CLIP_NORM)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                val_loss += model(pixel_values, input_ids, attention_mask, labels).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | {current_lr:>10.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                os.path.join(config.CHECKPOINT_DIR, "best_model.pt"),
            )
            print(f"  -> Checkpoint saved (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train radiology report generation model")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    train(parser.parse_args())
```

- [ ] **Step 2: Verify import**

```bash
python -c "from src.train import train; print('OK')"
```
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: add training loop with Adam, gradient clipping, ReduceLROnPlateau"
```

---

### Task 7: Evaluation script

**Files:**
- Create: `src/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluate.py


def test_bleu_perfect_score():
    from src.evaluate import compute_bleu
    hyps = ["the cat sat on the mat"]
    refs = ["the cat sat on the mat"]
    scores = compute_bleu(hyps, refs)
    assert scores["bleu1"] > 0.99
    assert scores["bleu4"] > 0.99


def test_bleu_zero_overlap():
    from src.evaluate import compute_bleu
    hyps = ["hello world foo"]
    refs = ["cat dog bird"]
    scores = compute_bleu(hyps, refs)
    assert scores["bleu1"] == 0.0


def test_rouge_l_perfect():
    from src.evaluate import compute_rouge_l
    hyps = ["the patient has pneumonia"]
    refs = ["the patient has pneumonia"]
    assert compute_rouge_l(hyps, refs) > 0.99


def test_cider_nonzero_for_matching():
    from src.evaluate import compute_cider
    hyps = ["the cat sat on the mat"] * 5
    refs = ["the cat sat on the mat"] * 5
    assert compute_cider(hyps, refs) > 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_evaluate.py -v
```
Expected: ImportError

- [ ] **Step 3: Write src/evaluate.py**

```python
# src/evaluate.py
import argparse
import json
import random

import torch
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer as rouge_lib
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
from src.dataset import load_datasets
from src.model import RadiologyReportModel


def compute_bleu(hypotheses: list, references: list) -> dict:
    sf = SmoothingFunction().method1
    tok_refs = [[r.split()] for r in references]
    tok_hyps = [h.split() for h in hypotheses]
    return {
        "bleu1": corpus_bleu(tok_refs, tok_hyps, weights=(1, 0, 0, 0), smoothing_function=sf),
        "bleu2": corpus_bleu(tok_refs, tok_hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=sf),
        "bleu3": corpus_bleu(tok_refs, tok_hyps, weights=(1/3, 1/3, 1/3, 0), smoothing_function=sf),
        "bleu4": corpus_bleu(tok_refs, tok_hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf),
    }


def compute_rouge_l(hypotheses: list, references: list) -> float:
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(references, hypotheses)]
    return sum(scores) / len(scores)


def compute_cider(hypotheses: list, references: list) -> float:
    refs_dict = {i: [r] for i, r in enumerate(references)}
    hyps_dict = {i: [h] for i, h in enumerate(hypotheses)}
    score, _ = Cider().compute_score(refs_dict, hyps_dict)
    return float(score)


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.BIOGPT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, _, test_ds = load_datasets(tokenizer)

    if args.max_test_samples and len(test_ds) > args.max_test_samples:
        indices = random.sample(range(len(test_ds)), args.max_test_samples)
        test_ds.records = [test_ds.records[i] for i in indices]

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = RadiologyReportModel()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    hypotheses, references = [], []
    for batch in test_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]

        output_ids = model.generate(pixel_values, max_new_tokens=256, num_beams=4)
        hyp = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        ref_ids = labels[0]
        ref_ids = ref_ids[ref_ids != -100]
        ref = tokenizer.decode(ref_ids, skip_special_tokens=True)

        hypotheses.append(hyp)
        references.append(ref)

    bleu = compute_bleu(hypotheses, references)
    rouge_l = compute_rouge_l(hypotheses, references)
    _, _, bert_f1 = bert_score_fn(
        hypotheses, references, lang="en", rescale_with_baseline=True, verbose=False
    )
    bertscore = bert_f1.mean().item()
    cider = compute_cider(hypotheses, references)

    results = {**bleu, "rouge_l": rouge_l, "bertscore_f1": bertscore, "cider": cider}

    print(f"\n{'Metric':<14} | {'Score':>8}")
    print("-" * 27)
    for name, val in results.items():
        print(f"{name:<14} | {val:>8.4f}")

    sample_indices = random.sample(range(len(hypotheses)), min(5, len(hypotheses)))
    print("\n--- Sample Outputs (5 random) ---")
    for i in sample_indices:
        print(f"\n[{i}] Reference : {references[i][:150]}...")
        print(f"[{i}] Generated : {hypotheses[i][:150]}...")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate radiology report generation model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--max_test_samples", type=int, default=500)
    parser.add_argument("--output", type=str, default=None, help="Path to save results.json")
    evaluate(parser.parse_args())
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_evaluate.py -v
```
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluation script with BLEU, ROUGE-L, BERTScore, CIDEr"
```

---

### Task 8: requirements.txt and README.md

**Files:**
- Create: `requirements.txt`
- Create: `README.md`

- [ ] **Step 1: Write requirements.txt**

```
torch>=2.0.0
torchvision>=0.15.0
monai>=1.3.0
transformers>=4.35.0
accelerate>=0.24.0
nltk>=3.8.1
rouge-score>=0.1.2
bert-score>=0.3.13
pycocoevalcap>=1.2
numpy>=1.24.0
pandas>=2.0.0
pillow>=10.0.0
pytest>=7.0.0
```

- [ ] **Step 2: Write README.md**

```markdown
# Radiology Report Generation

A baseline vision-language pipeline for automated chest X-ray report generation. CheXagent (frozen) extracts 32 visual query tokens; a linear projection maps them to BioGPT's embedding space; BioGPT generates the report autoregressively.

## Setup

**Prerequisites:** Python 3.10+. On macOS, run `xcode-select --install` before installing (required for pycocoevalcap compilation).

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

**Dataset layout** (relative to project root):
```
MIMIC-CXR dataset /
  mimic_cxr_aug_train.csv
  mimic_cxr_aug_validate.csv
  official_data_iccv_final/
    files/
      p10/  p11/  ...
```

## Usage

### Train
```bash
python -m src.train
python -m src.train --epochs 5 --batch_size 8 --lr 1e-4
python -m src.train --resume checkpoints/best_model.pt   # resume
```

Training logs one line per epoch:
```
Epoch | Train Loss |  Val Loss |         LR
    1 |     3.2541 |    3.1892 |   1.00e-04
```
Best checkpoint saved to `checkpoints/best_model.pt`.

### Evaluate
```bash
python -m src.evaluate --checkpoint checkpoints/best_model.pt
python -m src.evaluate --checkpoint checkpoints/best_model.pt --max_test_samples 500 --output results.json
```

### Run tests
```bash
pytest tests/ -v
```

## Architecture

| Component | Source | Status |
|---|---|---|
| Visual encoder | StanfordAIMI/CheXagent (Q-Former, 32 × 768) | Frozen |
| Projection | Linear(768→1024) + LayerNorm | Trained |
| Language model | microsoft/biogpt (1024-dim, 24 layers) | Trained |

**Image preprocessing:** MONAI pipeline — load JPEG → ensure 3 channels → resize 224×224 → scale [0,1] → CLIP normalization.

**Image selection:** PA view preferred; AP fallback; first available if neither.

**Training:** Adam lr=1e-4, gradient clipping (max_norm=1.0), ReduceLROnPlateau (patience=2), 5 epochs, batch 8.

## Metrics

BLEU-1/2/3/4 (nltk) · ROUGE-L (rouge-score) · BERTScore (bert-score) · CIDEr (pycocoevalcap)
```

- [ ] **Step 3: Install and smoke-test**

```bash
pip install -r requirements.txt
python -c "import torch, monai, transformers, nltk, rouge_score, bert_score; print('all imports OK')"
```
Expected: `all imports OK`

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v
```
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add requirements.txt README.md
git commit -m "feat: add requirements.txt and README"
```
