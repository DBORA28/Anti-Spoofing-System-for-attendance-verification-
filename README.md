# Anti-Spoofing System for Attendance Verification

## Use case

Field and site employees mark attendance by taking a **check-in / check-out photo** on a mobile app. A common fraud is a **replay attack**: a colleague holds up a **phone or screen showing the absent employee's face** to the camera, so the punch looks genuine.

This project builds a model and an **agent** that look at every attendance photo and decide:

| Decision | Meaning | Who acts |
|---|---|---|
| `AUTO_CLEAR` | Confidently a real person in front of the camera | Nobody — attendance accepted |
| `FLAG` | Confidently a phone/screen replay | A reviewer confirms before any action |
| `REVIEW` | Uncertain, odd image, or models disagree | A reviewer decides |

The system **never rejects an employee on its own**: a human confirms every suspected spoof. The goals are (1) catch replays, (2) avoid falsely accusing real employees, and (3) keep the human review queue small.

---

## The journey at a glance

We started from the **MiniFASNetV2** face anti-spoofing architecture, trained our own base model on the first spoof/real dataset, and then fine-tuned it in **two phases** on client data, switching from face crops to full frames along the way. The final model is wrapped in a human-in-the-loop agent.

```
STAGE 0  Base model ───────────────────────────────────────────────────────────────
         MiniFASNetV2 architecture (minivision) → trained from scratch at 224×224
         on the first real/spoof FACE dataset                → checkpoints/best_model.pth
                │
PHASE 1  First client dataset (old sites)
         ├── 1A  Face-crop fine-tune (BN frozen, 30% replay, group split)
         │       → best_model_ft.pth   test AUC 0.951
         └── 1B  Full-frame model: base backbone + NEW FFT moiré branch
                 (conv1–5 frozen, only conv6–7 + new parts trained)
                 → A_baseline_s42/best.pth   test AUC 0.961   ◄── chosen route
                │
PHASE 2  New dataset (new, unseen sites)
         ├── 2A  Face-crop fine-tune → unseen-site AUC 0.728, misses ~half of spoofs ✗ dropped
         └── 2B  Full-frame fine-tune from A_baseline, 50% replay of Phase 1 data
                 variants: ft_new (no replay) · ft_v2 (under-trained) · ft_v3 ✓
                 → ft_v3/best.pth   old sites AUC 0.981 · new sites AUC 0.935   ◄── FINAL MODEL
                │
STAGE 3  Agent: ft_v3 (+ optional ft_new second opinion) + decision bands + human review
         → real people wrongly flagged 20.8% → 1.7% on unseen sites
```

Side study: a **hand-crafted cue detector** (FFT/moiré, colour, glare, sharpness → gradient boosting) was built to check which visual cues actually carry the signal.

---

## Stage 0 — Base model: MiniFASNetV2 trained from scratch

**Notebook:** `notebooks/Untitled14.ipynb` (rename → `01_base_minifasnet_training.ipynb`)

| | |
|---|---|
| **Goal** | A face anti-spoofing backbone that later phases can fine-tune |
| **Architecture** | `MiniFASNetV2_Fixed` — 7 depthwise-separable conv blocks → avg-pool → FC 256→128→2 (≈149K params), adapted from [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) |
| **Starting weights** | **Random initialisation.** Minivision's pretrained 80×80 model was loaded and tested (cells 11–13) but its weights were **not** carried over — our model runs at 224×224 |
| **Data** | First real/spoof face dataset (`MyDrive/real/`, `MyDrive/spoof/`) |
| **Training** | Adam, lr 1e-4, weight decay 5e-4, ReduceLROnPlateau (factor 0.5, patience 3), early stopping, batch 8, augmentation in `ImagePreprocessor` (cell 28) |
| **Output** | `MyDrive/checkpoints/best_model.pth` |
| **Result** | ~85% accuracy on its own validation split; on client face validation **75.6% balanced accuracy** (real 89.0%, spoof only 62.1%) |
| **Lesson** | Good at recognising real faces, weak at catching client-style spoofs → needs fine-tuning on client data |

---

## Phase 1 — Fine-tuning on the first client dataset (old sites)

Data: client attendance photos from the first batch of sites — about **203 spoof** and **~500 real** images, each available as a **face crop** and a **full frame**. Splits are **grouped by `log_id`** so the same person/session never appears in both train and test.

### Phase 1A — Face-crop fine-tune

**Notebook:** `notebooks/MiniFASNetV2_FineTune_Pipeline.ipynb` (rename → `02_phase1_face_finetune.ipynb`)

| | |
|---|---|
| **Starting weights** | Stage 0 `best_model.pth` (loaded with strict key matching + smoke test) |
| **What is trained** | All conv layers, with **BatchNorm statistics frozen** (tiny batches would corrupt them) |
| **Techniques** | AdamW lr **1e-5**; **30% replay** of the original dataset (prevents forgetting); weighted sampler for class imbalance; early stopping on new-val balanced accuracy; **forgetting guard** on old-val; atomic, resumable checkpoints |
| **Evaluation** | Before vs after on an untouched test set, Wilson 95% CIs, APCER-targeted threshold sweep, misclassification CSV, reload-and-verify in a fresh model |
| **Output** | `MyDrive/finetune_runs/run_<stamp>/best_model_ft.pth` |
| **Result** | Val accuracy 91.4%; **test AUC 0.951**, balanced accuracy 88.0%, APCER 6.9% at threshold 0.5 |
| **Lesson** | Big jump from the base model — but a face crop **cuts away the phone bezel, fingers and screen edges**, the strongest evidence of a replay |

> The same notebook also contains an **experimental 2-phase LoRA pipeline** (cells 53–64, `L0`–`L8`). Only L0–L6a were run; it is **not** used in the final system.

### Phase 1B — Full-frame model with a moiré (FFT) branch ✓ chosen route

**Notebook:** `notebooks/Untitled19.ipynb`, cells 0–21 (rename → `04_fullframe_moire_phase1_phase2.ipynb`)

| | |
|---|---|
| **Why** | Use the whole photo so the model can see the phone, and add a branch that detects **moiré** — the interference pattern a camera picks up when photographing a screen |
| **Architecture** | `FullFrameMoireNet` (≈296K params): **RGB branch** = Stage 0 MiniFASNetV2 on the full frame letterboxed to 320×320 → 256 features; **moiré branch** = new 4-layer CNN on log-FFT spectra of 256-px grayscale patches (3×3 grid at eval, 4 random patches in training, resized to 128) → 128 features; **head** = Linear(384→128) → 2 |
| **Starting weights** | RGB branch loaded **strictly** from Stage 0 `best_model.pth`; moiré branch and head start fresh |
| **What is trained** | conv1–conv5 **frozen** (weights + BatchNorm); conv6–conv7, moiré branch and head trained (≈262K trainable params). Old FC layers unused |
| **Techniques** | Leakage audit + split decision before training; discriminative learning rates (head > conv6–7); APCER-capped threshold on validation; controlled runs `A` (baseline recipe) vs `B` (one change); moiré-ablation check |
| **Data** | Phase 1 full frames (`spoof_dataset/full`, corrected labels). Test: 55 real / 41 spoof |
| **Output** | `MyDrive/checkpoints_fullframe_moire/run_20261026_061459/A_baseline_s42/best.pth` (threshold 0.420) |
| **Result** | Val AUC 0.938; **test AUC 0.961** |
| **Lesson** | Full frames beat face crops on the same data → used as the starting point for Phase 2 |

---

## Phase 2 — Fine-tuning on a new dataset from unseen sites

Data: `spoof_dataset new` — **320 spoof + 1,000 real** photos from **new sites**, **split by site 70/15/15** so test sites are never seen in training (test: **173 real / 50 spoof**). Before fine-tuning, the Phase 1 model was evaluated on these sites to measure the drop (`P2-3`).

### Phase 2A — Face-crop route ✗ dropped

| | |
|---|---|
| **Starting weights** | Phase 1A `best_model_ft.pth` (+ auxiliary loss), `phase2b_face_aux_.../best_model_p2b.pth` |
| **Result** | Unseen-site **test AUC 0.728**, APCER 43–49% (misses roughly half the spoofs) |
| **Lesson** | Face crops do not generalise to new sites → face route abandoned |

### Phase 2B — Full-frame fine-tune ✓ final model `ft_v3`

**Notebook:** `notebooks/Untitled19.ipynb`, cells 22–29 (`P2-1` … `P2-4 v3`). Run the **SETUP** cell (26) first in any new session.

| | |
|---|---|
| **Starting weights** | Phase 1B `A_baseline_s42/best.pth` |
| **What is trained** | Same as Phase 1B: conv1–5 frozen (BN in eval mode), conv6–7 + moiré branch + head trainable |
| **Techniques** | **50% replay** of Phase 1 data (keeps old-site skill); learning rates head **3e-4**, conv6–7 **1e-4**; **EMA** weights (decay 0.98); **hard negatives ×1.5**; patience 12; model selection = lowest **new-site val loss** while **old-site val AUC** stays within a tolerance of its start; clean (non-augmented) train check to separate underfitting from poor generalisation |
| **Output** | `.../phase2_from_A_baseline_s42/ft_v3_from_A_s42_replay0.5_real1000/best.pth` — keys `state_dict`, `threshold` **0.2283**, `epoch` **19**, `ema`, `config` {GLOBAL_SIZE 320, PATCH 256, SPEC 128, GRID 3} |

**Variants trained to justify each choice:**

| Variant | Changed | Old-site val AUC | New-site val AUC | Verdict |
|---|---|---|---|---|
| `ft_new` | **no replay** (0.0) | 0.918 | 0.916 | Forgot old sites → replay is needed |
| `ft_v2` | replay 0.5, head LR too low | 0.937 | **0.943** | Stopped improving at epoch 4 (under-trained) |
| **`ft_v3`** | replay 0.5 + restored LRs + EMA + hard negatives | **0.958** | 0.940 | **Best balance → final** |

**Test results (never-seen images):**

| Model | Old sites (55 real / 41 spoof) | New sites (173 real / 50 spoof) |
|---|---|---|
| Phase 1B `A_baseline` | AUC 0.961 | AUC 0.928 · APCER 8.0% · BPCER 27.2% |
| **Phase 2B `ft_v3`** | **AUC 0.981 · APCER 2.4% · BPCER 5.5%** | **AUC 0.935 · APCER 8.0% · BPCER 20.8%** |

Phase 2 **improved the old sites** (0.961 → 0.981) instead of forgetting them, and on new sites caught the same share of spoofs while **wrongly rejecting 11 fewer real people**.

---

## Side study — hand-crafted cue detector

**Notebook:** `notebooks/Moire_Spoof_Detector.ipynb` (rename → `03_handcrafted_cue_detector.ipynb`)

Extracts interpretable cues at native resolution (FFT/moiré energy, colour, glare, sharpness, face count) and trains gradient-boosting classifiers, with duplicate/leakage checks, a metadata-only "shortcut" gate, validation-only threshold, and ablations. Output: `MyDrive/moire_runs/run_<stamp>/detector.pkl`.

**Findings:** test AUC 0.984 on its own split, but 41 near-duplicates between train and test inflate it; metadata alone reaches AUC 0.74 (shortcut risk); the ablation shows **colour/glare** drive the score and moiré features add little. Kept as an analysis tool, not deployed. Cells F1–F3 contain an extra face-crop fine-tune.

---

## Stage 3 — Agent around the final model

The model only outputs **P(spoof)**. The agent turns that score into an action and keeps humans in control:

```
observe → ft_v3 score (+ ft_new second opinion, optional) + image checks (size, backlight %)
decide  → bands chosen on VALIDATION:  p ≤ LOW → AUTO_CLEAR | p ≥ HIGH → FLAG | else REVIEW
          (small image / backlit / models disagree / tool error → always REVIEW)
act     → FLAG + REVIEW + a random 15% audit of AUTO_CLEARs go to the human review queue
record  → every decision and every human verdict is logged (future fine-tuning data)
```

- `LOW` is set so ≤2% of validation spoofs fall below it; `HIGH` so ≤2% of validation reals rise above it.
- Bands stay **fixed** on new data — re-tuning them on the evaluation set would inflate results.

**Agent results on the new-site test set (173 real / 50 spoof):**

| | Plain `ft_v3` @ 0.228 | `ft_v3` agent |
|---|---|---|
| Spoofs accepted with no human check | 4 / 50 (8.0%) | 3 / 50 (6.0%) |
| Real employees wrongly flagged | 36 / 173 (20.8%) | **3 / 173 (1.7%)** |
| Decided automatically | 100% | 49% |
| Sent to human review | 0% | 58% |

Agents built on `ft_new` and `ft_v2` auto-cleared 1/50 and 2/50 spoofs on the same test — within noise (one spoof = 2%). Recommended pilot set-up: **`ft_v3` + `ft_new`**, auto-clearing only when **both** models agree.

---

## Notebook guide

| # | Notebook | Stage | Run when |
|---|---|---|---|
| 01 | `Untitled14.ipynb` | Stage 0 base model | Rebuilding the base checkpoint |
| 02 | `MiniFASNetV2_FineTune_Pipeline.ipynb` | Phase 1A face fine-tune (+ LoRA experiment) | Reproducing the face-crop route |
| 03 | `Moire_Spoof_Detector.ipynb` | Side study | Analysing which cues matter |
| 04 | `Untitled19.ipynb` | Phase 1B + Phase 2B (final `ft_v3`) | Training / re-training the final model |
| — | `Attendance_Spoof_Detection.ipynb` | Early baseline: ResNet50 embeddings + KNN | Archive (earlier problem framing) |
| — | `CustomCNN_AttendanceSpoof_Training.ipynb` | Early baseline: custom CNN | Archive (earlier problem framing) |

All notebooks run on **Google Colab (T4 GPU)** and read from Google Drive; edit the `/content/drive/MyDrive/...` paths at the top of each notebook.

**Run order:** `01` → `best_model.pth` → `02` (optional, face route) → `04 Phase 1` → `A_baseline_s42/best.pth` → `04 Phase 2` → `ft_v3/best.pth`.

---

## Using the final model

```bash
pip install torch opencv-python pillow numpy pandas scikit-learn tqdm matplotlib
```

```python
import torch
ck = torch.load("models/ft_v3/best.pth", map_location="cpu", weights_only=False)
model = FullFrameMoireNet()                          # definition: notebook 04, cell 31
model.load_state_dict(ck["state_dict"], strict=True); model.eval()
print(ck["epoch"], ck["threshold"], ck["config"])    # 19  0.2283  {GLOBAL_SIZE 320, PATCH 256, SPEC 128, GRID 3}
```

Use the preprocessing from notebook 04 unchanged (`load_bgr` with EXIF rotation → letterbox to 320, scale to [-1, 1]; 3×3 grid of 256-px patches → Hann window → log-FFT → resize 128 → standardise). Any change in resizing or patch layout silently changes the scores.

---

## Metrics glossary

| Metric | Meaning | Good |
|---|---|---|
| **APCER** | Spoofs wrongly accepted as real | Low — security |
| **BPCER** | Real employees wrongly rejected | Low — fairness / friction |
| **ACER** | (APCER + BPCER) / 2 | Low |
| **AUC** | Chance a random spoof scores above a random real (threshold-free) | High |

With 50 test spoofs, one image = 2 percentage points of APCER, so small gaps between models are not significant.

---

## Data & privacy

- All images are **employee attendance photos** — personal data under India DPDPA, Malaysia/Singapore PDPA and Indonesia UU PDP. They stay on company-managed storage.
- **Never commit** images, label CSVs, split files (they contain `log_id`s), checkpoints, or notebook outputs that render photos. Strip outputs before every commit (`nbstripout --install`).

---

## Limitations & next steps

- **Small test sets** (50 new-site spoofs) → wide confidence intervals.
- **Phone-in-frame blind spot:** a small phone held up in low light shows little screen texture and can score as real. Next: add a phone/bezel detector as an agent tool so these are never auto-cleared.
- **Review load:** 58% of photos still go to humans; target is to reduce this without increasing missed spoofs.
- **Label quality:** some Phase 2 labels look inconsistent (e.g. a phone replay labelled real) → label audit.
- **Learning from reviewers:** use logged human verdicts to fine-tune the next model and to learn the routing policy.

---

## Repository checklist

- [ ] Strip all notebook outputs and purge them from git history (they contain employee photos).
- [ ] Rename notebooks as above; move the two early baselines to `notebooks/archive/`.
- [ ] In `Untitled19` cell 31, point to `..._replay0.5_real1000` (not `real900`).
- [ ] Add the evaluation notebook (4-model comparison, paired bootstrap, calibration, lighting analysis).
- [ ] Add the Phase 2A face notebook if it lives separately.
- [ ] Add the agent notebook and `agent/` code (`core.py`, `agent.py`, `agent_config.example.json`).
- [ ] Add `requirements.txt`, `.gitignore`, `LICENSE`.

---

## Acknowledgements

MiniFASNet architecture adapted from [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (Apache License 2.0). The base model in this project was trained from scratch at 224×224; no pretrained weights from that repository are used.
