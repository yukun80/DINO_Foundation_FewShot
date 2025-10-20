# AGENTS

This document summarises how the Foundation Few-Shot framework composes its
core “agents” (training, evaluation, prediction) after the recent frequency
adapter integration. It is intended as a quick guide for orchestrating runs
and understanding which configuration knobs must remain aligned across the
pipeline.

## Model Assembly

- **Backbone**: `models/backbones/dino.py` builds `DINO_linear`, loading
  DINOv1/v2 weights from `pretrain_dir` (default `./pretrain`).
- **Frequency Adapter**: `FrequencyEnhancer` wraps the lazy FFT modules from
  `modules/module1`, applying `MaskModule` (APM) and `PhaseAttention` (ACPA)
  per feature branch.
- **Branch Modes**:
  - `linear` / `svf`: single branch, adapter applied once.
  - `multilayer`: four branches concatenated; choose `freq_mask_mode` to share
    or specialise masks across branches (`per_layer` or `shared`).

## Configuration Keys (shared by train/eval/predict)

| Key | Purpose |
| --- | --- |
| `pretrain_dir` | Directory with DINO/DINOv2 pretraining weights. |
| `enable_frequency_adapter` | Toggles APM/ACPA stack; set `false` for ablations. |
| `freq_mask_mode` | Adapter sharing: `per_layer` (default) or `shared`. |
| `dinov2_size` | Backbone width (`small`, `base`, `large`). Must match pretrain files. |
| `method` | Few-shot head (`linear`, `multilayer`, `svf`, `lora`, `vpt`). |
| `number_of_shots` | Support/query split size; ensure split files agree. |

> All Sacred scripts (`train.py`, `eval.py`, `predict.py`) merge overrides
> supplied with `with key=value …`. Keep parameter sets consistent across
> stages to avoid shape mismatches.

## Training Agent

Run Sacred training with:

```bash
python3 train.py with method=<mode> nb_shots=<shots> lr=<lr> run_id=<n>
```

- Example for multilayer, 20-shot as used in experiments:

  ```bash
  python3 train.py with method=multilayer nb_shots=20 lr=0.001 run_id=1
  ```

- Artifacts land in `experiments/FSS_Training/<run_id>` with
  `best_model.pth` and Sacred logs.
- `svf`, `lora`, `vpt` still expect a pretrained linear decoder (see config
  `linear_weights_path`).

## Evaluation Agent

Evaluate a checkpoint while reusing the same hyperparameters:

```bash
python3 eval.py with \
  checkpoint_path='experiments/FSS_Training/<run>/best_model.pth' \
  method=<mode> nb_shots=<shots>
```

- The script rebuilds `DINO_linear` using `pretrain_dir` and the frequency
  settings from `configs/disaster.yaml` plus any overrides.
- Metrics (mIoU, OA, Precision/Recall/F1) are logged under
  `experiments/FSS_Evaluation/<run>`.

## Prediction Agent

Generate visual masks using the same pattern:

```bash
python3 predict.py with \
  checkpoint_path='experiments/FSS_Training/<run>/best_model.pth' \
  method=<mode> nb_shots=<shots>
```

- Predictions are saved inside the Sacred run folder under
  `experiments/FSS_Prediction/<run>` and attached as artifacts.
- Colour palette (`modules/module1`) defaults to background black / foreground
  red; adjust there for new datasets.

## Best Practices

- **Consistency**: always match `method`, `dinov2_size`, `freq_mask_mode`, and
  `enable_frequency_adapter` between training and downstream agents.
- **Precision**: the entire pipeline now runs in FP32 to preserve FFT
  stability; avoid re-enabling autocast unless phase operations are guarded.
- **Pretraining Assets**: verify the required `dinov2_vit*.pth` files exist
  under `pretrain_dir`. Missing files trigger an explicit error when models
  are built.
- **Ablations**: for baseline comparisons, set
  `enable_frequency_adapter=false` or use `freq_mask_mode=shared` to reduce
  adapter capacity and isolate gains.

