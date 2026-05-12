# Orthogonal Procrustes Residual Alignment (LN + Rescale)

This document describes the implementation of the **LN + Rescale + Orthogonal Procrustes** residual alignment for SD3 text features, including the offline precomputation step and the runtime inference changes.

## 1. Offline precomputation (calibration set)

**Script:** `compute_sd3_text_procrustes.py`

### What it does
- Uses the same dataset selection logic as `compute_sd3_text_grad_sensitivity.py`.
- Takes the **first `n` samples** from the specified dataset(s).
- Collects per-layer **text input states** (the `encoder_hidden_states` entering each transformer block).
- For every target layer `i` (default: layers `2..last`), solves the **Orthogonal Procrustes** problem:
  \[
  \min_R \|\bar{X}R - \bar{Y}\|_F^2, \; \text{s.t.}\; R^\top R = I
  \]
  where `X` is the origin layer token set and `Y` is the target layer token set (both centered).
- For each sample, the script **randomly samples a timestep index** (no fixed `timestep_idx`).

### Output format
The script saves a `.pt` file with:
```python
{
  "origin_layers": List[int],
  "origin_layer": Optional[int],  # kept when only one origin layer is used
  "target_layers": List[int],
  "rotation_matrices": Tensor[num_layers, d, d],
  "feature_dim": int,
  "residual_use_layernorm": bool,
  "num_samples": int,
  "num_valid_tokens": int,
  "timestep_idx": int,
  "use_padding_mask": bool,
}
```

### Example
```bash
python compute_sd3_text_procrustes.py \
  --model sd3 \
  --dataset coco \
  --datadir /path/to/datasets \
  --num-samples 200 \
  --origin-layers 1 2 3 \
  --target-layer-start 4 \
  --residual_use_layernorm 1 \
  --output procrustes_rotations.pt
```

`--residual_use_layernorm 1` means the script applies `apply_simulated_ln(...)` before Procrustes,
while `0` uses raw text features directly.

## 2. Runtime inference changes

### Core residual logic
**File:** `transformer.py`

The residual path now supports an optional **rotation matrix** per target layer:
1. **Token-wise LN (standardization)**
2. **Orthogonal rotation**: `o_norm = o_norm @ R`
3. **Residual addition**: `t_norm + w * o_norm`
4. **Optional LayerNorm**
5. **Rescale back to target statistics**

### Wiring the rotation matrices into inference
**Files:** `sampler.py`, `sample.py`, `generate_t2i.py`, `generate_geneval.py`, `generate_dpg.py`

- `SD3Transformer2DModel_Residual.forward(...)` accepts `residual_rotation_matrices`.
- `SD3Euler.sample_residual(...)` forwards the rotation matrices to the denoiser.
- In the sampling scripts, you can provide `--residual_procrustes_path` to load the saved matrices.
  - If `--residual_target_layers` is not specified, the script will use `target_layers`
    from the saved file automatically.
  - If neither `--residual_origin_layer` nor `--residual_origin_layers` is specified,
    the script will default to `origin_layers` (or legacy `origin_layer`) from the saved file.

### Example usage at inference
```bash
python sample.py \
  --prompt "a watercolor cat in a garden" \
  --save_dir ./outputs \
  --residual_origin_layers 1 2 3 \
  --residual_weights 0.5 \
  --residual_procrustes_path procrustes_rotations.pt
```
