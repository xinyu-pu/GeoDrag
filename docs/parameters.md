# Parameters

This document explains all configuration options used in `config.yaml`.

---

## 1. Model Configs

### `base_model`
- **Type:** string (Hugging Face model id or local path)  
- **Default:** `runwayml/stable-diffusion-v1-5`  
- **Description:**  
  The main diffusion backbone the system is built on (for UNet/CLIP/Tokenizer).  
  Must be compatible with the UNet and VAE versions used.

---

### `unet_path`
- **Type:** string (Hugging Face id or local path)  
- **Default:** `SimianLuo/LCM_Dreamshaper_v7`  
- **Description:**  
  Replaces the UNet in `base_model` (e.g., an LCM fine-tuned UNet for faster inference).  
  Make sure the scheduler and base model family match.

---

### `vae_path`
- **Type:** string (`'default'` | HF id | local path)  
- **Default:** `default`  
- **Description:**  
  Specifies the VAE used for encoding/decoding.  
  `'default'` means the built-in VAE from the base model.  
---

### `depth_predictor`
#### `encoder`
- **Type:** string  
- **Default:** `vitl`  
- **Description:**  
  Encoder backbone type for depth prediction.

#### `pretrained`
- **Type:** path  
- **Default:** `checkpoints/depth_anything_v2_vitl.pth`  
- **Description:**  
  Pretrained checkpoint for the depth model.  
  **NOTE:** Ensure the encoder type matches the pretrained weights.

---

## 2. Drag Configs

These parameters control the inversion, optimization, and motion behavior during dragging.

| Parameter | Type | Default | Description |
|------------|------|----------|--------------|
| `inversion_strength` | float | 0.7 | Controls the inversion strength of the input image. Higher → closer to latent inversion. |
| `n_inference_step` | int | 10 | Number of diffusion steps during editing. |
| `guidance_scale` | float | 1.0 | Classifier-free guidance strength. |
| `gamma` | float | 1.0 | Global weight for drag loss. Lower → stronger geometric guidance. |
| `upper_scale` | float | 5.0 | Upper bound for geometry-aware movement scaling. |
| `lower_scale` | float | 0.0 | Lower bound for geometry-aware movement scaling. |
| `alpha` | float | 1.0 | Modulation factor controls the sensitivity of displacement scaling to depth variations.. |
| `beta` | float | 1.0 | Modulation factor controls how sharply the influence falls off with pixel distance.|
| `eta` | float | 0 | Noise term (controls stochasticity in sampling). |
| `tau` | float | 1.0 | The temperature parameter for softmax-merge relocation model. |
| `points_scale` | float | 1.0 | Scales the influence of each control point’s motion. |
| `fill_mode` | string | `'interpolation'` | Defines how to fill NULL regions after relocation (`interpolation`, `inpaint`, `none`). |
| `interpolation_model` | string | `'static'` | Model used to interpolate NULL regions after relocation (`'dynamic'`, `'static'`). Dynamic gives adaptive interpolation; static is simpler.|
| `relocation_model` | string | `'first-win'` | Strategy to resolve pixel ownership when multiple relocation influences overlap. |

---

## 3. LoRA Configs (Optional)

LoRA (Low-Rank Adaptation) allows lightweight finetuning or adaptation of the UNet.

### `lora`
- **Type:** string or null  
- **Default:** empty  
- **Description:**  
  Path of the LoRA weights to apply.

---

### `lora_configs`
Optional hyperparameters for training or selecting specific LoRA checkpoints.

| Parameter | Type | Description |
|------------|------|-------------|
| `lora_step` | int | Number of training steps. |
| `lora_lr` | float | Learning rate for LoRA training. |
| `lora_batch_size` | int | Batch size for LoRA updates. |
| `lora_rank` | int | Rank of LoRA layers. |
| `save_interval` | int | Save checkpoint every N steps. |
| `which_lora` | string | Identifier of a specific step LoRA checkpoint to load. |

---

## 5. Example Settings

### Parameter Tuning Guide

> Quick rules of thumb for adjusting behavior and consistency.

- **Improve geometric consistency (stronger 3D awareness)**  
  - Decrease `gamma` or increase `alpha`.  
  - This enhances *depth-aware* geometric displacement, producing motion that aligns better with scene depth.  
  - ⚠️ Too small a `gamma` may cause artifact.

- **Enhance local consistency (more stable planar editing)**  
  - Increase `gamma` or increase `beta`.  
  - This emphasizes *plane-aware* local smoothness, keeping edges and nearby pixels more coherent.  
  - ⚠️ Large values may introduce over-dragging.

- **If the drag effect appears too weak**  
  - Slightly increase `points_scale` (e.g., from `1.0` → `1.05–1.1`).  
  - ⚠️ Avoid excessive values, which can lead to texture tearing.

- **If background looks blurry or lacks detail**  
  - Try increasing `eta` (e.g., `0.05–0.5`).  
  - Adds controlled stochasticity during sampling, improving fine-detail recovery.  

### Default Configuration
```yaml
# model configs
base_model: 'runwayml/stable-diffusion-v1-5'
unet_path: 'SimianLuo/LCM_Dreamshaper_v7'
vae_path: 'default'
depth_predictor: 
  encoder: 'vitl'
  pretrained: 'checkpoints/depth_anything_v2_vitl.pth'

# drag configs
inversion_strength: 0.7
n_inference_step: 10
guidance_scale: 1.0
gamma: 1.0
upper_scale: 5.0
lower_scale: 0.0
alpha: 1.0
beta: 1.0
eta: 0
tau: 1.0
points_scale: 1.0
fill_mode: 'interpolation'
interpolation_model: 'dynamic'
relocation_model: 'first-win'

# lora configs (optional)
lora: 
lora_configs:
```