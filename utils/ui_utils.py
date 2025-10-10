import os
import cv2
import numpy as np
import gradio as gr
from copy import deepcopy
import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
from .drag_utils import run_drag
import matplotlib
import pickle
from pathlib import Path
import torch
from .prepare_utils import (
    prepare_input,
    prepare_models,
    save_everything,
    postprocess_output
)

def mask_image(image,
               mask,
               color=[255,0,0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose. 
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    return out

def store_img(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    height,width,_ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length,int(length*height/width)), PIL.Image.BILINEAR)
    mask  = cv2.resize(mask, (length,int(length*height/width)), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], gr.Image.update(value=masked_img, interactive=True), mask

def get_points(img,
               sel_pix,
               evt: gr.SelectData):
    # collect the selected point
    sel_pix.append(evt.index)
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)

def undo_points(original_image,
                mask):
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(original_image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = original_image.copy()
    return masked_img, []

def clear_all(length=480):
    return gr.Image.update(value=None, height=length, width=length, interactive=True), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        [], None, None
        
def save_depth_map(depth, save_dir, timestamp):
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    raw_depth = Image.fromarray(depth.astype('uint16'))
    raw_depth.save(os.path.join(save_dir, timestamp, "row_depth.png"))

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(colored_depth).save(os.path.join(save_dir, timestamp, "colored_depth.png"))

    gray_depth = Image.fromarray(depth)
    gray_depth.save(os.path.join(save_dir, timestamp, "gray_depth.png"))

DEPTH_ANYTHING_V2 = {
    'vitl': 'checkpoints/depth_anything_v2_vitl.pth'
}

def run_drag_interface(
    models,
    
    source_image,
    image_with_clicks,
    mask,
    prompt,
    points,
    
    base_model,
    unet,
    vae,
    depth_predictor,
    
    inversion_strength,
    n_inference_step,
    gamma_ratio,
    upper_scale,
    lower_scale,
    alpha,
    beta,
    eta,
    tau,
    points_scale,
    guidance_scale,
    fill_model,
    interpolation_model,
    relocation_model,
    
    lora_path,
    
    device,
    save_dir='app_results',
    
    dtype=torch.float16,
):
    cfg = {
        'base_model': base_model,
        'unet_path': unet,
        'vae_path': vae,
        'depth_predictor': {'encoder': depth_predictor, 'pretrained': DEPTH_ANYTHING_V2[depth_predictor]},
        'inversion_strength': inversion_strength,
        'n_inference_step': n_inference_step,
        'n_actual_inference_step': round(inversion_strength * n_inference_step),
        'guidance_scale': guidance_scale,
        'gamma': gamma_ratio,
        'upper_scale': upper_scale,
        'lower_scale': lower_scale,
        'alpha': alpha,
        'beta': beta,
        'eta': eta,
        'tau': tau,
        'points_scale': points_scale,
        'fill_mode': fill_model,
        'interpolation_model': interpolation_model,
        'relocation_model': relocation_model,
        'lora': None,
        'lora_configs': None,
    }
    
    # prepare input
    output=Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_everything(
        cfg,
        original_image=Image.fromarray(source_image),
        dragged_image=None,
        meta_data={
            'points': points,
            'prompt': prompt,
            'mask': mask
        },
        output=output,
    )
    
    drag_model = models['drag_model']
    depth_predictor = models['depth_predictor']
    
    input = prepare_input(
        output / "original_image.png", output / "meta_data.pkl", 
        device, dtype, 
        vae_scale_factor=drag_model.vae_scale_factor,
        verbose=False)

    edited_image  = run_drag(
        drag_model,
        depth_predictor,
        model_input=input,
        configs=cfg,
        lora_path=lora_path,)
    edited_image = postprocess_output(edited_image, *input['original_image'].shape[2:])
    
    # save
    source_image = source_image.astype(np.uint8)
    image_with_clicks = image_with_clicks.astype(np.uint8)
    edited_image = edited_image.astype(np.uint8)
    
    H = source_image.shape[0]
    separator = np.ones((H, 25, 3), dtype=np.uint8) * 255  # 白条
    summary = np.concatenate([
        source_image,
        separator,
        image_with_clicks,
        separator,
        edited_image
    ], axis=1)

    Image.fromarray(summary).save(output / "summary.png")
    Image.fromarray(edited_image).save(output / "dragged_image.png")
        
    return edited_image

def store_sample(input_image, sample_save_dir, selected_points, prompt):
    sample_save_dir = Path(sample_save_dir)
    sample_save_dir.mkdir(parents=True, exist_ok=True)
    
    ori_image, mask = input_image["image"], input_image["mask"]
    
    if mask.ndim == 3:
        mask = np.float32(mask[:, :, 0]) / 255.
    else:
        mask = np.float32(mask) / 255.
        
    ori_image = Image.fromarray(ori_image)
    ori_image.save(sample_save_dir / 'original_image.png')
    
    with open(sample_save_dir / 'meta_data.pkl', "wb") as f:
        pickle.dump({
            "points": selected_points,
            "prompt": prompt,
            "mask": mask
        }, f)

def load_model(model_path, unet_path, vae_path, depthanythingv2, device, dtype):
    models = prepare_models(
        base_model=model_path,
        unet=unet_path,
        vae=vae_path,
        DepthPredictor={'encoder': depthanythingv2, 'pretrained': DEPTH_ANYTHING_V2[depthanythingv2]},
        device=device,
        dtype=dtype)
    return models
    