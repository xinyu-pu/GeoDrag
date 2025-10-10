
import numpy as np
import torch
import pickle
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
from einops import rearrange
import torch.nn.functional as F
import torchvision.transforms.functional as Fu
import cv2

from diffusers import (
    DDIMScheduler,
    AutoencoderKL
)

from pipelines import (
    DragPipeline,
    DepthAnythingV2,
    UNet2DConditionModel
)


def draw_click_img(
    img,
    sel_pix
):
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

def prepare_input(
    original_image_path: str,
    meta_data_path: str,
    device: str = 'cuda',
    dtype: str = 'float16',
    verbose = True,
    **kwargs
):
    # prepare input
    original_image_path = Path(original_image_path)
    meta_data_path = Path(meta_data_path)
    
    original_image = Image.open(original_image_path)
    ori_image_np = np.array(original_image)
    full_h, full_w = ori_image_np.shape[:2]
    with open(meta_data_path, 'rb') as f:
        meta_data = pickle.load(f)
    prompt = meta_data['prompt']
    mask = meta_data['mask']
    points = meta_data['points']
    
    # preprocess input 
    dows_sample_scale = kwargs['vae_scale_factor']
    
    mask = torch.from_numpy(mask).float() / 255.
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w")
    mask = Fu.resize(mask, 
                    (int(full_h/dows_sample_scale), int(full_w/dows_sample_scale)))
    
    handle_points = []
    target_points = []
    for idx, point in enumerate(points):
        cur_point = torch.tensor(point).to(device)
        cur_point = torch.round(cur_point / dows_sample_scale)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    handle_points = torch.stack(handle_points, dim=0) # [N, 2]
    target_points = torch.stack(target_points, dim=0) # [N, 2]
    
    ori_image_tf = torch.from_numpy(ori_image_np).float() / 127.5 - 1 # range(-1, 1)
    ori_image_tf = rearrange(ori_image_tf, "h w c -> 1 c h w")

    if verbose:
        print(f'Inputs have been prepared.')
    return {
        'original_image': ori_image_tf.to(device, dtype),
        'prompt': prompt,
        'mask': mask.to(device, dtype),
        'handle_points': handle_points,
        'target_points': target_points,
        'image': ori_image_np
    }

def postprocess_output(image: torch.Tensor, full_h: int, full_w: int):
    image = F.interpolate(image, (full_h, full_w), mode='bilinear')
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    return (image * 255).astype(np.uint8)

def prepare_models(base_model: str,
                   unet: str,
                   DepthPredictor: dict = {'encoder': 'vitl', 'pretrained': f'checkpoints/depth_anything_v2_vitl.pth'},
                   vae: str = None,
                   device: str = 'cuda',
                   dtype: torch.dtype = torch.float16,
                   verbose = True):
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
    model = DragPipeline.from_pretrained(base_model, scheduler=scheduler, torch_dtype=dtype)
    if unet:
        unet = UNet2DConditionModel.from_pretrained(
                        unet,
                        subfolder="unet",
                        torch_dtype=dtype,)
        model.unet = unet
    model.modify_unet_forward()
    if vae != 'default':
        model.vae = AutoencoderKL.from_pretrained(
            vae
        ).to(model.vae.device, model.vae.dtype)
    model.enable_model_cpu_offload(device=device)
    
    depth_anything_v2_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_predictor = DepthAnythingV2(**depth_anything_v2_configs[DepthPredictor['encoder']])
    depth_predictor.load_state_dict(torch.load(DepthPredictor['pretrained'], map_location='cpu'))
    depth_predictor = depth_predictor.to(device).eval()

    if verbose:
        print(f'Models have been prepared on {device}.')
    
    return {
        'drag_model': model,
        'depth_predictor': depth_predictor,
    }

def save_everything(
    cfg: dict,
    original_image: Image,
    dragged_image: Image,
    meta_data: dict,
    output: Path
):
    output.mkdir(parents=True, exist_ok=True)
    if original_image is not None:
        original_image.save(output / "original_image.png")
    if dragged_image is not None:
        dragged_image.save(output / "dragged_image.png")
    if meta_data is not None:
        with open(output / "meta_data.pkl", 'wb') as f:
            pickle.dump(meta_data, f)
        OmegaConf.save(cfg, output / "config.yaml")
    
    # save click img
    if original_image is not None and meta_data is not None and 'points' in meta_data:
        click_img = Image.fromarray(draw_click_img(
            np.array(original_image),
            meta_data['points']
        ))
        click_img.save(output / "click_image.png")