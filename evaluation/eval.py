import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import pickle
import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dift_sd import SDFeaturizer
from pytorch_lightning import seed_everything
from pathlib import Path
import lpips
import clip
from torchvision import transforms
from einops import rearrange
from tqdm import tqdm

def preprocess_image(image,
                     device):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
# = = = = = = = = = = = = = = = DAI = = = = = = = = = = = = = = #
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
""" Drag Accuracy Index (DAI) implementation
Reference:
[1] GoodDrag: Towards Good Practices for Drag Editing with Diffusion Models. https://arxiv.org/abs/2404.07206
"""
def cal_patch_size(radius: int):
    return (1 + 2 * radius) ** 2

def get_patch(image, center, radius):
    """ Extract a patch from the image centered at 'center' with given 'radius', with boundary check. """
    h, w = image.shape[:2]
    x, y = center

    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius + 1)
    y2 = min(h, y + radius + 1)

    patch = image[y1:y2, x1:x2]

    # 如果 patch 大小不符合 (2r+1, 2r+1)，补0（可选）
    expected_size = (2 * radius + 1, 2 * radius + 1)
    pad_h = expected_size[0] - patch.shape[0]
    pad_w = expected_size[1] - patch.shape[1]

    if pad_h > 0 or pad_w > 0:
        patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    return patch


def calculate_difference(patch1, patch2):
    """ Calculate the L2 norm (Euclidean distance) between two patches. """
    difference = patch1 - patch2
    squared_difference = np.square(difference)
    l2_distance = np.sum(squared_difference)

    return l2_distance

def compute_dai(original_image, result_image, points, radius):
    """ Compute the Drag Accuracy Index (DAI) for the given images and points. """
    dai = 0
    for start, target in points:
        original_patch = get_patch(original_image, start, radius)
        result_patch = get_patch(result_image, target, radius)
        dai += calculate_difference(original_patch, result_patch)
    dai /= len(points)
    dai /= cal_patch_size(radius)
    return dai / len(points)

def prepare_eval_models(device: str = 'cuda'):
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    print('Models have been prepared.')
    return {
        'dift': SDFeaturizer('stabilityai/stable-diffusion-2-1',device=device),
        'lpips': lpips.LPIPS(net='alex').to(device),
        'clip': clip_model,
        'clip_preprocess': clip_preprocess
    }

# Evaluation function
def eval(
    models,
    scores: list,
    image_path: str,
    meta_data_path: str,
    edited_image_path: str,
    device: str = 'cuda'
):
    
    meta_data_path = Path(meta_data_path)
    with open(meta_data_path, 'rb') as f:
        meta_data = pickle.load(f)
    prompt = meta_data['prompt']
    points = meta_data['points']
    
    original_image = Image.open(image_path).convert("RGB")
    edited_image = Image.open(edited_image_path).convert("RGB")
    edited_image = edited_image.resize(original_image.size, PIL.Image.BILINEAR)

    MD_output = IF_output = DAI_output = {}
    if 'MD' in scores:
        with torch.no_grad():
            # dift = SDFeaturizer('stabilityai/stable-diffusion-2-1',device=device) # use SD-2.1
            dift = models['dift']
            torch.cuda.empty_cache()
            
            handle_points = []
            target_points = []
            for idx, point in enumerate(points):
                # from now on, the point is in row,col coordinate
                cur_point = torch.tensor([point[1], point[0]])
                if idx % 2 == 0:
                    handle_points.append(cur_point)
                else:
                    target_points.append(cur_point)

            source_image_tensor = (transforms.PILToTensor()(original_image) / 255.0 - 0.5) * 2      
            dragged_image_tensor = (transforms.PILToTensor()(edited_image) / 255.0 - 0.5) * 2
            
            _, H, W = source_image_tensor.shape
            
            ft_source = dift.forward(source_image_tensor,
                        prompt=prompt,
                        t=261,
                        up_ft_index=1,    
                        ensemble_size=6,
                        device=device)  # return size: [1, c, h, w]
            ft_source = F.interpolate(ft_source, (H, W), mode='bilinear')
            
            ft_dragged = dift.forward(dragged_image_tensor,
                        prompt=prompt,
                        t=261,
                        up_ft_index=1,
                        ensemble_size=6,
                        device=device)  # return size: [1, c, h, w]
            ft_dragged = F.interpolate(ft_dragged, (H, W), mode='bilinear') 
            
            cos = nn.CosineSimilarity(dim=1)
            
            all_dist = []
            for pt_idx in range(len(handle_points)):
                hp = handle_points[pt_idx]
                tp = target_points[pt_idx]

                num_channel = ft_source.size(1)
                src_vec = ft_source[0, :, hp[0], hp[1]].view(1, num_channel, 1, 1)
                cos_map = cos(src_vec, ft_dragged).cpu().numpy()[0]  # H, W
                max_rc = np.unravel_index(cos_map.argmax(), cos_map.shape) # the matched row,col

                # calculate distance
                dist = (tp - torch.tensor(max_rc)).float().norm()
                all_dist.append(dist)
            mean_dist = torch.tensor(all_dist).mean().item()
        
        MD_output = {
            'Mean MD': mean_dist,
            'All MD': list(map(lambda x: x.item(), all_dist)),
        }
    if 'IF' in scores:
        # loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        
        # clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
        loss_fn_alex = models['lpips']
        clip_model = models['clip']
        clip_preprocess = models['clip_preprocess']
        
        source_image = preprocess_image(np.array(original_image), device)
        dragged_image = preprocess_image(np.array(edited_image), device)
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            source_image_224x224 = F.interpolate(source_image, (224,224), mode='bilinear')
            dragged_image_224x224 = F.interpolate(dragged_image, (224,224), mode='bilinear')
            
            cur_lpips = loss_fn_alex(source_image_224x224, dragged_image_224x224)
            
        source_image_clip = clip_preprocess(original_image).unsqueeze(0).to(device)
        dragged_image_clip = clip_preprocess(edited_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            source_feature = clip_model.encode_image(source_image_clip)
            dragged_feature = clip_model.encode_image(dragged_image_clip)
            source_feature /= source_feature.norm(dim=-1, keepdim=True)
            dragged_feature /= dragged_feature.norm(dim=-1, keepdim=True)
            cur_clip_sim = (source_feature * dragged_feature).sum()
        IF_output = {
            'LPIPS': cur_lpips.cpu().item(),
            'CLIP': cur_clip_sim.cpu().item(),
            'IF': 1 - cur_lpips.cpu().item()
        }
    if 'DAI' in scores:
        radius = [1, 10, 20]
        for r in radius:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            point_pairs = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
            source_image_dai = transform(np.array(original_image)).permute(1, 2, 0).numpy()
            edited_image_dai = transform(np.array(edited_image)).permute(1, 2, 0).numpy()
            dai = compute_dai(source_image_dai, edited_image_dai, point_pairs, r)
            
            DAI_output[f'DAI_r{r}'] = dai
            
    return {**MD_output, **IF_output, **DAI_output}

# Single evaluation function
def single_eval(
    scores: list,
    image_path: str,
    meta_data_path: str,
    edited_image_path: str,
    device: str = 'cuda'
):
    seed_everything(42)
    models = prepare_eval_models()
    return eval(
        models,
        scores,
        image_path,
        meta_data_path,
        edited_image_path,
        device
    )

# Full DRAGBENCH evaluation function
def eval_DRAGBENCH(
    benchmark_result_path: str,
    device: str = 'cuda'
):
    seed_everything(42)
    models = prepare_eval_models()
    DRAGBENCH_CATEGORY = [
        'art_work',
        'land_scape',
        'building_city_view',
        'building_countryside_view',
        'animals',
        'human_head',
        'human_upper_body',
        'human_full_body',
        'interior_design',
        'other_objects',
    ]
    benchmark_result_path = Path(benchmark_result_path)
    scores = ['MD', 'IF', 'DAI']
    all_MD = []
    all_IF = []
    all_DAI_1 = []
    all_DAI_10 = []
    all_DAI_20 = []

    
    with tqdm(total=205, desc="DRAGBENCH") as pbar:
        for cat in DRAGBENCH_CATEGORY:
            CAT_PATH = benchmark_result_path / cat
            assert CAT_PATH.exists(), f"{CAT_PATH} does not exist."
            for case in CAT_PATH.iterdir():
                if case.is_dir():
                    original_image_path = case / 'original_image.png'
                    meta_data_path = case / 'meta_data.pkl'
                    edited_image_path = case / 'dragged_image.png'
                    
                    quantitative_results = eval(
                        models,
                        scores,
                        original_image_path,
                        meta_data_path,
                        edited_image_path,
                        device
                    )
                    
                    all_MD = all_MD + quantitative_results['All MD']
                    all_IF.append(quantitative_results['IF'])
                    all_DAI_1.append(quantitative_results['DAI_r1'])
                    all_DAI_10.append(quantitative_results['DAI_r10'])
                    all_DAI_20.append(quantitative_results['DAI_r20'])
                    
                    pbar.set_description(f"DRAGBENCH | {cat}/{case.name}")
                    pbar.set_postfix({
                        "MD": f"{quantitative_results['Mean MD']:.3f}",
                        "IF": f"{quantitative_results['IF']:.3f}"
                    })
                    pbar.update(1)

    print(f"Mean MD: {torch.tensor(all_MD).mean().item()}")
    print(f"Mean IF: {torch.tensor(all_IF).mean().item()}")
    print(f"Mean DAI_r1: {torch.tensor(all_DAI_1).mean().item()}")
    print(f"Mean DAI_r10: {torch.tensor(all_DAI_10).mean().item()}")
    print(f"Mean DAI_r20: {torch.tensor(all_DAI_20).mean().item()}")
