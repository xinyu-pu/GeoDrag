import fire
import torch
import pickle
from PIL import Image
from pathlib import Path
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf

from utils import (
    run_drag,
    prepare_models,
    prepare_input,
    save_everything,
    postprocess_output,
)

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

def evaluate(
    benchmark_path: str,
    cfg_file_name: str,
    output: str,
    lora_path_root: str = None,
    which_lora: str = '80',
    device: str = 'cuda',
    type_data: str = 'float16'
):
    device = torch.device(device)
    dtype = getattr(torch, type_data)
    
    cfg = OmegaConf.load(cfg_file_name)
    cfg['n_actual_inference_step'] = round(cfg['inversion_strength'] * cfg['n_inference_step'])
    
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    
    # prepare model
    models = prepare_models(
        base_model=cfg['base_model'],
        unet=cfg['unet_path'],
        vae=cfg['vae_path'],
        DepthPredictor=cfg['depth_predictor'],
        device=device,
        dtype=dtype)
    drag_model = models['drag_model']
    depth_predictor = models['depth_predictor']
    
    benchmark_path = Path(benchmark_path)
    if lora_path_root is not None:
        lora_path_root = Path(lora_path_root)
        assert lora_path_root.exists(), f"{lora_path_root} does not exist."
    for cat in DRAGBENCH_CATEGORY:
        CAT_PATH = benchmark_path / cat
        assert CAT_PATH.exists(), f"{CAT_PATH} does not exist."
        print(f"Processing {cat} ...")
        if lora_path_root is not None:
            lora_path_cat = lora_path_root / f"{cat}_lora.safetensors"
            assert lora_path_cat.exists(), f"{lora_path_cat} does not exist."
        else:
            lora_path_cat = None
        for case in CAT_PATH.iterdir():
            if case.is_dir():
                if lora_path_cat is not None:
                    lora_path = lora_path_cat / f"{case.name}" / f"{which_lora}"
                    print(f"Using LoRA from {lora_path}")
                else:
                    lora_path = None
                
                original_image_path = case / 'original_image.png'
                meta_data_path = case / 'meta_data.pkl'
                
                # prepare input
                input = prepare_input(
                    original_image_path, meta_data_path, 
                    device, dtype, 
                    vae_scale_factor=drag_model.vae_scale_factor,
                    verbose=False)
                
                # inference
                with torch.no_grad():
                    dragged_image = run_drag(
                        drag_model,
                        depth_predictor,
                        model_input=input,
                        configs=cfg,
                        lora_path=lora_path,)
                dragged_image = postprocess_output(dragged_image, *input['original_image'].shape[2:])
                
                # save results
                case_output = output / cat / case.name
                save_everything(
                    cfg,
                    Image.open(original_image_path),
                    Image.fromarray(dragged_image),
                    pickle.load(open(meta_data_path, 'rb')),
                    case_output
                )

if __name__ == "__main__":
    seed_everything(42)
    fire.Fire(evaluate)