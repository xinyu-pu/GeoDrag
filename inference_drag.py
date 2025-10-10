
import fire
import torch
import pickle
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf

from utils import (
    run_drag,
    train_lora,
    prepare_input,
    prepare_models,
    save_everything,
    postprocess_output
)

def train_LoRA(input, LoRA_Config, Model_Config):
    def inner_progress(iterator, desc):
        return tqdm(iterator, desc=desc, leave=False)
    if LoRA_Config is None:
        return None
    source_image = input['original_image']
    prompt = input['prompt']
    save_lora_path = Path(LoRA_Config['lora_output'])
    if save_lora_path.exists():
        return save_lora_path
    save_lora_path.mkdir(parents=True)
    train_lora(source_image, prompt,
            model_path = Model_Config['base_model'],
            vae_path = Model_Config.get('vae', 'default'), 
            unet=Model_Config.get('unet', None), 
            save_lora_path=save_lora_path,
            lora_step = LoRA_Config['lora_step'], 
            lora_lr = LoRA_Config['lora_lr'], 
            lora_batch_size = LoRA_Config['lora_batch_size'], 
            lora_rank = LoRA_Config['lora_rank'],
            progress = inner_progress, 
            save_interval = LoRA_Config['save_interval'])
    return save_lora_path / LoRA_Config['which_lora']

def _prepare_everything(
    cfg_file_name: str,
    original_image_path: str,
    meta_data_path: str,
    output: str,
    device: torch.device,
    dtype: torch.dtype
):
    # load config
    cfg = OmegaConf.load(cfg_file_name)
    cfg['n_actual_inference_step'] = round(cfg['inversion_strength'] * cfg['n_inference_step'])
    
    # prepare model
    models = prepare_models(
        base_model=cfg['base_model'],
        unet=cfg['unet_path'],
        vae=cfg['vae_path'],
        DepthPredictor=cfg['depth_predictor'],
        device=device,
        dtype=dtype)
    drag_model = models['drag_model']
    
    # prepare input
    input = prepare_input(
        original_image_path, meta_data_path, 
        device, dtype, 
        vae_scale_factor=drag_model.vae_scale_factor)
    
    # prepare LoRA (Optional)
    if cfg.lora is None:
        lora_configs = None
        if cfg.lora_configs is not None:
            assert isinstance(cfg.lora_configs, dict), 'lora_configs should be a dict'
            lora_path = train_LoRA(input, 
                                    cfg.lora_configs, 
                                    {'base_model': cfg['base_model'],
                                        'unet': cfg['unet_path'],
                                        'vae': cfg['vae_path']})
        else:
            lora_path= None
    else:
        lora_path = Path(cfg.lora)
    
    assert lora_path is None or lora_path.exists(), f"{lora_path} does not exist."
    
    # prepare output dir
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    
    return models, input, lora_path, cfg, output


def inference(
    cfg_file_name: str,
    image_path: str,
    meta_data_path: str,
    output: str,
    device: str = 'cuda',
    type_data: str = 'float16'
):
    device = torch.device(device)
    dtype = getattr(torch, type_data)
    
    # prepare everything
    models, input, lora_path, cfg, output = _prepare_everything(
        cfg_file_name,
        image_path,
        meta_data_path,
        output,
        device,
        dtype)
    drag_model = models['drag_model']
    depth_predictor = models['depth_predictor']
    
    edited_img_tf = run_drag(drag_model,
                             depth_predictor,
                             model_input=input,
                             configs=cfg,
                             lora_path=lora_path,)
    
    edited_img = postprocess_output(edited_img_tf, *input['original_image'].shape[2:])
    
    # save results and configures
    save_everything(
        cfg,
        Image.open(image_path),
        Image.fromarray(edited_img),
        pickle.load(open(meta_data_path, 'rb')),
        output
    )

if __name__ == '__main__':
    seed_everything(42) # random seed used by a lot of people for unknown reason
    fire.Fire(inference)