import importlib

__attributes = {
    'run_drag': 'drag_utils',
    'train_lora': 'lora_utils',
    'draw_click_img':'ui_utils',
    'store_img': 'ui_utils', 
    'get_points': 'ui_utils',  
    'undo_points': 'ui_utils', 
    'clear_all': 'ui_utils', 
    'run_drag_interface': 'ui_utils', 
    'store_sample': 'ui_utils',
    'load_model': 'ui_utils',
    
    'prepare_input': 'prepare_utils',
    'prepare_models': 'prepare_utils',
    'save_everything': 'prepare_utils',
    'postprocess_output': 'prepare_utils',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]