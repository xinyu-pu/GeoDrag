import os
import gradio as gr
import torch
from pathlib import Path
from utils import (
    store_img, 
    get_points, 
    undo_points, 
    clear_all, 
    run_drag_interface, 
    store_sample,
    load_model,
)
app_theme = gr.themes.Default()
# -------- Based configure for Web GUI -------- #

LENGTH=480 # length of the square area displaying/editing images
DEFAULT_MODEL_PATH = "runwayml/stable-diffusion-v1-5"
DEFAULT_UNET_PATH  = "SimianLuo/LCM_Dreamshaper_v7"
DEFAULT_VAE_PATH   = "default"
DEFAULT_DEPTHANY   = "vitl"
DEFAULT_DEVICE     = torch.device("cuda")
DEFAULT_DTYPE      = torch.float16

loading_css = """
:root,[data-theme]{
  --background-fill-secondary:#fff;
  --body-background-fill:#fff;
  --panel-background-fill:#fff;
}

#loading_shell{
  min-height:70vh;
  display:flex;
  align-items:center;
  justify-content:center;
}

.loading-card{
  width:min(640px,92vw);
  padding:28px;
  border:1px solid #e5e7eb;
  border-radius:16px;
  box-shadow:0 6px 24px rgba(0,0,0,.06);
  background:#fff;
}

.loading-title{font-size:28px;font-weight:700;color:#111827;margin:0;}
.loading-sub{color:#6b7280;margin:0 0 18px 0;}

.spinner{
  width:56px;height:56px;border-radius:50%;
  border:6px solid #e5e7eb;border-top-color:#2563eb;
  animation:spin 1s linear infinite;margin:10px auto 16px auto;
}
@keyframes spin{to{transform:rotate(360deg);}}

.progress{height:8px;background:#f3f4f6;border-radius:999px;overflow:hidden;margin:8px 0 6px 0;}
.progress .bar{width:40%;height:100%;background:#2563eb;border-radius:999px;animation:indet 1.2s ease-in-out infinite;}
@keyframes indet{
  0%{transform:translateX(-100%);}
  50%{transform:translateX(60%);}
  100%{transform:translateX(120%);}
}

.tip{
  display:flex;gap:10px;align-items:flex-start;padding:10px 12px;background:#f8fafc;
  border:1px dashed #e2e8f0;border-radius:12px;color:#475569;font-size:14px;
}

.loading-footer{color:#94a3b8;font-size:12px;text-align:center;margin-top:10px;}
"""
# ---------------------------------------------- #

with gr.Blocks(theme=app_theme, css=loading_css) as demo:
    mask = gr.State(value=None) # store mask
    selected_points = gr.State([]) # store points
    original_image = gr.State(value=None) # store original input image
    models = gr.State(value=None) 
    
    # ============== Loading ==============
    with gr.Column(visible=True, elem_id="loading_group") as loading_group:
        with gr.Column(elem_id="loading_shell"):
            with gr.Column(elem_classes="loading-card"):
                with gr.Row(elem_classes="loading-headline"):
                    gr.Markdown("<h1 class='loading-title'>Official Implementation of GeoDrag</h1>")
                gr.Markdown("<p class='loading-sub'>Booting up models & warming GPU…</p>")
                gr.HTML("<div class='spinner'></div>")  
                gr.HTML("<div class='progress'><div class='bar'></div></div>")  

                status_md = gr.Markdown("Loading models, please wait a moment……")

                gr.HTML("""
                <div class="tip">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 8v4" stroke="#2563eb" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="16" r="1.2" fill="#2563eb"/></svg>
                <div>
                    <div><strong>Tip:</strong> The first load will download/initialize weights, which may take a bit longer; subsequent sessions will be faster. </div>
                </div>
                </div>
                """)

                gr.Markdown("<div class='loading-footer'>If this screen persists unusually long, check console logs.</div>")

    with gr.Column(visible=False, elem_id="main_group") as main_group:
        # ------------ layout definition ------------ #
        with gr.Row():
            gr.Markdown("""
            # Official Implementation of GeoDrag
            """)
        # ------------------------------------------- #
        
        with gr.Row():
            with gr.Column(scale=1, elem_classes="up-pane"):
                with gr.Accordion("⚙️ Settings", open=False):
                    with gr.Tabs():
                        with gr.Tab("Drag Config"):
                            with gr.Row():
                                inversion_strength = gr.Slider(0, 1.0,
                                    value=0.7,
                                    label="inversion strength",
                                    info="The latent at [inversion-strength * total-sampling-steps] is optimized for dragging.")
                                n_inference_step = gr.Slider(0, 1000,
                                    value=10,
                                    label="inference step",
                                    info="The latent at [inversion-strength * total-sampling-steps] is optimized for dragging.")
                                fill_model = gr.Radio(choices=["0", "random", "interpolation", "ori"], value="interpolation", label="Latent fill model",
                                    info="The model to fill the NULL region after relocation.")
                                interpolation_model = gr.Radio(choices=["dynamic", "static"], value="dynamic", label="Interpolation model",
                                    info="The interpolation model after relocation.")
                                relocation_model = gr.Radio(choices=["first-win", "last-win", "mean-merge", "softmax-merge"], value="first-win", label="Relocation model",
                                    info="The relocation model to use.")
                            with gr.Row():
                                gamma_ratio = gr.Number(value=1.0, label="Gamma ratio", precision=None)
                                alpha = gr.Number(value=1.0, label="Alpha", precision=None)
                                beta = gr.Number(value=1.0, label="Beta", precision=None)
                                upper_scale = gr.Number(value=5, label="Upper scale", precision=None)
                                lower_scale = gr.Number(value=0, label="Lower scale", precision=None)
                                eta = gr.Number(value=0.0, label="Eta", precision=None)
                                points_scale = gr.Number(value=1.0, label="Points scale", precision=None)
                                guidance_scale = gr.Number(value=1.0, label="Guidance scale", precision=None)
                                tau = gr.Number(value=1.0, label="Tau", precision=None)
                                
                        with gr.Tab("Base Model Config"):
                            with gr.Row():
                                local_models_dir = Path('local_pretrained_models')
                                local_models_dir.mkdir(exist_ok=True, parents=True)
                                local_models_choice = \
                                    [os.path.join(local_models_dir,d) for d in os.listdir(local_models_dir) if os.path.isdir(os.path.join(local_models_dir,d))]
                                model_path = gr.Dropdown(value="runwayml/stable-diffusion-v1-5",
                                    label="Diffusion Model Path",
                                    choices=[
                                        "runwayml/stable-diffusion-v1-5",
                                        "gsdf/Counterfeit-V2.5",
                                        "stablediffusionapi/anything-v5",
                                        "SG161222/Realistic_Vision_V2.0",
                                    ] + local_models_choice
                                )
                                unet_path = gr.Dropdown(value="SimianLuo/LCM_Dreamshaper_v7",
                                    label="UNet choice",
                                    choices=["SimianLuo/LCM_Dreamshaper_v7"] + local_models_choice
                                )
                                vae_path = gr.Dropdown(value="default",
                                    label="VAE choice",
                                    choices=["default",
                                    "stabilityai/sd-vae-ft-mse"] + local_models_choice
                                )
                                depthanythingv2 = gr.Dropdown(value="vitl",
                                    label="Depth Predictor choice",
                                    choices=["vitl", "vitg", "vitb"]
                                )
                                load_model_button = gr.Button("Reload Model")
        # ------------ Drag ------------ #
        with gr.Tab(label="Editing Image"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""<p style="text-align: center; font-size: 20px">Choose Region</p>""")
                    canvas = gr.Image(type="numpy", tool="sketch", label="Draw Mask",
                        show_label=True, height=LENGTH, width=LENGTH) # for mask painting
                with gr.Column():
                    gr.Markdown("""<p style="text-align: center; font-size: 20px">Click Points</p>""")
                    input_image = gr.Image(type="numpy", label="Click Points",
                        show_label=True, height=LENGTH, width=LENGTH, interactive=False) # for points clicking
                with gr.Column():
                    gr.Markdown("""<p style="text-align: center; font-size: 20px">Editing Results</p>""")
                    output_image = gr.Image(type="numpy", label="Editing Results",
                        show_label=True, height=LENGTH, width=LENGTH, interactive=False)
                    
            with gr.Row():
                undo_button = gr.Button("Undo point")
                run_button = gr.Button("Run")
                clear_all_button = gr.Button("Clear All")
            with gr.Accordion("LoRA (Optional)", open=False):
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", info='Text prompt for the model')
                    lora_path = gr.Textbox(value="./lora", label="LoRA path", info="Path to save LoRA model")
                    train_lora_button = gr.Button("Train LoRA (optional)")
                    lora_status_bar = gr.Textbox(label="display LoRA training status", info="Training status of LoRA model")
                    lora_path = gr.Textbox(value=None, label="Which LoRA to use", info="Path to the trained LoRA model")
                with gr.Row():
                    lora_step = gr.Number(value=80, label="LoRA training steps", precision=0)
                    lora_lr = gr.Number(value=0.0005, label="LoRA learning rate")
                    lora_batch_size = gr.Number(value=4, label="LoRA batch size", precision=0)
                    lora_rank = gr.Number(value=16, label="LoRA rank", precision=0)
        with gr.Tab(label="Make Sample"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""<p style="text-align: center; font-size: 20px">Choose Region</p>""")
                    canvas_sample = gr.Image(type="numpy", tool="sketch", label="Draw Mask",
                        show_label=True) # for mask painting
                with gr.Column():
                    gr.Markdown("""<p style="text-align: center; font-size: 20px">Click Points</p>""")
                    input_image_sample = gr.Image(type="numpy", label="Click Points",
                        show_label=True, interactive=False) # for points clicking
            with gr.Row(equal_height=True):
                save_dir = gr.Textbox(value="./sample", label="Sample save path")
                save_button = gr.Button("Save")
    # ------------ Trigger ------------ #
    canvas.edit(
        store_img,
        [canvas],
        [original_image, selected_points, input_image, mask]
    )
    canvas_sample.edit(
        store_img,
        [canvas_sample],
        [original_image, selected_points, input_image_sample, mask]
    )
    
    input_image.select(
        get_points,
        [input_image, selected_points],
        [input_image],
    )
    input_image_sample.select(
        get_points,
        [input_image_sample, selected_points],
        [input_image_sample],
    )


    undo_button.click(
        undo_points,
        [original_image, mask],
        [input_image, selected_points]
    )
    clear_all_button.click(
        clear_all,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [canvas,
        input_image,
        output_image,
        selected_points,
        original_image,
        mask]
    )
    
    def run_guard(models_obj, *rest):
        if models_obj is None:
            raise gr.Error("Model not ready yet. Please wait for loading to finish.")
        return run_drag_interface(models_obj, *rest)

    run_button.click(
        run_guard,
        [models,
         original_image,
         input_image,
         mask,
         prompt,
         selected_points,
         model_path,
         unet_path,
         vae_path,
         depthanythingv2,
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
         gr.State(DEFAULT_DEVICE)
        ],
        [output_image]
    )
    
    save_button.click(
        store_sample,
        [canvas, save_dir, selected_points, prompt],
        []
    )
    
    
    def startup_load(model_path_v, unet_path_v, vae_path_v, depth_v, device_v, dtype_v):
        try:
            mdls = load_model(model_path_v, unet_path_v, vae_path_v, depth_v, device_v, dtype_v)

            return (
                mdls,
                gr.update(value="✅ The model has finished loading!"),
                gr.update(visible=False), 
                gr.update(visible=True), 
            )

        except Exception as e:
            import traceback
            tb = traceback.format_exc()

            error_html = f"""
            ❌  Model loading failed! 
           
           
            {tb}
           
            """
            return (
                gr.update(value=None),
                gr.update(value=error_html),  
                gr.update(visible=True),      
                gr.update(visible=False),    
            )
        
    demo.load(
        startup_load,
        inputs=[  
            gr.State(DEFAULT_MODEL_PATH),
            gr.State(DEFAULT_UNET_PATH),
            gr.State(DEFAULT_VAE_PATH),
            gr.State(DEFAULT_DEPTHANY),
            gr.State(DEFAULT_DEVICE),
            gr.State(DEFAULT_DTYPE),
        ],
        outputs=[models, status_md, loading_group, main_group]
    )
    
    def reload_and_notify(model_path_v, unet_path_v, vae_path_v, depth_v, device_v, dtype_v):
        yield (
            gr.update(value="Reloading the model……"),
            None,
            gr.update(visible=True),
            gr.update(visible=False)
        )
        mdls = load_model(model_path_v, unet_path_v, vae_path_v, depth_v, device_v, dtype_v)
        yield (
            gr.update(value="The model has been reloaded!"), 
            mdls,
            gr.update(visible=False), 
            gr.update(visible=True),
        )


    load_model_button.click(
        reload_and_notify,
        inputs=[model_path, unet_path, vae_path, depthanythingv2, gr.State(DEFAULT_DEVICE), gr.State(DEFAULT_DTYPE)],
        outputs=[status_md, models, loading_group, main_group]
    )

demo.queue(concurrency_count=1, max_size=20)     
demo.launch()