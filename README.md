<div align="center">
  <h1>Dragging with Geometry: From Pixels to Geometry-Guided Image Editing</h1>
  <p>
    <strong>Xinyu Pu</strong> &nbsp;&nbsp;
    <strong>Hongsong Wang</strong> &nbsp;&nbsp;
    <strong>Jie Gui</strong> &nbsp;&nbsp;
    <strong>Pan Zhou</strong><br>
    <b>Southeast University</b> &nbsp; | &nbsp; <b>Singapore Management University</b>
  </p>
  <p>
    <a href="https://arxiv.org/abs/2509.25740"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2509.25740-b31b1b.svg"></a>
    <a href="https://geodrag-site.github.io/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Website-orange"></a>
  </p>
  <img src="./assets/demo.gif" width="700">
  <h5>⭐ Like our project? Don’t forget to star it on GitHub and stay tuned for future updates!</h5>
</div>

## Disclaimer
This is a research project, NOT a commercial product. Users are granted the freedom to create images using this tool, but they are expected to comply with local laws and utilize it in a responsible manner. The developers do NOT assume any responsibility for potential misuse by users.

## News and Updates
* 🚨 Released the implementation code of GeoDrag.
* 🚨 **Paper Portal for Top Conferences in the Field of Artificial intelligence: [CV_Paper_Portal](https://hongsong-wang.github.io/CV_Paper_Portal/)**
* 🚨 **[Abstract Paper Portal of ICLR 2026](https://hongsong-wang.github.io/ICLR2026_Abstract/)**

## Installation
> **Recommended environment:** Linux system with an NVIDIA GPU.  
> The project has not been fully tested on other operating systems or hardware configurations.  
> <span style="color:#1E90FF;">*Running our method currently needs around 6 GB of GPU memory. We will continue to optimize memory efficiency.*</span>

To install the required libraries, simply run the following command:
```bash
conda env create -f environment.yaml
conda activate geodrag
```

Next, download the pretrained **Depth Anything V2** weights by running:
```bash
bash download.sh model
```

## Run GeoDrag
### Gradio demo
From the command line, run the following to launch the Gradio user interface:
```bash
python3 app.py
```

You can refer to [GIF above](https://github.com/Yujun-Shi/DragDiffusion/blob/main/release-doc/asset/github_video.gif) for a step-by-step demonstration of how to use the UI.

### CLI demo
To run GeoDrag directly from the command line, use:
```bash
python3 inference_drag.py \
    -c path/to/config \
    -i path/to/image \
    -m path/to/meta-data \
    -o path/to/output-dir
  
# example
# python3 inference_drag.py \
#     -c configs/base_configs.yaml \
#     -i datasets/DragBench/animals/JH_2023-09-14-1820-16/original_image.png \
#     -m datasets/DragBench/animals/JH_2023-09-14-1820-16/meta_data.pkl \
#     -o output

# -c: config file path
# -i: input image
# -m: metadata (drag points, mask, and prompt)
# -o: output directory
```

### DRAGBENCH demo
Download [DragBench](https://github.com/Yujun-Shi/DragDiffusion/releases/download/v0.1.1/DragBench.zip) into the folder "datasets" by running:
```bash
bash download.sh dragbench
```

To evaluate GeoDrag on the **DRAGBENCH** benchmark, run the following command:
```bash
python3 inference_dragbench.py \
    -c path/to/config \
    -b path/to/benchmark \
    -o path/to/output

# example
# python3 inference_dragbench.py \
#     -c configs/base_configs.yaml \
#     -b datasets/DragBench
#     -o outputs
```

## Quantitative evaluation
### Single-result evaluation
To compute quantitative evaluation metrics on a single result, run:
```bash
python3 -m evaluation single \
    -s '["MD", "IF", "DAI"]' \
    -i path/to/image \
    -m path/to/meta-data \
    -e path/to/edited-image 
  
# The `-s` argument specifies which scores to compute.
# You can choose any combination of the following:
#   MD  – Mean Distance
#   IF  – Image Fidelity
#   DAI – Dragging Accuracy Index
# For example:
#   -s '["MD"]'
#   -s '["MD", "IF"]'
#   -s '["MD", "IF", "DAI"]'
```

### DRAGBENCH evaluation
To directly compute quantitative results on the DRAGBENCH benchmark, run:
```bash
python3 -m evaluation benchmark \
    -b path/to/benchmark-results
```
Results will be summarized and printed in the terminal.

## Explanation for parameters
See [parameters.md](docs/parameters.md) for additional information.

## BibTeX
If you find our repo helpful, please consider leaving a star or cite our paper :)
```bibtex
@article{pu2025geodrag,
  title={Dragging with Geometry: From Pixels to Geometry-Guided Image Editing}, 
  author={Xinyu Pu and Hongsong Wang and Jie Gui and Pan Zhou},
  journal={arXiv preprint arXiv:2509.25740},
  year={2025}
}
```

## Contact
For any questions on this project, please contact [Xinyu](https://xinyu-pu.github.io/) (xinyupu@seu.edu.cn)

## Acknowledgement
The code is built builds upon DragDiffusion, FastDrag and diffusers, thanks for their outstanding work!    
We’d also like to express our appreciation to the amazing open-source community behind diffusion models, libraries, and research that inspired this work.

## Related Links
* [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)
* [DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing](https://yujun-shi.github.io/projects/dragdiffusion.html)
* [Emergent Correspondence from Image Diffusion](https://diffusionfeatures.github.io/)
* [DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models](https://mc-e.github.io/project/DragonDiffusion/)
* [FreeDrag: Feature Dragging for Reliable Point-based Image Editing](https://lin-chen.site/projects/freedrag/)
* [FastDrag: Manipulate Anything in One Step](https://fastdrag-site.github.io/)
* [Drag Your Noise: Interactive Point-based Editing  via Diffusion Semantic Propagation](https://github.com/haofengl/DragNoise)
* [Lightning fast text-guided image editing via one-step diffusion](https://lightning-drag.github.io/)
