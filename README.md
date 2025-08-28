<h3 align="center">
    <img src="assets/uso.webp" alt="Logo" style="vertical-align: middle; width: 95px; height: auto;">
    </br>
    Unified Style and Subject-Driven Generation via Disentangled and Reward Learning
</h3>

<p align="center"> 
<a href="https://bytedance.github.io/USO/"><img alt="Build" src="https://img.shields.io/badge/Project%20Page-USO-blue"></a> 
<a href="https://arxiv.org/abs/2508.18966"><img alt="Build" src="https://img.shields.io/badge/Tech%20Report-USO-b31b1b.svg"></a>
<a href="https://huggingface.co/bytedance-research/USO"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=green"></a>
<a href="https://huggingface.co/spaces/bytedance-research/USO"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=demo&color=orange"></a>
</p>
</p>

><p align="center"> <span style="color:#137cf3; font-family: Gill Sans">Shaojin Wu,</span><sup></sup></a>  <span style="color:#137cf3; font-family: Gill Sans">Mengqi Huang,</span></a> <span style="color:#137cf3; font-family: Gill Sans">Yufeng Cheng,</span><sup></sup></a>  <span style="color:#137cf3; font-family: Gill Sans">Wenxu Wu,</span><sup></sup> </a> <span style="color:#137cf3; font-family: Gill Sans">Jiahe Tian,</span><sup></sup></a> <span style="color:#137cf3; font-family: Gill Sans">Yiming Luo,</span><sup></sup></a> <span style="color:#137cf3; font-family: Gill Sans">Fei Ding,</span></a> <span style="color:#137cf3; font-family: Gill Sans">Qian He</span></a> <br> 
><span style="font-size: 13.5px">UXO Team</span><br> 
><span style="font-size: 12px">Intelligent Creation Lab, Bytedance</span></p>

<p align="center">
    <img src="assets/teaser.webp" width="1024"/>
<p>

## 🔥 News
* **2025.08.28** 🔥 The [demo](https://huggingface.co/spaces/bytedance-research/USO) of USO is released. Try it Now! ⚡️
* **2025.08.28** 🔥 Update fp8 mode as a primary low vmemory usage support. Gift for consumer-grade GPU users. The peak Vmemory usage is ~16GB now.
* **2025.08.27** 🔥 The [inference code](https://github.com/bytedance/USO) and [model](https://huggingface.co/bytedance-research/USO) of USO are released.
* **2025.08.27** 🔥 The [project page](https://bytedance.github.io/USO) of USO is created.
* **2025.08.27** 🔥 The [technical report](https://arxiv.org/abs/2508.18966) of USO is released.

## 📖 Introduction
Existing literature typically treats style-driven and subject-driven generation as two disjoint tasks: the former prioritizes stylistic similarity, whereas the latter insists on subject consistency, resulting in an apparent antagonism. We argue that both objectives can be unified under a single framework because they ultimately concern the disentanglement and re-composition of “content” and “style”, a long-standing theme in style-driven research. To this end, we present USO, a Unified framework for Style driven and subject-driven GeneratiOn. First, we construct a large-scale triplet dataset consisting of content images, style images, and their corresponding stylized content images. Second, we introduce a disentangled learning scheme that simultaneously aligns style features and disentangles content from style through two complementary objectives, style-alignment training and content–style disentanglement training. Third, we incorporate a style reward-learning paradigm to further enhance the model’s performance.

## ⚡️ Quick Start

### 🔧 Requirements and Installation

Install the requirements
```bash
## create a virtual environment with python >= 3.10 <= 3.12, like
python -m venv uso_env
source uso_env/bin/activate
## or
conda create -n uso_env python=3.10 -y
conda activate uso_env
## then install the requirements by you need
pip install -r requirements.txt # legacy installation command
```

Then download checkpoints in one of the following ways:
- **Suppose you already have some of the checkpoints**
```bash
# 1. download USO official checkpoints
pip install huggingface_hub
huggingface-cli download bytedance-research/USO --local-dir <YOU_SAVE_DIR> --local-dir-use-symlinks False

# 2. Then set the environment variable for FLUX.1 base model
export AE="YOUR_AE_PATH"
export FLUX_DEV="YOUR_FLUX_DEV_PATH"
export T5="YOUR_T5_PATH"
export CLIP="YOUR_CLIP_PATH"
# or export HF_HOME="YOUR_HF_HOME"

# 3. Then set the environment variable for USO
export LORA="<YOU_SAVE_DIR>/uso_flux_v1.0/dit_lora.safetensors"
export PROJECTION_MODEL="<YOU_SAVE_DIR>/uso_flux_v1.0/projector.safetensors"
```
- Directly run the inference scripts, the checkpoints will be downloaded automatically by the `hf_hub_download` function in the code.

### ✍️ Inference
* Start from the examples below to explore and spark your creativity. ✨
```bash
# the first image is a content reference, and the rest are style references.

# for subject-driven generation
python inference.py --prompt "The man in flower shops carefully match bouquets, conveying beautiful emotions and blessings with flowers. " --image_paths "assets/gradio_examples/identity1.jpg" --width 1024 --height 1024
# for style-driven generation
# please keep the first image path empty
python inference.py --prompt "A cat sleeping on a chair." --image_paths "" "assets/gradio_examples/style1.webp" --width 1024 --height 1024
# for style-subject driven generation (or set the prompt to empty for layout-preserved generation)
python inference.py --prompt "The woman gave an impassioned speech on the podium." --image_paths "assets/gradio_examples/identity2.webp" "assets/gradio_examples/style2.webp" --width 1024 --height 1024
# for multi-style generation
# please keep the first image path empty
python inference.py --prompt "A handsome man." --image_paths "" "assets/gradio_examples/style3.webp" "assets/gradio_examples/style4.webp" --width 1024 --height 1024
```
* You can also compare your results with the results in the `assets/gradio_examples` folder.

* For more examples, visit our [project page](https://bytedance.github.io/USO) or try the live [demo](https://huggingface.co/spaces/bytedance-research/USO).

### 🌟 Gradio Demo

```bash
python app.py
```

**For low vmemory usage**, please pass the `--offload` and `--name flux-dev-fp8` args. The peak memory usage will be 16GB (Single reference) ~ 18GB (Multi references).

```bash
python app.py --offload --name flux-dev-fp8
```

## 📄 Disclaimer
<p>
  We open-source this project for academic research. The vast majority of images 
  used in this project are either generated or from open-source datasets. If you have any concerns, 
  please contact us, and we will promptly remove any inappropriate content. 
  Our project is released under the Apache 2.0 License. If you apply to other base models, 
  please ensure that you comply with the original licensing terms. 
  <br><br>This research aims to advance the field of generative AI. Users are free to 
  create images using this tool, provided they comply with local laws and exercise 
  responsible usage. The developers are not liable for any misuse of the tool by users.</p>

## 🚀 Updates
For the purpose of fostering research and the open-source community, we plan to open-source the entire project, encompassing training, inference, weights, dataset etc. Thank you for your patience and support! 🌟
- [x] Release technical report.
- [x] Release github repo.
- [x] Release inference code.
- [x] Release model checkpoints.
- [x] Release huggingface space demo.
- Release training code.
- Release dataset.

##  Citation
If USO is helpful, please help to ⭐ the repo.

If you find this project useful for your research, please consider citing our paper:
```bibtex
@article{wu2025uso,
    title={USO: Unified Style and Subject-Driven Generation via Disentangled and Reward Learning},
    author={Shaojin Wu and Mengqi Huang and Yufeng Cheng and Wenxu Wu and Jiahe Tian and Yiming Luo and Fei Ding and Qian He},
    year={2025},
    eprint={2508.18966},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
}
```