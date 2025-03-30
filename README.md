# VideoDirector: Precise Video Editing via Text-to-Video Models (CVPR2025)
Yukun Wang, 
Longguang Wang, 
Zhiyuan Ma, 
Qibin Hu, 
Kai Xu, 
Yulan Guo


[![arXiv](https://img.shields.io/badge/arXiv-VideoDirector-b31b1b.svg)](https://arxiv.org/abs/2411.17592) ![Pytorch](https://img.shields.io/badge/PyTorch->=2.0.0-Red?logo=pytorch)


### [<a href="https://yukun66.github.io/VideoDirector/" target="_blank">Project Page</a>]

## Edited results
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **Input Video** &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  ***Edited Results***
<td><img src="__assets__/output.gif"></td>


### Abstract
[//]: # (### Abstract)
>**VideoDirector** harness the powerful temporal generation capability of the text-to-video (T2V) model for precise video editing. VideoDirector produces results with high quality in terms of accuracy, fidelity, motion smoothness, and realism. For more see the [project webpage](https://yukun66.github.io/VideoDirector/).



## 🔧 Installations (python==3.11.3 recommended)

### Setup repository and conda environment

```
git clone https://github.com/Yukun66/Video_Director.git 
cd Video_Director

conda env create -f environment.yaml
conda activate videodirector
```

## 💡 Pretrained Model Preparations

### Download Stable Diffusion V1.5

Download Stable Diffusion, weights path is:
 ```
 models/StableDiffusion/stable-diffusion-v1-5
 ```

### Prepare Community Models

Manually download the community `.safetensors` models from [RealisticVision](https://civitai.com/models/4201?modelVersionId=130072).
Community checkpoints path is:
```
models/DreamBooth_LoRA/realisticVisionV60B1_v51VAE.safetensors
```

### Prepare AnimateDiff Motion Modules

Manually download the AnimateDiff modules from [AnimateDiff](https://github.com/guoyww/AnimateDiff). Save the modules to: 
```
models/Motion_Module
```
## 📌 Preprocess
### Mask prediction
We utilize the SAM2 model (https://github.com/facebookresearch/sam2) to generate masks for our method. The model is located in the SAM2_model directory and requires installation before use:


```
cd SAM2_model
pip install -e ".[demo]"
cd ..
```
We provide a **using example** to get mask of `resources/bear.mp4` in: `SAM2_model/notebooks/video_predictor_example.ipynb`.

## 🚗 Editing video
### Run our method:
```
bash run_editing.sh
```
### Config details
Our editing config file is in `editing_config_yaml/bear_editing_config.yaml`.
The config parameters are detailed below. 

<details>
  <summary> <b>Prompts</b></summary>

- **inversion_prompt**: original video description prompt. Example:
```
 "A brown bear, walking on rocky terrain, next to a stone wall."
```
- **new_prompt**: target video description prompt. Example:
```
"A tiger, walking on rocky terrain, next to a stone wall."
```
- **p2p_eq_params_words**: the new inserted words in new prompt. Example:
```
- tiger
```

</details>

<details>
  <summary> <b>STDG_guide</b></summary>

- Coefficient of STDG guidance. Example:
```
-STDG_guide:
 0.5
 0.5
 0.0
 0.5
```
</details>

<details>
  <summary> <b>p2p_self_replace_steps</b></summary>

- $\tau_s$ in paper Sec 3.3. Example:
```
p2p_self_replace_steps: 0.4
```
</details>

<details>
  <summary> <b>p2p_cross_replace_steps</b></summary>

- $\tau_c$ in paper Sec 3.3. Example:
```
p2p_cross_replace_steps: 0.8
```
</details>