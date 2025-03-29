# VideoDirector: Precise Video Editing via Text-to-Video Models (CVPR2025)
## [<a href="https://yukun66.github.io/VideoDirector/" target="_blank">Project Page</a>]

[![arXiv](https://img.shields.io/badge/arXiv-VideoDirector-b31b1b.svg)](https://arxiv.org/abs/2411.17592) ![Pytorch](https://img.shields.io/badge/PyTorch->=1.10.0-Red?logo=pytorch)
<!-- [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/weizmannscience/tokenflow) -->
<!-- ![Pytorch](https://img.shields.io/badge/PyTorch->=1.10.0-Red?logo=pytorch) -->



[//]: # ([![Replicate]&#40;https://replicate.com/cjwbw/multidiffusion/badge&#41;]&#40;https://replicate.com/cjwbw/multidiffusion&#41;)

[//]: # ([![Hugging Face Spaces]&#40;https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue&#41;]&#40;https://huggingface.co/spaces/weizmannscience/text2live&#41;)




<!-- https://github.com/omerbt/TokenFlow/assets/52277000/93dccd63-7e9a-4540-a941-31962361b0bb -->


**TokenFlow** is a video editing method.

[//]: # (### Abstract)
>Despite the typical inversion-then-editing paradigm using text-to-image (T2I) models has demonstrated promising results, directly extending it to text-to-video (T2V) models still suffers severe artifacts
%significant deviations such as color flickering and content distortion. Consequently, current video editing methods primarily rely on T2I models, which inherently lack temporal-coherence generative ability, often resulting in inferior editing results. In this paper, we attribute the failure of the typical editing paradigm to: 1. **Tightly Spatial-temporal Coupling.** The vanilla pivotal-based inversion strategy struggles to disentangle spatial-temporal information in the video diffusion model; 2. **Complicated Spatial-temporal Layout.** The vanilla cross-attention control is deficient in preserving the unedited content. To address these limitations, we propose a spatial-temporal decoupled guidance~(**STDG**) and multi-frame null-text optimization strategy to provide pivotal temporal cues for more precise pivotal inversion. Furthermore, we introduce a self-attention control strategy to maintain higher fidelity for precise partial content editing. Experimental results demonstrate that our method (termed **VideoDirector**) effectively harnesses the powerful temporal generation capabilities of T2V models, producing edited videos with state-of-the-art performance in accuracy, motion smoothness, realism, and fidelity to unedited content. 

For more see the [project webpage](https://yukun66.github.io/VideoDirector/).

## Sample results
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **Input Video** &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ***Edited Results***
<td><img src="__assets__/output.gif"></td>



## Environment
```
conda create -n VideoDirector python=3.9
conda activate VideoDirector
pip install -r requirements.txt
```
## Preprocess
