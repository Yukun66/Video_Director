import os
import json
import argparse
from datetime import datetime
from omegaconf import OmegaConf
import gc
import torch
import torchvision.transforms as transforms
from einops import rearrange, repeat
import numpy as np
import imageio
import pickle
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from videodirector.models.unet import UNet3DConditionModel
from videodirector.pipelines.pipeline import VideoDirectorPipeline
from videodirector.pipelines.additional_components import customized_step, set_timesteps, customized_step_with_grad
from videodirector.utils.util import load_weights
from videodirector.utils.util import video_preprocess
from videodirector.utils.util import set_all_seed
from videodirector.utils.util import save_video

# @torch.no_grad()
def main(args):

    if not os.path.exists("inversion"):
        os.makedirs("inversion")
    
    config  = OmegaConf.load(args.config)
    adopted_dtype = torch.bfloat16
    device = "cuda"
    
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device).to(dtype=adopted_dtype)
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device).to(dtype=adopted_dtype)
    
    config.W = config.get("W", args.W)
    config.H = config.get("H", args.H)
    config.L = config.get("L", args.L)

    model_config = OmegaConf.load(config.get("model_config", args.model_config))
    model_config.unet_additional_kwargs["motion_module_kwargs"]["multiframe_NT"] = config.multiframe_NT
    unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(model_config.unet_additional_kwargs),).to(device).to(dtype=adopted_dtype)
    
    controlnet = None
    # set xformers
    if is_xformers_available() and (not args.without_xformers):
        unet.enable_xformers_memory_efficient_attention()
        if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()

    pipeline = VideoDirectorPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=controlnet,
        scheduler=DDIMScheduler(**OmegaConf.to_container(model_config.noise_scheduler_kwargs)),
    ).to(device)
    pipeline.scheduler.customized_step = customized_step.__get__(pipeline.scheduler)
    pipeline.scheduler.customized_step_with_grad = customized_step_with_grad.__get__(pipeline.scheduler)
    pipeline.scheduler.added_set_timesteps = set_timesteps.__get__(pipeline.scheduler)
    # config.num_inference_step = 1000
    pipeline.scheduler.added_set_timesteps(config.num_inference_step)
    
    pipeline = load_weights(
        pipeline,
        # motion module
        motion_module_path         = config.get("motion_module", ""),
        dreambooth_model_path      = config.get("dreambooth_path", ""),
    ).to(device)
    
    seed = config.get("seed", args.default_seed)
    set_all_seed(seed)
    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(seed)

    unet.eval()
    cond_video = video_preprocess(config)

    # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    eta = 0.0
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)
    
    ddim_latents, uncond_embeddings, score_list = pipeline.recon_guidance_pipe(
        video = cond_video, 
        config = config,
        DDIM_inversion_CFG = False, 
        extra_step_kwargs = extra_step_kwargs
    )
    gc.collect()  
    torch.cuda.empty_cache()  
    new_denoise_latent = pipeline.editing_pipe(  
        ddim_latents, 
        uncond_embeddings, 
        generator, 
        config, 
        score_list
    ) 


    # Post-processing
    videos = pipeline.decode_latents(new_denoise_latent)
    videos = rearrange(videos, "b c f h w -> b f h w c")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(config.video_path))[0]
    target_name = config.p2p_eq_params_words[0].strip().split()[0]  # 取第一个单词
    save_path = os.path.join(
        args.save_dir,
        f"{video_name}_to_"
        f"_{target_name}_{current_time}.mp4"
    )
    
    # save edited video
    save_video(videos[0], save_path, fps=8)
    print("edited video saved path: ", save_path)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path",   type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--model-config",            type=str, default="configs/model_config/model_config.yaml")    
    parser.add_argument("--config",                  type=str, default="configs/example.yaml")
    # parser.add_argument("--inversion_save_dir",      type=str, default="inversion/")
    parser.add_argument("--examples",                type=str, default=None)
    parser.add_argument("--save_dir",          type=str, default="samples/")
    
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--default-seed", type=int, default=42)

    parser.add_argument("--without-xformers", action="store_true")

    args = parser.parse_args()
    main(args)
