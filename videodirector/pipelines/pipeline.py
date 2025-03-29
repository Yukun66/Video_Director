# Adapted from https://github.com/LPengYang/MotionClone/blob/main/motionclone/pipelines/pipeline_animation.py
import inspect
from typing import Callable, List, Optional, Union, Any, Dict
from dataclasses import dataclass
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler

import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import omegaconf
from omegaconf import OmegaConf
import einops
import imageio
import matplotlib.pyplot as plt
import yaml
import gc  
from ..models.attention import make_controller, regiter_crossattn_editor_diffusers_p2p, regiter_selfattn_editor_diffusers_p2p

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..models.sparse_controlnet import SparseControlNetModel
import pdb

from ..utils.xformer_attention import *
from ..utils.conv_layer import *
from ..utils.util import *
from ..utils.util import _in_step, _classify_blocks, ddim_inversion

from .additional_components import *

from torch.optim.adam import Adam
import torch.nn.functional as nnf
from einops import rearrange, repeat
from PIL import Image, ImageDraw
from skimage import measure
from skimage.draw import ellipse

# from ..models.attention import regiter_selfattn_editor_diffusers
# from ..models.attention import regiter_crossattn_editor_diffusers
# from ..models.attention import regiter_tempattn_editor_diffusers
from ..models.attention import MutualSelfAttention_p2p

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class VideoDirectorPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class VideoDirectorPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        controlnet: Union[SparseControlNetModel, None] = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)
        

        
    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def invert(self,
               video = None,
               config: omegaconf.dictconfig = None,
               save_path = None,
               ):
        # perform DDIM inversion 
        import time
        start_time = time.time()
        generator = None
        video_latent = self.vae.encode(video.to(self.vae.dtype).to(self.vae.device)).latent_dist.sample(generator)
        video_latent = self.vae.config.scaling_factor * video_latent
        video_latent = video_latent.unsqueeze(0)
        video_latent = einops.rearrange(video_latent, "b f c h w -> b c f h w")                                                                 
        ddim_latents_dict, cond_embeddings = ddim_inversion(self, self.scheduler, video_latent, config.num_inference_step, config.inversion_prompt)
        
        end_time = time.time()
        # import pdb; pdb.set_trace()
        print("Inversion time", end_time - start_time)

        video_data: Dict = {
            'inversion_prompt': config.inversion_prompt,
            'all_latents_inversion': ddim_latents_dict,
            'raw_video': video,
            'inversion_prompt_embeds': cond_embeddings,
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(video_data, f)
                
    # @torch.no_grad()
    def recon_guidance_pipe(self,
               video = None,
               config: omegaconf.dictconfig = None,
               save_path = None,
               DDIM_inversion_CFG = True,
               extra_step_kwargs = None
               ):
        # perform DDIM inversion 
        import time
        start_time = time.time()
        # generator = None
        video_latent = self.vae.encode(video.to(self.vae.dtype).to(self.vae.device)).latent_dist.sample(extra_step_kwargs['generator']).clone().detach()
        video_latent = self.vae.config.scaling_factor * video_latent
        video_latent = video_latent.unsqueeze(0)
        video_latent = einops.rearrange(video_latent, "b f c h w -> b c f h w")  
        
        print("DDIM inversion...")  
        ddim_latents, context = ddim_inversion_with_context(self, self.scheduler, video_latent, config)


        print("Null-text optimization...") 
        uncond_embeddings, STDG_list = self.null_optimization(ddim_latents, config, context, extra_step_kwargs)
        # config.null_inner_steps, config.early_stop_epsilon, config.num_inference_step
        
        return ddim_latents, uncond_embeddings, STDG_list
    
    # STDG ：
    def calculate_STDG(self, latent_cur_GT, latent_cur, i, step_t, cond_embeddings,  config):
        with torch.no_grad():
            # GT for temp_attn and spatial_attn (Sec 3.2, Fig 3)
            # cond_embedding，and DDIM inv latents input unet
            _ = get_noise_pred_single(latent_cur_GT, step_t, cond_embeddings, self.unet) 
            temp_attn_prob_GT = self.get_temp_attn_prob()
            if self.input_config.app_guidance.block_type =="temp": 
                attn_key_GT = self.get_temp_attn_key()
            else:
                attn_key_GT = self.get_spatial_attn1_key()

        weight_each_motion = torch.tensor(config.temp_guidance.weight_each).to(latent_cur_GT.dtype).to(latent_cur_GT.device) 
        weight_each_motion = torch.repeat_interleave(weight_each_motion/100.0, repeats=6) 
        #  the 100.0 here is only to avoid numberical overflow under float16
        weight_each_app = torch.tensor(config.app_guidance.weight_each).to(latent_cur_GT.dtype).to(latent_cur_GT.device) # config.app_guidance.weight_each
        if self.input_config.app_guidance.block_type == "temp":
            weight_each_app = torch.repeat_interleave(weight_each_app/100.0, repeats=6) 
        else:
            weight_each_app = torch.repeat_interleave(weight_each_app/100.0, repeats=3)
        latent_cur.requires_grad = True  
        _ = get_noise_pred_single(latent_cur, step_t, cond_embeddings, self.unet) 
        temp_attn_prob = self.get_temp_attn_prob()

        if self.input_config.app_guidance.block_type =="temp": 
            attn_key = self.get_temp_attn_key()
        else:
            attn_key = self.get_spatial_attn1_key()

        # global temporal guidance:
        loss_motion = compute_temp_loss(temp_attn_prob_GT, temp_attn_prob, weight_each_motion.detach(), None)
        # global_app_guidance:
        loss_appearance = compute_semantic_loss(attn_key_GT,attn_key, weight_each_app.detach(),
                                                None, None, None,None,block_type=self.input_config.app_guidance.block_type) 
        loss_total = 100.0*(loss_motion + loss_appearance) 
        
        # gradient of loss_total about latent_cur:
        gradient = torch.autograd.grad(loss_total, latent_cur, allow_unused=True)[0] 

        assert gradient is not None, f"Step {i}: grad is None"
        grad_guidance_threshold = None
        if grad_guidance_threshold is not None: # self.input_config.grad_guidance_threshold 
            threshold = self.input_config.grad_guidance_threshold
            gradient_clamped = torch.where(
                    gradient.abs() > threshold,
                    torch.sign(gradient) * threshold,
                    gradient
                )
            STDG = gradient_clamped.detach()
        else:
            STDG = gradient.detach()
        latent_cur.requires_grad = False  
        del gradient, loss_total, loss_motion, loss_appearance, temp_attn_prob, attn_key, temp_attn_prob_GT, attn_key_GT 
        gc.collect()  
        torch.cuda.empty_cache()  
        return STDG

    # read mask:
    def _read_sam_mask(self, mask_dir, device):
        file_names = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        masks = []

        # mask -> bool
        for file_name in file_names:
            image_path = os.path.join(mask_dir, file_name)
            image = Image.open(image_path).convert('L')  
            image_array = np.array(image)
            # 1->True,0->False
            bool_array = image_array > 0  
            tensor = torch.from_numpy(bool_array).unsqueeze(0).to(device) 
            masks.append(tensor)
        stacked_masks = torch.stack(masks) 
        return stacked_masks
    
    # read mask with expanded ellipse:
    def _read_sam_mask_with_ellipse(self, mask_dir, device):
        output_dir = os.path.join(mask_dir, 'expanded_mask')
        os.makedirs(output_dir, exist_ok=True)
        file_names = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        masks = []

        # mask -> bool
        for file_name in file_names:
            image_path = os.path.join(mask_dir, file_name)
            image = Image.open(image_path).convert('L')  
            image_array = np.array(image)
            
            # calculate ellipse:
            _, ellipsoid_mask = self.generate_ellipsoid_mask(image_array, device)
            # merge original mask and ellipsoid_mask:
            merged_mask = np.logical_or(ellipsoid_mask, image_array)
            output_image_path = os.path.join(output_dir, f"merged_{file_name}")
            self.save_ellipse_image(merged_mask, output_image_path)
            merged_tensor = torch.from_numpy(merged_mask).unsqueeze(0).to(device)
            masks.append(merged_tensor)
            
        stacked_masks = torch.stack(masks)  # [N, 1, H, W]
        return stacked_masks

    # use scikit-image calculate ellipse mask:
    def generate_ellipsoid_mask(self, image_array, device):
        labeled_mask = measure.label(image_array)
        regions = measure.regionprops(labeled_mask)
        ellipsoid_mask = np.zeros_like(image_array, dtype=np.uint8)

        if len(regions) > 0:
            region = regions[0]

            # calculate ellipse:
            minr, minc, maxr, maxc = region.bbox
            center_y, center_x = region.centroid  # Ellipse center  
            major_axis_length = region.major_axis_length / 2  # Semi-major axis length  
            minor_axis_length = region.minor_axis_length / 2  # Semi-minor axis length  
            orientation = np.degrees(region.orientation)  # Rotation angle, converted to degrees  

            rr, cc = ellipse(int(center_y), int(center_x), 
                             int(major_axis_length), int(minor_axis_length),
                             rotation=np.radians(orientation), shape=image_array.shape)
            ellipsoid_mask[rr, cc] = 1  # fill the ellipse
        ellipsoid_tensor = torch.from_numpy(ellipsoid_mask > 0).unsqueeze(0).to(device)
        return ellipsoid_tensor, ellipsoid_mask  

    def save_ellipse_image(self, ellipsoid_mask, output_image_path):
        image = Image.fromarray((ellipsoid_mask * 255).astype(np.uint8))
        image.save(output_image_path)
    
    
    # temporal STDG ：
    def calculate_STDG_motion(self, latent_cur_GT, latent_cur, i, step_t, cond_embeddings, config, mask):
        with torch.no_grad():
            # GT for temp_attn (Sec 3.2, Fig 3)
            # cond_embedding，and DDIM inv latents input unet
            _ = get_noise_pred_single(latent_cur_GT, step_t, cond_embeddings, self.unet)
            temp_attn_prob_GT = self.get_temp_attn_prob()

        weight_each_motion = torch.tensor(config.temp_guidance.weight_each).to(latent_cur_GT.dtype).to(latent_cur_GT.device)
        weight_each_motion = torch.repeat_interleave(weight_each_motion/100.0, repeats=6) 
        #  the 100.0 here is only to avoid numberical overflow under float16
        
        latent_cur.requires_grad = True  
        _ = get_noise_pred_single(latent_cur, step_t, cond_embeddings, self.unet) 
        temp_attn_prob = self.get_temp_attn_prob()
        # global temporal guidance:
        loss_motion = compute_temp_loss_with_mask(
            temp_attn_prob_GT, 
            temp_attn_prob, 
            weight_each_motion.detach(), 
            mask
        )
        loss = 100.0*(loss_motion)
        # gradient of loss about latent_cur:
        gradient_motion = torch.autograd.grad(loss, latent_cur, allow_unused=True)[0]
        assert gradient_motion is not None, f"Step {i}: grad is None"
        score_motion = gradient_motion.detach()
        latent_cur.requires_grad = False 
        
        gc.collect() 
        torch.cuda.empty_cache() 
        return score_motion

    # appearance STDG ：
    def calculate_STDG_appearance(self, latent_cur_GT, latent_cur, i, step_t, cond_embeddings, config, sam_mask):
        with torch.no_grad():
            # GT for temp_attn (Sec 3.2, Fig 3)
            # cond_embedding，and DDIM inv latents input unet
            _ = get_noise_pred_single(latent_cur_GT, step_t, cond_embeddings, self.unet) 
            if self.input_config.app_guidance.block_type =="temp": 
                attn_key_GT = self.get_temp_attn_key()
            else:
                attn_key_GT = self.get_spatial_attn1_key()
        
        weight_each_app = torch.tensor(config.app_guidance.weight_each).to(latent_cur_GT.dtype).to(latent_cur_GT.device) 
        if self.input_config.app_guidance.block_type == "temp":
            weight_each_app = torch.repeat_interleave(weight_each_app/100.0, repeats=6) 
        else:
            weight_each_app = torch.repeat_interleave(weight_each_app/100.0, repeats=3)
        latent_cur.requires_grad = True
        _ = get_noise_pred_single(latent_cur, step_t, cond_embeddings, self.unet) 
        if self.input_config.app_guidance.block_type =="temp": 
            attn_key = self.get_temp_attn_key()
        else:
            attn_key = self.get_spatial_attn1_key()
        # global appearance guidance:
        loss_appearance = compute_semantic_loss_with_mask(
            attn_key_GT, 
            attn_key, 
            weight_each_app.detach(),
            sam_mask, 
            block_type=self.input_config.app_guidance.block_type
        )
        loss = 100.0*(loss_appearance) 
        # gradient of loss about latent_cur:
        gradient_appearance = torch.autograd.grad(loss, latent_cur, allow_unused=True)[0] 
        assert gradient_appearance is not None, f"Step {i}: grad is None"

        # STDG = gradient.detach()
        score_appearance = gradient_appearance.detach()

        latent_cur.requires_grad = False 
        
        gc.collect()  
        torch.cuda.empty_cache() 
        return score_appearance


    def null_optimization(self, latents, config, context, extra_step_kwargs):
        # assert config is not None, "config is required for FreeControl pipeline"
        if not hasattr(self, 'config'):
            setattr(self, 'input_config', config)
        self.input_config = config
        if not hasattr(self, 'video_name'):
            setattr(self, 'video_name', config.video_path.split('/')[-1].split('.')[0])
        self.video_name = config.video_path.split('/')[-1].split('.')[0]

        self.unet = prep_unet_attention(self.unet)
        self.unet = prep_unet_conv(self.unet)

        ## read mask
        mask_dir = os.path.join(os.path.join(os.path.dirname(config.video_path), "sam2_mask"), self.video_name)
        if config.using_ellipse_mask:
            sam_mask = self._read_sam_mask_with_ellipse(mask_dir, self.unet.device) 
        else:
            sam_mask = self._read_sam_mask(mask_dir, self.unet.device) 
        self.sam_mask = sam_mask
        
        video_length = latents[0].shape[2] # frame
        uncond_embeddings, cond_embeddings = context.chunk(2)
        # multiframe_NT:
        if config.multiframe_NT:
            uncond_embeddings = repeat(uncond_embeddings, 'b n c -> (b f) n c', f=video_length)  
            cond_embeddings = repeat(cond_embeddings, 'b n c -> (b f) n c', f=video_length)  

        uncond_embeddings_list = []
        STDG_list = []
        STDG = None
        STDG_motion_fore = None
        STDG_motion_back = None
        STDG_appearance_fore = None
        STDG_appearance_back = None

        latent_cur = latents[-1]
        bar = tqdm(total=config.null_inner_steps * config.num_inference_step)

        for i in range(config.num_inference_step):
            uncond_embeddings = uncond_embeddings.clone().detach() 
            latent_cur = latent_cur.clone().detach() 
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / float(2*config.num_inference_step)))
            latent_prev = latents[len(latents) - i - 2]
            latent_prev = latent_prev.clone().detach()
            t = self.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = get_noise_pred_single(latent_cur, t, cond_embeddings, self.unet)
            
            # STDG：
            latent_cur_GT = latents[len(latents) - i - 1]
            STDG_motion_fore = self.calculate_STDG_motion(latent_cur_GT, latent_cur, i, t, cond_embeddings, config, sam_mask)
            STDG_motion_back = self.calculate_STDG_motion(latent_cur_GT, latent_cur, i, t, cond_embeddings, config, ~sam_mask)
            STDG_appearance_fore = self.calculate_STDG_appearance(latent_cur_GT, latent_cur, i, t, cond_embeddings, config, sam_mask)
            STDG_appearance_back = self.calculate_STDG_appearance(latent_cur_GT, latent_cur, i, t, cond_embeddings, config, ~sam_mask)
            
            STDG = (config.score_guide[0] * STDG_motion_fore + 
                        config.score_guide[1] * STDG_motion_back + 
                        config.score_guide[2] * STDG_appearance_fore + 
                        config.score_guide[3] * STDG_appearance_back)
            STDG_list.append(STDG)

            for j in range(config.null_inner_steps):
                noise_pred_uncond = get_noise_pred_single(
                    latent_cur, t, 
                    uncond_embeddings, 
                    self.unet
                )
                
                noise_pred = noise_pred_cond + config.cfg_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.scheduler.customized_step_with_grad(
                    noise_pred, t, 
                    latent_cur, 
                    score=STDG,
                    guidance_scale=self.input_config.grad_guidance_scale,
                    indices=[0],
                    **extra_step_kwargs, 
                    return_dict=False
                )[0] 

                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                del latents_prev_rec, noise_pred 
                gc.collect() 
                torch.cuda.empty_cache()  
                optimizer.step()
                assert not torch.isnan(uncond_embeddings.abs().mean())
                loss_item = loss.item()
                bar.update()
                #
                if loss_item < config.early_stop_epsilon + i * 2e-5:
                    break
            
            for j in range(j + 1, config.null_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings.detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])

                latents_input = torch.cat([latent_cur] * 2)
                assert context is not None
                noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                noise_pred = noise_prediction_text + config.cfg_scale * (noise_prediction_text - noise_pred_uncond)
                latent_cur = self.scheduler.customized_step(
                    noise_pred, t, 
                    latent_cur, 
                    score=STDG, 
                    guidance_scale=self.input_config.grad_guidance_scale,
                    indices=[0],
                    **extra_step_kwargs, 
                    return_dict=False
                )[0].detach()
        bar.close()
        return uncond_embeddings_list, STDG_list    

    # editing latent
    def editing_pipe(
        self, 
        ddim_latents, 
        uncond_embeddings, 
        generator, 
        config, 
        STDG_list, 
        eta: float = 0.0,
    ):
        with torch.no_grad():
            assert uncond_embeddings is not None, "editing_pipe() needs uncond_embeddings!!!"
            batch_size = 1
            num_videos_per_prompt = 1
            num_channels_latents = self.unet.in_channels
            device = self._execution_device
            negative_prompt = config.negative_prompt 

            # perform classifier_free_guidance in default
            cfg_scale = config.cfg_scale or 7.5
            do_classifier_free_guidance = True

            # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
            with_uncond_embedding = do_classifier_free_guidance if uncond_embeddings is None else False
            # target text embedding
            new_text_embeddings = self._encode_prompt(
                config.new_prompt, 
                device, 
                num_videos_per_prompt, 
                with_uncond_embedding, 
                negative_prompt,
            )
            # original text embedding
            invert_text_embeddings = self._encode_prompt(
                config.inversion_prompt, 
                device, 
                num_videos_per_prompt, 
                True, 
                negative_prompt,
            )

            video_length = ddim_latents[-1].shape[2]
            
            if config.multiframe_NT:
                invert_text_embeddings = repeat(invert_text_embeddings, 'b n c -> (b f) n c', f=video_length)  
                new_text_embeddings = repeat(new_text_embeddings, 'b n c -> (b f) n c', f=video_length)  

            # if config.MutualAttn_p2p: 
            prompts = [self.input_config.inversion_prompt,
                        self.input_config.new_prompt
                    ]

            cross_replace_steps = {'default_': config.p2p_cross_replace_steps, }
            cross_replace_layers = config.p2p_cross_replace_layers
            self_replace_steps = config.p2p_self_replace_steps
            if config.p2p_blend_word_base is not None and config.p2p_blend_word_new is not None:
                blend_word = (((config.p2p_blend_word_base,), (config.p2p_blend_word_new,)))
            else:
                blend_word = None
            eq_params = {"words": tuple(config.p2p_eq_params_words), "values": tuple(config.p2p_eq_params_values)} 
            controller = make_controller(config, self.tokenizer,self.unet.device, 
                                            prompts, config.p2p_cross_is_replace_controller, 
                                            cross_replace_steps, 
                                            cross_replace_layers, 
                                            self_replace_steps, blend_word, eq_params)

            
            regiter_crossattn_editor_diffusers_p2p(self.unet, controller)
            SELF_START_STEP = self.input_config.MutualSelfAttn_steps[0]
            SELF_END_STEP = self.input_config.MutualSelfAttn_steps[1]
            SELF_START_LAYER = self.input_config.MutualSelfAttn_layers[0]
            SELF_END_LAYER = self.input_config.MutualSelfAttn_layers[1]
            selfattn_editor_p2p = MutualSelfAttention_p2p(config, self_replace_steps_p2p=self_replace_steps,
                                                            start_step=SELF_START_STEP,end_step=SELF_END_STEP, 
                                                            start_layer=SELF_START_LAYER, end_layer=SELF_END_LAYER,
                                                            sam_masks = self.sam_mask, num_frames=video_length)
            regiter_selfattn_editor_diffusers_p2p(self.unet, selfattn_editor_p2p)
            
                
            # Prepare timesteps
            self.scheduler.set_timesteps(config.num_inference_step, device=device)
            timesteps = self.scheduler.timesteps

            new_latent = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                config.H,
                config.W,
                invert_text_embeddings.dtype,
                device,
                generator,
                ddim_latents[-1],
            )

            # Denoising loop
            with self.progress_bar(total=config.num_inference_step) as progress_bar:
                if uncond_embeddings is not None:
                    start_time = config.num_inference_step
                    assert (timesteps[-start_time:] == timesteps).all()
                for i, step_t in enumerate(timesteps):
                    invert_latent = ddim_latents[len(ddim_latents)-i-1]
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([torch.cat([new_latent] * 2), torch.cat([invert_latent] * 2)]) if do_classifier_free_guidance else new_latent
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, step_t)
                    text_embeddings_input = torch.cat([torch.cat([uncond_embeddings[i], new_text_embeddings]), invert_text_embeddings]) 
                    noise_pred = self.unet(
                        latent_model_input, 
                        step_t, 
                        encoder_hidden_states=text_embeddings_input,
                    ).sample.to(dtype=new_latent.dtype)
                    
                    new_noise_pred = noise_pred[[1]] + cfg_scale * (noise_pred[[1]] - noise_pred[[0]]) # cfg 
                    if len(STDG_list)>0:
                        new_latent = self.scheduler.customized_step(new_noise_pred, step_t, new_latent, score=STDG_list[i],
                                                guidance_scale=self.input_config.grad_guidance_scale,
                                                indices=[0],
                                                **extra_step_kwargs, return_dict=False)[0].detach()
                        
                    else:
                        new_latent = self.scheduler.customized_step(new_noise_pred, step_t, new_latent, score=None,
                                                guidance_scale=self.input_config.grad_guidance_scale,
                                                indices=[0], 
                                                **extra_step_kwargs, return_dict=False)[0].detach()
                    
                    progress_bar.update()
            return new_latent 


# https://github.com/LPengYang/MotionClone/blob/main/motionclone/utils/motionclone_functions.py
    def get_temp_attn_prob(self,index_select=None):

        attn_prob_dic = {}
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if "VersatileAttention" in module_name and _classify_blocks(self.input_config.temp_guidance.blocks, name):
                key = module.processor.key
                if index_select is not None:
                    get_index = torch.repeat_interleave(torch.tensor(index_select), repeats=key.shape[0]//len(index_select))
                    index_all = torch.arange(key.shape[0])
                    index_picked = index_all[get_index.bool()]
                    key = key[index_picked]
                key = module.reshape_heads_to_batch_dim(key).contiguous()
                
                query = module.processor.query
                if index_select is not None:
                    query = query[index_picked]
                query = module.reshape_heads_to_batch_dim(query).contiguous()
                

                attention_probs = module.get_attention_scores(query, key, None)         
                attention_probs = attention_probs.reshape(-1, module.heads,attention_probs.shape[1], attention_probs.shape[2])
                
                attn_prob_dic[name] = attention_probs

        return attn_prob_dic
    
    def get_temp_attn_key(self,index_select=None):

        attn_key_dic = {}
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if "VersatileAttention" in module_name and _classify_blocks(self.input_config.app_guidance.blocks, name):
                key = module.processor.key
                if index_select is not None:
                    get_index = torch.repeat_interleave(torch.tensor(index_select), repeats=key.shape[0]//len(index_select))
                    index_all = torch.arange(key.shape[0])
                    index_picked = index_all[get_index.bool()]
                    key = key[index_picked]
                
                attn_key_dic[name] = key

        return attn_key_dic
        
    def get_spatial_attn1_key(self, index_select=None):
        attn_key_dic = {}
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if "Attention" in module_name and 'attn1' in name and 'attentions' in name and _classify_blocks(self.input_config.app_guidance.blocks, name):
                key = module.processor.key
                # [64,256,1280]
                if index_select is not None:
                    get_index = torch.repeat_interleave(torch.tensor(index_select), repeats=key.shape[0]//len(index_select))
                    index_all = torch.arange(key.shape[0])
                    index_picked = index_all[get_index.bool()]
                    key = key[index_picked]
                
                attn_key_dic[name] = key
                # [frame, H*W, head*dim] [16,256,1280]

        return attn_key_dic

