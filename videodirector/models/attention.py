# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import FeedForward, AdaLayerNorm

from einops import rearrange, repeat
import math
from typing import Optional, Union, Tuple, List, Dict
from p2p_module import ptp_utils, seq_aligner

@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,

        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        multiframe_NT=None,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.multiframe_NT = multiframe_NT

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,

                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        # multiframe_NT
        if not self.multiframe_NT:
            encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)     

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,

        unet_use_cross_frame_attention = None,
        unet_use_temporal_attention = None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention

        # SC-Attn
        assert unet_use_cross_frame_attention is not None
        if unet_use_cross_frame_attention:
            self.attn1 = SparseCausalAttention2D(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn1 = SelfAttention( # CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        assert unet_use_temporal_attention is not None
        if unet_use_temporal_attention:
            self.attn_temp = TempAttention( #CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, op=None):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        # if self.only_cross_attention:
        #     hidden_states = (
        #         self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
        #     )
        # else:
        #     hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states

        # pdb.set_trace()
        if self.unet_use_cross_frame_attention:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states
        else:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask) + hidden_states

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        if self.unet_use_temporal_attention:
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            norm_hidden_states = (
                self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
            )
            hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states

class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False
        self.added_kv_proj_dim = added_kv_proj_dim

        #### add processer
        self.processor = None

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        self._slice_size = slice_size

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        # query = self.reshape_heads_to_batch_dim(query) # move backwards

        if self.added_kv_proj_dim is not None:
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

            ######record###### record before reshape heads to batch dim
            if self.processor is not None:
                self.processor.record_qkv(self, hidden_states, query, key, value, attention_mask)
            ##################

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

            key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            ######record######
            if self.processor is not None:
                self.processor.record_qkv(self, hidden_states, query, key, value, attention_mask)
            ##################

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

        query = self.reshape_heads_to_batch_dim(query) # reshape query

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        ######record######
        if self.processor is not None:
            self.processor.record_attn_mask(self, hidden_states, query, key, value, attention_mask)
        ##################

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def _attention(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim, attention_mask):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]

            if self.upcast_attention:
                query_slice = query_slice.float()
                key_slice = key_slice.float()

            attn_slice = torch.baddbmm(
                torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
                query_slice,
                key_slice.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )

            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[start_idx:end_idx]

            if self.upcast_softmax:
                attn_slice = attn_slice.float()

            attn_slice = attn_slice.softmax(dim=-1)

            # cast back to the original dtype
            attn_slice = attn_slice.to(value.dtype)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states
    
    def set_processor(self, processor: "AttnProcessor") -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor
        
    def get_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

       

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs


class SelfAttention(CrossAttention):
    pass
class TempAttention(CrossAttention):
    pass

# editor base class
class MutualAttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


    def __call__(self, q, k, v, attention_mask=None, batch_size=None, num_heads=None, scale=None):
        out = self.forward(query=q, key=k, value=v, attention_mask=attention_mask, batch_size=batch_size, heads=num_heads, scale=scale)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1

        return out

    def forward(self, query, key, value, attention_mask=None, batch_size=None, heads=None, scale=None):
        hidden_states = self._memory_efficient_attention_xformers(query=query, key=key, value=value, attention_mask=attention_mask, num_heads=heads)
        # Some versions of xformers return output in fp32, cast it back to the dtype of the input
        hidden_states = hidden_states.to(query.dtype)
        return hidden_states

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def reshape_batch_dim_to_heads(self, tensor, num_heads):
        batch_size, seq_len, dim = tensor.shape
        head_size = num_heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask, num_heads):
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states, num_heads)
        return hidden_states


class MutualSelfAttention_p2p(MutualAttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, config, self_replace_steps_p2p=0.2, 
                 start_step=4, end_step=100, start_layer=10, end_layer=16, 
                 sam_masks=None, num_frames=None, 
                 layer_idx=None, step_idx=None, total_steps=50, model_type="SD"): 
        super().__init__()
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.sam_masks = sam_masks
        self.num_frames = num_frames
        if type(self_replace_steps_p2p) is float:
            self_replace_steps_p2p = 0, self_replace_steps_p2p
        self.num_self_replace_p2p = (int(config.num_inference_step * self_replace_steps_p2p[0]), 
                                int(config.num_inference_step * self_replace_steps_p2p[1]))
        
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = self.num_self_replace_p2p[1] 
        self.end_step = end_step
        self.start_layer = start_layer
        self.layer_idx = list(range(start_layer, end_layer))
        self.step_idx = list(range(self.start_step, self.end_step)) 
        self.p2p_step_idx = list(range(self.num_self_replace_p2p[0], self.num_self_replace_p2p[1])) 
        print("p2p_MutualSelfAttention at denoising steps: ", self.p2p_step_idx) 
        print("MutualSelfAttention at denoising steps: ", self.step_idx)
        print("MutualSelfAttention at U-Net layers: ", self.layer_idx)

    # attn mask：
    def attn_mask_cal(self, masks, num_heads, H, W, dtype):
        masks = masks.masked_fill(masks == 1, float('-inf'))
        masks = masks.expand(1, (H*W), self.num_frames, H, W)
        attention_mask = rearrange(masks, "heads c frame h w -> (frame heads) c (h w)", frame=self.num_frames, h=H, w=W)
        target_attention_mask = torch.zeros_like(attention_mask)
        attention_mask = torch.cat([attention_mask, target_attention_mask], dim=-1)

        return attention_mask.to(dtype)
    
    
    def attn_batch(self, q, k, v, masks, num_heads, attention_mask):
        H = W = int(math.sqrt(q.shape[1])) # h*w
        masks = F.interpolate(masks.float(), size=(H, W), mode='nearest') # [f, 1, h, w] 
        masks = masks.permute(1, 0, 2, 3).unsqueeze(0)
        masks = masks.to(k.dtype)

        k_target = k[:, :H*W]
        k_source_bg = k[:, H*W:]
        v_target = v[:, :H*W]
        v_source_bg = v[:, H*W:]

        k_source_bg = rearrange(k_source_bg, "(frame n_head) (h w) c -> n_head c frame h w", frame=self.num_frames, h=H, w=W)
        k_source_bg = k_source_bg * (1-masks)
        k_source_bg = rearrange(k_source_bg, "n_head c frame h w -> (frame n_head) (h w) c", frame=self.num_frames, h=H, w=W)

        key = torch.cat([k_source_bg, k_target], dim=1) # torch.cat([k_source_fg, k_source_bg, k_target], dim=1)
        value = torch.cat([v_source_bg, v_target], dim=1) # torch.cat([v_source_fg, v_source_bg, v_target], dim=1)

        # attention_mask：
        attention_mask_converted = self.attn_mask_cal(masks, num_heads, H, W, q.dtype) # attention_mask shape: (B or 1, n_queries, number of keys) 
        batch_size, seq_len, dim = q.shape
        q = q.reshape(batch_size // num_heads, num_heads, seq_len, dim) # [(B H), M, K]->[B, H, M, K] B: batch size, M: sequence length, H:number of heads, K: embeding size per head
        q = q.permute(0, 2, 1, 3) #  [B, H, M, K]->[B, M, H, K]
        key = key.reshape(batch_size // num_heads, num_heads, 2*seq_len, dim) # [(B H), M, K]->[B, H, M, K] 
        key = key.permute(0, 2, 1, 3) #  [B, H, M, K]->[B, M, H, K]
        value = value.reshape(batch_size // num_heads, num_heads, 2*seq_len, dim) # [(B H), M, K]->[B, H, M, K] 
        value = value.permute(0, 2, 1, 3) #  [B, H, M, K]->[B, M, H, K]

        
        # because of mask，calculate hidden states at one time leads to OOM:
        hidden_states_heads = []
        for i in range(num_heads):
            # single head:
            q_single_head = q[:, :, i:i+1, :]  # i-th head
            k_single_head = key[:, :, i:i+1, :]
            v_single_head = value[:, :, i:i+1, :]
            # single head attention
            hidden_states_single_head = xformers.ops.memory_efficient_attention(q_single_head, k_single_head, v_single_head, attn_bias=attention_mask_converted)
            hidden_states_heads.append(hidden_states_single_head)
        # reshape
        hidden_states = torch.cat(hidden_states_heads, dim=2)
        hidden_states = hidden_states.permute(0, 2, 1, 3) # [B, M, H, K] -> [B, H, M, K]
        hidden_states = hidden_states.reshape(batch_size, seq_len, dim)  # [B, H, M, K] -> [(B H), M, K]

        return hidden_states
    
    
    def forward(self, query, key, value, attention_mask, batch_size, heads, scale):
        ## 1. self attn-1 (Sec 3.3)
        if self.cur_step in self.p2p_step_idx and query.shape[1] <= 32 ** 2:
            # softmax, then use controller calculate attention map:
            sim = torch.einsum("b i d, b j d -> b i j", query, key) * scale
            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            uncond_edit, cond_edit, uncond_invert, cond_invert = attn.chunk(4)
            attn_new = torch.cat([uncond_edit, cond_invert.clone(), uncond_invert, cond_invert], dim=0)
            hidden_states = torch.einsum("b i j, b j d -> b i d", attn_new, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states, heads)
            return hidden_states
        #######################################################################################################################################################################
        
        
        ## 2. self attn-2 (Sec 3.3)
        elif self.cur_step in self.step_idx and self.cur_att_layer in self.layer_idx:
            query_uncond_new, query_new, query_uncond_invert, query_invert = query.clone().detach().chunk(4)
            key_uncond_new, key_new, key_uncond_invert, key_invert = key.clone().detach().chunk(4)
            value_uncond_new, value_new, value_uncond_invert, value_invert = value.clone().detach().chunk(4) 
            # recon:
            hidden_states_uncond_invert = xformers.ops.memory_efficient_attention(query_uncond_invert, key_uncond_invert, value_uncond_invert, attn_bias=attention_mask)
            hidden_states_invert = xformers.ops.memory_efficient_attention(query_invert, key_invert, value_invert, attn_bias=attention_mask)
            # generate:
            # hidden_states_uncond_new = xformers.ops.memory_efficient_attention(query_uncond_new, key_uncond_new, value_uncond_new, attn_bias=attention_mask)
            hidden_states_uncond_new = self.attn_batch(q=query_uncond_new, 
                                                k=torch.cat([key_uncond_new, key_uncond_invert], dim=1), 
                                                v=torch.cat([value_uncond_new, value_uncond_invert], dim=1),
                                                masks=self.sam_masks, 
                                                num_heads=heads, attention_mask=attention_mask)
            
            hidden_states_new = self.attn_batch(q=query_new, 
                                                k=torch.cat([key_new, key_invert], dim=1), 
                                                v=torch.cat([value_new, value_invert], dim=1),
                                                masks=self.sam_masks, 
                                                num_heads=heads, attention_mask=attention_mask)

            hidden_states = torch.cat([hidden_states_uncond_new, hidden_states_new, hidden_states_uncond_invert, hidden_states_invert], dim=0)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states, heads)
            
            return hidden_states
        ## 3. else
        else:
            return super().forward(query=query, key=key, value=value, 
                                   attention_mask=attention_mask, batch_size=batch_size, heads=heads)


MAX_NUM_WORDS=77

class LocalBlend:
    
    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = F.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts, words, tokenizer, device, config, substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * config.num_inference_step) # NUM_DDIM_STEPS
        self.counter = 0 
        self.th=th


    
class AttentionControl:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        uncond_edit, cond_edit, uncond_invert, cond_invert = attn.clone().detach().chunk(4)
        cond_edit_new = self.forward(torch.cat([cond_edit, cond_invert], dim=0), is_cross, place_in_unet)
        attn = torch.concat([uncond_edit, cond_edit_new[0], uncond_invert, cond_edit_new[1]], dim=0) # cond_invert

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        
class AttentionControlEdit(AttentionControl):
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if ((is_cross and (self.cross_replace_layers[0] <= self.cur_att_layer < self.cross_replace_layers[1])) or 
        (not is_cross and self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1])):
            cond_edit, cond_invert = attn.chunk(2)
            attn_replace = cond_edit.clone().detach().unsqueeze(0)
            attn_base = cond_invert.clone().detach()
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_replace_new = self.replace_cross_attention(attn_base, attn_replace) * alpha_words + (1 - alpha_words) * attn_replace
                attn = torch.cat([attn_replace_new.to(attn_base.dtype), attn_base.unsqueeze(0)], dim=0) 
            else:
                # pass
                attn_replace_new = self.replace_self_attention(attn_base, attn_replace, place_in_unet) 
                attn = torch.cat([attn_replace_new.to(attn_base.dtype), attn_base.unsqueeze(0)], dim=0)
        else:
            cond_edit, cond_invert = attn.chunk(2)
            attn_replace = cond_edit.unsqueeze(0)
            attn_base = cond_invert
            attn = torch.cat([attn_replace, attn_base.unsqueeze(0)], dim=0) 

        return attn
    
    def __init__(self, prompts, tokenizer, device, 
                 num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 cross_replace_layers,
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        self.cross_replace_layers = cross_replace_layers
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper.to(attn_base.dtype))
      
    def __init__(self, prompts, tokenizer, device, 
                 num_steps: int, 
                 cross_replace_steps: float, 
                 cross_replace_layers,
                 self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, tokenizer, device, 
                                               num_steps, 
                                               cross_replace_steps, 
                                               cross_replace_layers,
                                               self_replace_steps, 
                                               local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, tokenizer, device, 
                 num_steps: int, 
                 cross_replace_steps: float, 
                 cross_replace_layers,
                 self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, tokenizer, device, 
                                              num_steps, 
                                              cross_replace_steps, 
                                              cross_replace_layers,
                                              self_replace_steps, 
                                              local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base * self.equalizer[:, None, None, :].to(attn_base.dtype) 
        return attn_replace

    def __init__(self, prompts, tokenizer, device, 
                 num_steps: int, 
                 cross_replace_steps: float, 
                 cross_replace_layers,
                 self_replace_steps: float, 
                 equalizer,
                local_blend: Optional[LocalBlend] = None, 
                controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, tokenizer, device, 
                                                num_steps, 
                                                cross_replace_steps, 
                                                cross_replace_layers,
                                                self_replace_steps, 
                                                local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(tokenizer, text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def make_controller(config, tokenizer, device, 
                    prompts: List[str], is_replace_controller: bool, 
                    cross_replace_steps: Dict[str, float], 
                    cross_replace_layers,
                    self_replace_steps: float, 
                    blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, tokenizer, device, config)
    if is_replace_controller:
        controller = AttentionReplace(prompts, tokenizer, device, 
                                      config.num_inference_step, 
                                      cross_replace_steps=cross_replace_steps, 
                                      cross_replace_layers=cross_replace_layers,
                                      self_replace_steps=self_replace_steps, 
                                      local_blend=lb)
    else:
        controller = AttentionRefine(prompts, tokenizer, device, 
                                     config.num_inference_step, 
                                     cross_replace_steps=cross_replace_steps, 
                                     cross_replace_layers=cross_replace_layers,
                                     self_replace_steps=self_replace_steps, 
                                     local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(tokenizer, prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, tokenizer, device, 
                                       config.num_inference_step, 
                                       cross_replace_steps=cross_replace_steps,
                                       cross_replace_layers=cross_replace_layers,
                                       self_replace_steps=self_replace_steps, 
                                       equalizer=eq, local_blend=lb, controller=controller)
    return controller

# refer from [https://github.com/google/prompt-to-prompt]
def regiter_crossattn_editor_diffusers_p2p(model, controller): 
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None): 
                is_cross = encoder_hidden_states is not None
                assert is_cross
                batch_size, sequence_length, _ = hidden_states.shape
                encoder_hidden_states = encoder_hidden_states

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                dim = query.shape[-1]

                if self.added_kv_proj_dim is not None:
                    key = self.to_k(hidden_states)
                    value = self.to_v(hidden_states)
                    encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                    encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                    ######record###### record before reshape heads to batch dim
                    if self.processor is not None:
                        self.processor.record_qkv(self, hidden_states, query, key, value, attention_mask)
                    ##################

                    key = self.reshape_heads_to_batch_dim(key)
                    value = self.reshape_heads_to_batch_dim(value)
                    encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
                    encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

                    key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
                    value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
                else:
                    encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                    key = self.to_k(encoder_hidden_states)
                    value = self.to_v(encoder_hidden_states)

                    if self.processor is not None:
                        self.processor.record_qkv(self, hidden_states, query, key, value, attention_mask)

                    key = self.reshape_heads_to_batch_dim(key)
                    value = self.reshape_heads_to_batch_dim(value)

                query = self.reshape_heads_to_batch_dim(query) # reshape query

                if attention_mask is not None:
                    if attention_mask.shape[-1] != query.shape[1]:
                        target_length = query.shape[1]
                        attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                        attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

                if self.processor is not None:
                    self.processor.record_attn_mask(self, hidden_states, query, key, value, attention_mask)

                if is_cross: 
                    sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
                    # attention, what we cannot get enough of
                    attn = sim.softmax(dim=-1)
                    attn = controller(attn, is_cross, place_in_unet)
                    hidden_states = torch.einsum("b i j, b j d -> b i d", attn, value)
                else:
                    assert is_cross, "shouldn't be not-cross!"
                hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
                # linear proj
                hidden_states = self.to_out[0](hidden_states)

                # dropout
                hidden_states = self.to_out[1](hidden_states)
                return hidden_states
                ######

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention': # or net_.__class__.__name__ == 'SelfAttention': # or net_.__class__.__name__ == 'VanillaTemporalModule':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count


    cross_att_count = 0
    for net_name, net in model.named_children():
        if "down" in net_name:
            cross_att_count += register_recr(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_recr(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_recr(net, 0, "up")
    controller.num_att_layers = cross_att_count
    print("p2p_cross_att_count: ", controller.num_att_layers)
    
    
    
def regiter_selfattn_editor_diffusers_p2p(model, editor): 
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
                is_cross = encoder_hidden_states is not None
                assert not is_cross, "shouldn't be cross!"
                batch_size, sequence_length, _ = hidden_states.shape
                encoder_hidden_states = encoder_hidden_states

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                dim = query.shape[-1]

                if self.added_kv_proj_dim is not None:
                    key = self.to_k(hidden_states)
                    value = self.to_v(hidden_states)
                    encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                    encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                    ######record###### record before reshape heads to batch dim
                    if self.processor is not None:
                        self.processor.record_qkv(self, hidden_states, query, key, value, attention_mask)
                    key = self.reshape_heads_to_batch_dim(key)
                    value = self.reshape_heads_to_batch_dim(value)
                    encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
                    encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

                    key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
                    value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
                else:
                    encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                    key = self.to_k(encoder_hidden_states)
                    value = self.to_v(encoder_hidden_states)
                    if self.processor is not None:
                        self.processor.record_qkv(self, hidden_states, query, key, value, attention_mask)
                    key = self.reshape_heads_to_batch_dim(key)
                    value = self.reshape_heads_to_batch_dim(value)

                query = self.reshape_heads_to_batch_dim(query) # reshape query

                if attention_mask is not None:
                    if attention_mask.shape[-1] != query.shape[1]:
                        target_length = query.shape[1]
                        attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                        attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
                if self.processor is not None:
                    self.processor.record_attn_mask(self, hidden_states, query, key, value, attention_mask)
                hidden_states = editor(q=query, k=key, v=value, attention_mask=attention_mask, batch_size=batch_size, num_heads=self.heads, scale=self.scale)
                # linear proj
                hidden_states = self.to_out[0](hidden_states)

                # dropout
                hidden_states = self.to_out[1](hidden_states)
                return hidden_states
                ##########################################

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'SelfAttention': # net_.__class__.__name__ == 'CrossAttention' or  # or net_.__class__.__name__ == 'VanillaTemporalModule':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count


    self_att_count = 0
    for net_name, net in model.named_children():
        if "down" in net_name:
            self_att_count += register_recr(net, 0, "down")
        elif "mid" in net_name:
            self_att_count += register_recr(net, 0, "mid")
        elif "up" in net_name:
            self_att_count += register_recr(net, 0, "up")
    editor.num_att_layers = self_att_count
    print("self_att_count: ", editor.num_att_layers)