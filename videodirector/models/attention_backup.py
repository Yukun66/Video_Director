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
import pdb, math

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
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

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
        # encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)    

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

# 为了修改模块名称，把self attn，cross attn，temp attn 都分开：
class SelfAttention(CrossAttention):
    pass
class TempAttention(CrossAttention):
    pass

# editor 的 base 类，里面是原来的 attn 的 qkv attn 计算模块：
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
        # print("We are in the MutualAttentionBase !!!!!!")
        # out = torch.einsum('b i j, b j d -> b i d', attn, v)
        # out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        # attention, what we cannot get enough of
        # if self._use_memory_efficient_attention_xformers:
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


# attn1 模块中，满足3个条件之一就进入MutualAttentionBase，否则就用新定义的editor计算qkv：
# 1.is_cross(属于交互注意力) or 
# 2. self.cur_step not in self.step_idx(不在交换KV的轮数，比如要求从第4步才开始交换KV) or 
# 3. self.cur_att_layer // 2 not in self.layer_idx (不在所要求的层数，比如要求在所有attn层的最后5层交换KV)
class MutualSelfAttention(MutualAttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, end_step=100, start_layer=10, end_layer=16, 
                 sam_masks=None, num_frames=None, 
                 layer_idx=None, step_idx=None, total_steps=50, model_type="SD"): 
        super().__init__()
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.sam_masks = sam_masks
        self.num_frames = num_frames
        
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.end_step = end_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        print("MutualSelfAttention at denoising steps: ", self.step_idx)
        print("MutualSelfAttention at U-Net layers: ", self.layer_idx)

    # 计算attn mask：
    def attn_mask_cal(self, masks, num_heads, H, W):
        #############################################################
        masks = masks.masked_fill(masks == 1, float('-inf'))
        masks = masks.expand(1, (H*W), self.num_frames, H, W)
        attention_mask = rearrange(masks, "b c f h w -> (b f) c (h w)", f=self.num_frames, h=H, w=W)
        attention_mask = torch.cat([attention_mask, torch.zeros_like(attention_mask)], dim=-1)

        return attention_mask
        #############################################################
    
    
    def attn_batch(self, q, k, v, masks, num_heads, attention_mask):
        # q_new 需要用 k_new和k_invert, 但是cat之前需要用mask计算，因为只保留背景部分：
        H = W = int(math.sqrt(q.shape[1])) # 高*宽
        masks = F.interpolate(masks.float(), size=(H, W), mode='nearest') # .bool() # [16, 1, 32, 32] 后面的(32，32)是(高, 宽)
        masks = masks.permute(1, 0, 2, 3).unsqueeze(0)
        masks = masks.to(k.dtype)

        k_target = k[:, :H*W]
        # k_source_fg = k[:, H*W:]
        k_source_bg = k[:, H*W:]
        v_target = v[:, :H*W]
        # v_source_fg = v[:, H*W:]
        v_source_bg = v[:, H*W:]

        # k_source_fg = rearrange(k_source_fg, "(b f) (h w) c -> b c f h w", f=self.num_frames, h=H, w=W)
        # k_source_fg = k_source_fg * 0 # masks
        # k_source_fg = rearrange(k_source_fg, "b c f h w -> (b f) (h w) c", f=self.num_frames, h=H, w=W)

        k_source_bg = rearrange(k_source_bg, "(b f) (h w) c -> b c f h w", f=self.num_frames, h=H, w=W)
        k_source_bg = k_source_bg * (1-masks)
        k_source_bg = rearrange(k_source_bg, "b c f h w -> (b f) (h w) c", f=self.num_frames, h=H, w=W)

        
        key = torch.cat([k_source_bg, k_target], dim=1) # torch.cat([k_source_fg, k_source_bg, k_target], dim=1)
        value = torch.cat([v_source_bg, v_target], dim=1) # torch.cat([v_source_fg, v_source_bg, v_target], dim=1)

        # 计算attention_mask：
        #############################################################
        attention_mask_converted = self.attn_mask_cal(masks, num_heads, H, W) # attention_mask shape: (B or 1, n_queries, number of keys) 

        # 把 q key value 的维度改一下，xformers.ops.memory_efficient_attention函数可以计算带heads的attention：
        batch_size, seq_len, dim = q.shape
        q = q.reshape(batch_size // num_heads, num_heads, seq_len, dim) # [(B H), M, K]->[B, H, M, K] B: batch size, M: sequence length, H:number of heads, K: embeding size per head
        q = q.permute(0, 2, 1, 3) #  [B, H, M, K]->[B, M, H, K]
        key = key.reshape(batch_size // num_heads, num_heads, 2*seq_len, dim) # [(B H), M, K]->[B, H, M, K] 
        key = key.permute(0, 2, 1, 3) #  [B, H, M, K]->[B, M, H, K]
        value = value.reshape(batch_size // num_heads, num_heads, 2*seq_len, dim) # [(B H), M, K]->[B, H, M, K] 
        value = value.permute(0, 2, 1, 3) #  [B, H, M, K]->[B, M, H, K]

        #############################################################

        # hidden_states = xformers.ops.memory_efficient_attention(q, key, value, attn_bias=attention_mask_converted)
        
        # 由于有mask，一起算hidden states会OOM，改成每个head分别计算attn：
        hidden_states_heads = []
        for i in range(num_heads):
            # 计算单个 head 的 attention
            q_single_head = q[:, :, i:i+1, :]  # 取出第 i 个 head
            k_single_head = key[:, :, i:i+1, :]
            v_single_head = value[:, :, i:i+1, :]
            
            # 只给当前 head 使用相应的 attention mask
            # 计算单个 head 的 attention
            hidden_states_single_head = xformers.ops.memory_efficient_attention(q_single_head, k_single_head, v_single_head, attn_bias=attention_mask_converted)
            hidden_states_heads.append(hidden_states_single_head)
        # 将所有 heads 的输出拼接回原来的形状
        hidden_states = torch.cat(hidden_states_heads, dim=2)

        #############################################################
        # 维度还得变回来：
        hidden_states = hidden_states.permute(0, 2, 1, 3) # [B, M, H, K] -> [B, H, M, K]
        hidden_states = hidden_states.reshape(batch_size, seq_len, dim)  # [B, H, M, K] -> [(B H), M, K]
        #############################################################

        return hidden_states
    
    
    def forward(self, query, key, value, attention_mask, batch_size, heads, scale):
        # 这里除2是因为每轮去噪需要跑2个unet。如果后面改变了unet运行的次数，就需要调整这里的除数。
        # if self.cur_step // 2 not in self.step_idx or self.cur_att_layer not in self.layer_idx:
        if self.cur_step not in self.step_idx or self.cur_att_layer not in self.layer_idx:
            return super().forward(query=query, key=key, value=value, attention_mask=attention_mask, batch_size=batch_size, heads=heads)

        ## 共享 K V：####################
        query_uncond_new, query_new, query_uncond_invert, query_invert = query.chunk(4)
        key_uncond_new, key_new, key_uncond_invert, key_invert = key.chunk(4)
        value_uncond_new, value_new, value_uncond_invert, value_invert = value.chunk(4) 
        # recon:
        hidden_states_uncond_invert = xformers.ops.memory_efficient_attention(query_uncond_invert, key_uncond_invert, value_uncond_invert, attn_bias=attention_mask)
        hidden_states_invert = xformers.ops.memory_efficient_attention(query_invert, key_invert, value_invert, attn_bias=attention_mask)
        # 生成:
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
        #################################################################################################

# 交换cross attn的注意力图：
class MutualCrossAttention(MutualAttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, end_step=100, start_layer=10, end_layer=16, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"): 
        super().__init__()
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.end_step = end_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        print("MutualCrossAttention at denoising steps: ", self.step_idx)
        print("MutualCrossAttention at U-Net layers: ", self.layer_idx)

    def forward(self, query, key, value, attention_mask, batch_size, heads, scale):
        # 这里除2是因为每轮去噪需要跑2个unet。如果后面改变了unet运行的次数，就需要调整这里的除数。
        # if self.cur_step //2 not in self.step_idx or self.cur_att_layer not in self.layer_idx:
        if self.cur_step not in self.step_idx or self.cur_att_layer not in self.layer_idx:
            return super().forward(query=query, key=key, value=value, attention_mask=attention_mask, batch_size=batch_size, heads=heads)
        
        ## 类似prompt2prompt，交换cross attn：####################
        # hidden_states = self._attention(query, key, value, attention_mask)
        #############
        # if self.upcast_attention:
        #     query = query.float()
        #     key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # if self.upcast_softmax:
        #     attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)

        #### 交换cross attn map: 4是对应着 一开始输入的latents有4个
        attention_probs = attention_probs.reshape(4, batch_size//4, heads, query.shape[1], key.shape[1])
        attention_probs[1] = attention_probs[3].detach()
        attention_probs = attention_probs.reshape(batch_size*heads, query.shape[1], key.shape[1])

        ####

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states, heads)
        #############
        return hidden_states
        #################################################################################################

# 交换temp attn的注意力图：
class MutualTempAttention(MutualAttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, end_step=100, start_layer=10, end_layer=16, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"): 
        super().__init__()
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.end_step = end_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        print("MutualTempAttention at denoising steps: ", self.step_idx)
        print("MutualTempAttention at U-Net layers: ", self.layer_idx)

    def forward(self, query, key, value, attention_mask, batch_size, heads, scale):
        # 这里除2是因为每轮去噪需要跑2个unet。如果后面改变了unet运行的次数，就需要调整这里的除数。
        # if self.cur_step //2 not in self.step_idx or self.cur_att_layer not in self.layer_idx:
        if self.cur_step not in self.step_idx or self.cur_att_layer not in self.layer_idx:
            return super().forward(query=query, key=key, value=value, attention_mask=attention_mask, batch_size=batch_size, heads=heads)
        
        # hidden_states = self._attention(query, key, value, attention_mask)
        #############

        # 1. 交换temp attn, 类似prompt2prompt
        # if self.upcast_attention:
        #     query = query.float()
        #     key = key.float()
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # if self.upcast_softmax:
        #     attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)

        if True:
            #### 交换temp attn map: 4是对应着 一开始输入的latents有4个
            attention_probs = attention_probs.reshape(4, -1, heads, query.shape[1], key.shape[1])
            attention_probs[1] = attention_probs[3].detach()
            attention_probs = attention_probs.reshape(-1, query.shape[1], key.shape[1])

            ####

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        # hidden_states = self.reshape_batch_dim_to_heads(hidden_states, heads)
        #############
        
        
        # 2. 把KV 进行替换，然后算个新的hidden_states并替换原来的：
        if False:
            key_f = key.reshape(4, -1, heads, key.shape[1], key.shape[-1])
            key_f[3] = key_f[1].detach()
            key_f = key_f.reshape(key.shape[0], key.shape[1], key.shape[-1])

            value_f = value.reshape(4, -1, heads, value.shape[1], value.shape[-1])
            value_f[3] = value_f[1].detach()
            value_f = value_f.reshape(value.shape[0], value.shape[1], value.shape[-1])

            # hidden_states_f = self._memory_efficient_attention_xformers(query=query, key=key_f, value=value_f, attention_mask=attention_mask, num_heads=heads)
            attention_scores_f = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key_f.shape[1], dtype=query.dtype, device=query.device),
                query,
                key_f.transpose(-1, -2),
                beta=0,
                alpha=scale,
            )

            if attention_mask is not None:
                attention_scores_f = attention_scores_f + attention_mask

            # if self.upcast_softmax:
            #     attention_scores_f = attention_scores_f.float()

            attention_probs_f = attention_scores_f.softmax(dim=-1)

            # cast back to the original dtype
            attention_probs_f = attention_probs_f.to(value_f.dtype)

            # compute attention output
            hidden_states_f = torch.bmm(attention_probs_f, value_f)
            
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states_f = hidden_states_f.to(query.dtype)
            hidden_states = hidden_states.reshape(4, -1, hidden_states.shape[-2], hidden_states.shape[-1])
            hidden_states_f = hidden_states_f.reshape(4, -1, hidden_states_f.shape[-2], hidden_states_f.shape[-1])
            hidden_states[-1] = hidden_states_f[-1].detach()
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])

        hidden_states = self.reshape_batch_dim_to_heads(hidden_states, heads)

        return hidden_states
        #################################################################################################


# 修改 self attn 层的 forward函数：交换KV之用
def regiter_selfattn_editor_diffusers(model, editor: MutualAttentionBase): 
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None): # video_length=None, source_masks=None, target_masks=None, rectangle_source_masks=None
            
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
                hidden_states = editor(q=query, k=key, v=value, attention_mask=attention_mask, batch_size=batch_size, num_heads=self.heads)
                # return out

                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
                # linear proj
                hidden_states = self.to_out[0](hidden_states)

                # dropout
                hidden_states = self.to_out[1](hidden_states)
                return hidden_states
                ##################


        return forward

    def register_editor(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'SelfAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_editor(net__, count, place_in_unet)
        return count


    self_att_count = 0
    for net_name, net in model.named_children():
        if "down" in net_name:
            self_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            self_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            self_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = self_att_count
    print("self_att_count: ", editor.num_att_layers)


# 修改 cross attn 层的 forward函数：交换cross attn map之用：
def regiter_crossattn_editor_diffusers(model, editor: MutualAttentionBase): 
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None): # video_length=None, normal_infer=False, source_masks=None, target_masks=None, rectangle_source_masks=None
            
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
                hidden_states = editor(q=query, k=key, v=value, attention_mask=attention_mask, batch_size=batch_size, num_heads=self.heads, scale=self.scale)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)

                # dropout
                hidden_states = self.to_out[1](hidden_states)
                return hidden_states
                ##########################################

        return forward

    def register_editor(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention' : # or net_.__class__.__name__ == 'SelfAttention' or net_.__class__.__name__ == 'VanillaTemporalModule':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_editor(net__, count, place_in_unet)
        return count


    cross_att_count = 0
    for net_name, net in model.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count
    print("cross_att_count: ", editor.num_att_layers)


# 修改 temporal attn 层的 forward函数：交换 attn map 之用：
def regiter_tempattn_editor_diffusers(model, editor: MutualAttentionBase): 
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None): # video_length=None, normal_infer=False, source_masks=None, target_masks=None, rectangle_source_masks=None
            
                batch_size, sequence_length, _ = hidden_states.shape

                if self.attention_mode == "Temporal":
                    d = hidden_states.shape[1]
                    hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
                    
                    if self.pos_encoder is not None:
                        hidden_states = self.pos_encoder(hidden_states)
                    
                    encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
                else:
                    raise NotImplementedError

                encoder_hidden_states = encoder_hidden_states

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                dim = query.shape[-1]
                # query = self.reshape_heads_to_batch_dim(query) # move backwards

                if self.added_kv_proj_dim is not None:
                    raise NotImplementedError

                encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                ######record###### record before reshape heads to batch dim
                if self.processor is not None:
                    self.processor.record_qkv(self, hidden_states, query, key, value, attention_mask)
                ##################

                key = self.reshape_heads_to_batch_dim(key)
                value = self.reshape_heads_to_batch_dim(value)

                query = self.reshape_heads_to_batch_dim(query) # reshape query here

                if attention_mask is not None:
                    if attention_mask.shape[-1] != query.shape[1]:
                        target_length = query.shape[1]
                        attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                        attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

                ######record######
                if self.processor is not None:
                    self.processor.record_attn_mask(self, hidden_states, query, key, value, attention_mask)
                ##################

                hidden_states = editor(q=query, k=key, v=value, attention_mask=attention_mask, batch_size=batch_size, num_heads=self.heads, scale=self.scale)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)

                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if self.attention_mode == "Temporal":
                    hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

                return hidden_states
                ##################

        return forward

    def register_editor(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'VersatileAttention' : # or net_.__class__.__name__ == 'SelfAttention' or net_.__class__.__name__ == 'VersatileAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_editor(net__, count, place_in_unet)
        return count


    temp_att_count = 0
    for net_name, net in model.named_children():
        if "down" in net_name:
            temp_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            temp_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            temp_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = temp_att_count
    print("temp_att_count: ", editor.num_att_layers)