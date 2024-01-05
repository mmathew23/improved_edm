import torch
import torch.nn as nn
from diffusers.models.unet_2d import UNet2DOutput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention_processor import Attention
import diffusers.models.attention_processor
from diffusers.utils import is_xformers_available, logging, is_torch_version
from typing import Optional, Callable, Any, Dict, Tuple, Union
import numpy as np
import torch.nn.functional as F


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

# Mostly adapted from the diffusers library, thanks HuggingFace!


# Taken from paper
def weight_normalize(x, eps=1e-4):
    dim = list(range(1, x.ndim))
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    alpha = np.sqrt(n.numel() / x.numel())
    return x / torch.add(eps, n, alpha=alpha)


class Conv2d(nn.Conv2d):
    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(weight_normalize(self.weight))
        fan_in = self.weight[0].numel()
        weight = weight_normalize(self.weight) / np.sqrt(fan_in)
        return self._conv_forward(x, weight, None)


class Linear(nn.Linear):
    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(weight_normalize(self.weight))
        fan_in = self.weight[0].numel()
        weight = weight_normalize(self.weight) / np.sqrt(fan_in)
        return F.linear(x, weight, None)


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float = 1e-4,
    resnet_out_scale_factor: float = 1.0,
    attention_head_dim: Optional[int] = None,
    dropout: float = 0.0,
) -> nn.Module:
    if down_block_type == 'DownBlock2D':
        return DownBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            output_scale_factor=resnet_out_scale_factor,
            add_downsample=add_downsample,
        )
    elif down_block_type == 'AttnDownBlock2D':
        return AttnDownBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            output_scale_factor=resnet_out_scale_factor,
            attention_head_dim=attention_head_dim,
            add_downsample=add_downsample,
        )
    else:
        raise ValueError(f'Unknown down block type {down_block_type}')


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float = 1e-4,
    resolution_idx: Optional[int] = None,
    resnet_out_scale_factor: float = 1.0,
    attention_head_dim: Optional[int] = None,
    dropout: float = 0.0,
) -> nn.Module:
    if up_block_type == 'UpBlock2D':
        return UpBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            output_scale_factor=resnet_out_scale_factor,
            add_upsample=add_upsample,
        )
    elif up_block_type == 'AttnUpBlock2D':
        return AttnUpBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resolution_idx=resolution_idx,
            output_scale_factor=resnet_out_scale_factor,
            attention_head_dim=attention_head_dim,
            add_upsample=add_upsample,
        )
    else:
        raise ValueError(f'Unknown up block type {up_block_type}')


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
    ):
        super().__init__()
        linear_cls = Linear

        self.linear_1 = linear_cls(in_channels, time_embed_dim, bias=False)

    def forward(self, sample):
        sample = self.linear_1(sample)
        return sample


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(
        self, embedding_size: int = 256, scale: float = 1.0, log=True, flip_sin_to_cos=False
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
        self.phase = nn.Parameter(torch.rand(embedding_size), requires_grad=False)
        self.log = log

    def forward(self, x):
        if self.log:
            x = torch.log(x)

        x_proj = (x[:, None] * self.weight[None, :] + self.phase[None, :]) * 2 * np.pi

        out = torch.cos(x_proj)
        return out


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super().__init__()

        self.num_classes = num_classes
        self.linear = Linear(num_classes, embedding_size, bias=False)

    def forward(self, class_idx, device, dtype):
        class_embedding = F.one_hot(class_idx, self.num_classes).to(dtype=dtype, device=device)
        return self.linear(class_embedding * np.sqrt(self.num_classes))


class Upsample2D(nn.Module):
    """A 2D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        conv = None
        if use_conv:
            conv = Conv2d(self.channels, self.out_channels, 3, padding=1, bias=False)

        self.conv = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2

        if use_conv:
            conv = Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding, bias=False)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        self.conv = conv

    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        eps: float = 1e-6,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_2d_out_channels: Optional[int] = None,
        type: str = 'down', # or 'up'
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.block_type = type
        self.output_scale_factor = output_scale_factor

        linear_cls = Linear
        conv_cls = Conv2d

        conv_2d_out_channels = conv_2d_out_channels or out_channels
        if self.block_type == 'down':
            self.conv1 = conv_cls(conv_2d_out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            assert self.block_type == 'up'
            self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if temb_channels is not None:
            self.time_emb_proj = linear_cls(temb_channels, out_channels, bias=False)

        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_cls(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.nonlinearity = nn.SiLU()

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample2D(in_channels, use_conv=True)
        elif self.down:
            self.downsample = Downsample2D(in_channels, use_conv=True, padding=1)

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = conv_cls(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        if self.block_type == 'up':
            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if input_tensor.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                input_tensor = (
                    self.upsample(input_tensor, scale=scale)
                )
            hidden_states = input_tensor
            hidden_states = self.nonlinearity(hidden_states)
            if self.conv_shortcut is not None:
                input_tensor = (
                    self.conv_shortcut(input_tensor)
                )
        elif self.block_type == 'down':
            if self.downsample is not None:
                input_tensor = (
                    self.downsample(input_tensor, scale=scale)
                )
            if self.conv_shortcut is not None:
                input_tensor = (
                    self.conv_shortcut(input_tensor)
                )
            input_tensor = pixel_norm(input_tensor)
            hidden_states = input_tensor
            hidden_states = self.nonlinearity(hidden_states)
        else:
            if self.conv_shortcut is not None:
                input_tensor = (
                    self.conv_shortcut(input_tensor)
                )
            input_tensor = pixel_norm(input_tensor)
            hidden_states = input_tensor
            hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = (
                self.time_emb_proj(temb)[:, :, None, None]
            )

        hidden_states = hidden_states * (1+temb)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        output_tensor.clamp(-256, 256)

        return output_tensor


def pixel_norm(x: torch.FloatTensor, eps=1e-4):
    return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)


class CosineAttnProcessor(nn.Module):
    """Replace group norm pre conv with pixel norm post conv to replace AttnProcessor2_0"""
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        args = ()
        query = pixel_norm(attn.to_q(hidden_states, *args))

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = pixel_norm(attn.to_k(encoder_hidden_states, *args))
        value = pixel_norm(attn.to_v(encoder_hidden_states, *args))

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class XFormersCosineAttnProcessor(nn.Module):
    """Replace group norm pre conv with pixel norm post conv to replace XFormersAttnProcessor"""
    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states

        args = ()

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        query = pixel_norm(attn.to_q(hidden_states, *args))

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = pixel_norm(attn.to_k(encoder_hidden_states, *args))
        value = pixel_norm(attn.to_v(encoder_hidden_states, *args))

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# Patch Attentionsprocessors to use mine
diffusers.models.attention_processor.AttnProcessor2_0.__call__ = CosineAttnProcessor.__call__
diffusers.models.attention_processor.XFormersAttnProcessor.__call__ = XFormersCosineAttnProcessor.__call__


def patch_attention_linear_layers(attention):
    # Patch attention.to_q, attention.to_k, attention.to_v, attention.to_out to use Linear
    # instead of nn.Linear
    attention.to_q = Linear(attention.to_q.in_features, attention.to_q.out_features, bias=False)
    attention.to_k = Linear(attention.to_k.in_features, attention.to_k.out_features, bias=False)
    attention.to_v = Linear(attention.to_v.in_features, attention.to_v.out_features, bias=False)
    attention.to_out[0] = Linear(attention.to_out[0].out_features, attention.to_out[0].out_features, bias=False)


class AttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                    down=False,
                    type='down',
                )
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    residual_connection=True,
                    bias=False,
                    out_bias=False,
                    upcast_softmax=False, # Cosine attention should allow for fully 16bit training
                    _from_deprecated_attn_block=True,
                )
            )
            patch_attention_linear_layers(attentions[-1])

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsampler = ResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                dropout=dropout,
                output_scale_factor=output_scale_factor,
                down=True,
                type='down',
            )
        else:
            self.downsampler = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        upsample_size: Optional[int] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        lora_scale = cross_attention_kwargs.get("scale", 1.0)

        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            cross_attention_kwargs.update({"scale": lora_scale})
            hidden_states = resnet(hidden_states, temb, scale=lora_scale)
            hidden_states = attn(hidden_states, **cross_attention_kwargs)
            hidden_states.clamp(-256, 256)
            output_states = output_states + (hidden_states,)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb=temb, scale=lora_scale)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                    down=False,
                    type='down',
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsampler = ResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                dropout=dropout,
                output_scale_factor=output_scale_factor,
                down=True,
                type='down',
            )
        else:
            self.downsampler = None

        self.gradient_checkpointing = False

    def forward(
        self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None, scale: float = 1.0
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb, scale=scale)

            output_states = output_states + (hidden_states,)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb=temb, scale=scale)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class AttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: int = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                    type='up',
                )
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    residual_connection=True,
                    bias=False,
                    out_bias=False,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )
            patch_attention_linear_layers(attentions[-1])

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsampler = ResnetBlock2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        dropout=dropout,
                        output_scale_factor=output_scale_factor,
                        up=True,
                        type='up',
                    )
        else:
            self.upsampler = None

        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        upsample_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb, scale=scale)
            cross_attention_kwargs = {"scale": scale}
            hidden_states = attn(hidden_states, **cross_attention_kwargs)
            hidden_states.clamp(-256, 256)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb=temb, scale=scale)

        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                    type='up',
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsampler = ResnetBlock2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        dropout=dropout,
                        output_scale_factor=output_scale_factor,
                        up=True,
                        type='up',
                    )
        else:
            self.upsampler = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        upsample_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb, scale=scale)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb=temb, scale=scale)

        return hidden_states


def recursive_he_uniform_init(module: nn.Module, a: float = -1.0, b: float = 1.0):
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            fan_in = child.weight.data.shape[1]
            scale = np.sqrt(1/fan_in)
            nn.init.uniform_(child.weight, a=-scale, b=scale)
            if child.bias is not None:
                print('There is a bias')
                nn.init.constant_(child.bias, 0.0)
        else:
            recursive_he_uniform_init(child, a=a, b=b)


def recursive_normal_init(module: nn.Module, mean: float = 0.0, std: float = 1.0):
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(child.weight, mean=mean, std=std)
            if child.bias is not None:
                print('There is a bias')
                nn.init.constant_(child.bias, 0.0)
        else:
            recursive_normal_init(child, mean=mean, std=std)


class UNet2DModel(ModelMixin, ConfigMixin):
    r"""
    A 2D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample. Dimensions must be a multiple of `2 ** (len(block_out_channels) -
            1)`.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        time_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of time embedding to use.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for Fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip sin to cos for Fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`):
            Tuple of downsample block types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            Block type for middle of UNet, it can be either `UNetMidBlock2D` or `UnCLIPUNetMidBlock2D`.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(224, 448, 672, 896)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for normalization.
        num_class_embeds (`int`, *optional*, defaults to `None`):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim` when performing class
            conditioning with `class_embed_type` equal to `None`.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        down_block_types: Tuple[str] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        attention_head_dim: Optional[int] = 8,
        norm_eps: float = 1e-4,
        add_attention: bool = True,
        num_class_embeds: Optional[int] = None,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # input in_channels+1 due to concating of one's to mitigate removing bias
        self.conv_in = Conv2d(in_channels+1, block_out_channels[0], kernel_size=3, padding=(1, 1), bias=False)

        # time
        self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=0.25)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # loss weighting 
        self.loss_mlp = nn.Sequential(GaussianFourierProjection(embedding_size=block_out_channels[0], scale=0.25), Linear(timestep_input_dim, 1, bias=False))

        # class embedding
        if num_class_embeds is not None:
            self.class_embedding = ClassEmbedding(num_classes=num_class_embeds, embedding_size=time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        resnet_out_scale_factor = 1.0
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_out_scale_factor=resnet_out_scale_factor,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = nn.ModuleList()
        self.add_attention = add_attention
        if add_attention:
            self.mid_block.append(
                Attention(
                    block_out_channels[-1],
                    heads=block_out_channels[-1] // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=mid_block_scale_factor,
                    eps=norm_eps,
                    residual_connection=True,
                    bias=False,
                    out_bias=False,
                    upcast_softmax=False, #CosineAttnProcessor should allow for 16bit fully
                    _from_deprecated_attn_block=True,
                )
            )
        self.mid_block.append(
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                eps=norm_eps,
                dropout=dropout,
                output_scale_factor=mid_block_scale_factor,
            )
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1
            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                dropout=dropout,
                resnet_out_scale_factor=resnet_out_scale_factor,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_out = Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1, bias=False)

        # init weights to normal since weight normalization
        recursive_normal_init(self)
        self.gain = nn.Parameter(torch.ones(1, 1, 1, 1))

    def get_loss_module_weight(self, timestep):
        return self.loss_mlp(timestep)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_loss_mlp: bool = False,
    ) -> Union[UNet2DOutput, Tuple]:
        r"""
        The [`UNet2DModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            class_labels (`torch.FloatTensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_2d.UNet2DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            class_emb = self.class_embedding(class_labels, sample.device, self.dtype).to(dtype=self.dtype)
            emb = emb + class_emb
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        # 2. pre-process
        skip_sample = sample

        # Create a tensor of ones with the same dtype and device
        b, c, h, w = sample.shape
        ones_tensor = torch.ones(b, 1, h, w, dtype=sample.dtype, device=sample.device)
        # Concatenate along the channel dimension
        sample = torch.cat((sample, ones_tensor), dim=1)
        c_in = 1 / torch.sqrt(0.25+timesteps**2)
        sample = self.conv_in(sample*c_in[:, None, None, None])

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        for i, block in enumerate(self.mid_block):
            if i == 0 and self.add_attention:
                sample = block(sample)
            else:
                sample = block(sample, emb)
        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        c_out = (timesteps*0.5) / torch.sqrt(timesteps**2 + 0.25)
        sample = self.conv_out(sample) * c_out[:, None, None, None]

        if skip_sample is not None:
            c_skip = 0.25 / (0.25+timesteps**2)
            sample += skip_sample * c_skip[:, None, None, None]

        if return_loss_mlp:
            loss_w = self.get_loss_module_weight(timesteps)
            if not return_dict:
                return (sample,), loss_w

            return UNet2DOutput(sample=sample), loss_w
        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)
