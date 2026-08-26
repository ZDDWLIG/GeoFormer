"""Self-contained model architecture for first-break picking Transformer.

Merged from model/block.py, model/geom_transformer.py.
No external dependencies beyond PyTorch.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Qwen3 style)."""
    def __init__(self, hidden_size, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

class GeomAdaLN(nn.Module):
    """Geometry-conditioned Adaptive Layer Normalization.

    Uses per-trace geometric attributes (offset, relative_elevation) to predict
    channel-wise scale (γ) and shift (β) that modulate the normalized features.
    """

    def __init__(self, d_model, geom_dim=2, eps=1e-8, hidden=128):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=eps)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(geom_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model * 2),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, geom):
        γ, β = self.adaLN_modulation(geom.to(dtype=x.dtype)).chunk(2, dim=-1)
        return (1 + γ) * self.norm(x) + β

def get_norm_layer(norm_type, d_model, eps=1e-8):
    if norm_type.lower() == 'rms':
        return RMSNorm(d_model, eps=eps)
    elif norm_type.lower() in ('layer', 'layernorm'):
        return nn.LayerNorm(d_model, eps=eps)
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    """1D sequential RoPE, applied to Q/K of shape (B, H, L, D)."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"RoPE requires head_dim to be even, got {head_dim}")
        self.head_dim = head_dim
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def apply_to_qk(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        t = position_ids.to(dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("bl,d->bld", t, self.inv_freq)
        cos = freqs.cos().to(dtype=x.dtype)
        sin = freqs.sin().to(dtype=x.dtype)
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(1)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(1)
        return (x * cos) + (_rotate_half(x) * sin)

class AbsoluteCoordinateEncoding(nn.Module):
    """Absolute coordinate encoder based on sincos — supports 4D coordinates."""

    def __init__(self, d_model, coord_dim=4, max_freq=1.0):
        super().__init__()
        self.d_model = d_model
        self.coord_dim = coord_dim
        self.max_freq = max_freq
        self.gamma = nn.Parameter(torch.tensor(0.1))

        if coord_dim == 4:
            if d_model % 4 != 0:
                raise ValueError(f"For 4D coords, d_model={d_model} must be divisible by 4")
            self.dim_per_coord = d_model // 4
        elif coord_dim == 2:
            if d_model % 2 != 0:
                raise ValueError(f"For 2D coords, d_model={d_model} must be even")
            self.dim_per_coord = d_model // 2
        else:
            if d_model % 2 != 0:
                raise ValueError(f"For 1D coords, d_model={d_model} must be even")
            self.dim_per_coord = d_model

        self.freq_bands = []
        for _dim in range(coord_dim):
            dim_t = torch.arange(0, self.dim_per_coord // 2, dtype=torch.float32)
            freqs = 10000.0 ** (2.0 * dim_t / self.dim_per_coord)
            self.freq_bands.append(freqs)

    def _encode_1d_coord(self, coord_values, freqs):
        batch_size, seq_len, _ = coord_values.shape
        coord_max = coord_values.max().item()
        coord_min = coord_values.min().item()
        if coord_max <= 1.0 + 1e-6 and coord_min >= -1e-6:
            pos = coord_values * 1000.0
        else:
            pos = coord_values
        angles = pos / (freqs.unsqueeze(0).unsqueeze(0).to(coord_values.device) + 1e-8)
        emb = torch.zeros(batch_size, seq_len, self.dim_per_coord,
                          dtype=coord_values.dtype, device=coord_values.device)
        emb[:, :, 0::2] = torch.sin(angles)
        emb[:, :, 1::2] = torch.cos(angles)
        return emb

    def _encode_coordinates(self, coords):
        batch_size, seq_len, coord_dim = coords.shape
        encoded_list = []
        for dim in range(coord_dim):
            coord_values = coords[:, :, dim:dim + 1]
            freqs = self.freq_bands[dim].to(coords.device)
            dim_encoded = self._encode_1d_coord(coord_values, freqs)
            encoded_list.append(dim_encoded)
        return torch.cat(encoded_list, dim=-1)

    def forward(self, x, coords):
        coord_encoded = self._encode_coordinates(coords)
        return x + coord_encoded * self.gamma

        enc_end = self._encode_1d(t_end, self.freqs)
        encoded = torch.cat([enc_start, enc_end], dim=-1)
        return x + self.gamma * encoded

def compute_geom_attention_bias(geom, temperature=1.0):
    """Compute pairwise geometry-distance attention bias."""
    diff = geom.unsqueeze(1) - geom.unsqueeze(2)
    dist = torch.sqrt((diff ** 2).sum(-1) + 1e-8)
    bias = -dist.unsqueeze(1) / temperature
    return bias

def expand_geom_for_tokens(geom, seq_len, channels):
    """Expand per-trace geom (B, L, G) to match token count after channel expansion."""
    if channels == 1:
        return geom
    return geom.repeat_interleave(channels, dim=1)

def expand_attn_bias_for_tokens(attn_bias, seq_len, channels):
    """Expand attention bias (B, 1, L, L) to (B, 1, L*C, L*C)."""
    if channels == 1:
        return attn_bias
    return attn_bias.repeat_interleave(channels, dim=2).repeat_interleave(channels, dim=3)

class GatedMultiHeadAttention(nn.Module):
    """Gated multi-head attention (Qwen3 style).

    Supports headwise/elementwise gating, QK normalization, RoPE, and
    optional geometry-aware attention bias.
    """

    def __init__(self, d_model, n_heads, dropout=0.1,
                 headwise_attn_output_gate=False,
                 elementwise_attn_output_gate=False,
                 use_qk_norm=False,
                 qkv_bias=False,
                 rms_norm_eps=1e-8,
                 use_rope: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.hidden_size = d_model
        self.num_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim)
        else:
            self.rope = None
        self.attention_dropout = dropout
        self.use_qk_norm = use_qk_norm
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate

        if self.headwise_attn_output_gate:
            q_proj_out_dim = d_model + n_heads
        elif self.elementwise_attn_output_gate:
            q_proj_out_dim = d_model * 2
        else:
            q_proj_out_dim = d_model

        self.q_proj = nn.Linear(d_model, q_proj_out_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=qkv_bias)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, query, key, value, mask=None, position_ids=None, attn_bias=None):
        batch_size, seq_len, _ = query.size()
        if self.use_rope and position_ids is None:
            position_ids = torch.arange(
                seq_len, device=query.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)

        query_states = self.q_proj(query)
        key_states = self.k_proj(key)
        value_states = self.v_proj(value)

        if self.headwise_attn_output_gate:
            query_states = query_states.view(batch_size, seq_len, self.num_heads, -1)
            query_states, gate_score = torch.split(query_states, [self.head_dim, 1], dim=-1)
            gate_score = gate_score.reshape(batch_size, seq_len, self.num_heads, 1)
            query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        elif self.elementwise_attn_output_gate:
            query_states = query_states.view(batch_size, seq_len, self.num_heads, -1)
            query_states, gate_score = torch.split(query_states, [self.head_dim, self.head_dim], dim=-1)
            gate_score = gate_score.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            gate_score = None

        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if self.use_rope and self.rope is not None and position_ids is not None:
            query_states = self.rope.apply_to_qk(query_states, position_ids)
            key_states = self.rope.apply_to_qk(key_states, position_ids)

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias.to(dtype=attn_weights.dtype)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]
            neg = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(mask == 0, neg)

        attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True)[0]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)

        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

class Qwen3MLP(nn.Module):
    """Qwen3-style MLP with gating: down_proj(act_fn(gate_proj(x)) * up_proj(x))."""

    def __init__(self, hidden_size, intermediate_size, dropout=0.1, hidden_act='gelu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        act_fn_map = {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'silu': nn.SiLU()}
        self.act_fn = act_fn_map.get(hidden_act.lower(), nn.GELU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_state):
        gate_output = self.act_fn(self.gate_proj(hidden_state))
        up_output = self.up_proj(hidden_state)
        output = self.down_proj(gate_output * up_output)
        return self.dropout(output)

class FeedForward(nn.Module):
    """Feed-forward network — backward-compatible interface, uses Qwen3MLP internally."""

    def __init__(self, d_model, d_ff, dropout=0.1, hidden_act='gelu'):
        super().__init__()
        self.mlp = Qwen3MLP(d_model, d_ff, dropout, hidden_act)

    def forward(self, x):
        return self.mlp(x)

class GatedTransformerEncoderBlock(nn.Module):
    """Transformer encoder block with Gated Attention — True Pre-LN + residual.

    Supports Qwen3-style gating, QK normalization, RoPE, and GeomAdaLN.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 headwise_attn_output_gate=False,
                 elementwise_attn_output_gate=False,
                 use_qk_norm=False,
                 qkv_bias=False,
                 norm_type='rms',
                 rms_norm_eps=1e-8,
                 hidden_act='gelu',
                 use_rope: bool = False,
                 use_adaln: bool = False,
                 geom_dim: int = 2,
                 adaln_hidden: int = 128,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_adaln = use_adaln
        self.use_checkpoint = use_checkpoint
        if use_adaln:
            self.norm1 = GeomAdaLN(d_model, geom_dim=geom_dim, eps=rms_norm_eps, hidden=adaln_hidden)
            self.norm2 = GeomAdaLN(d_model, geom_dim=geom_dim, eps=rms_norm_eps, hidden=adaln_hidden)
        else:
            self.norm1 = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)
            self.norm2 = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)

        self.attn = GatedMultiHeadAttention(
            d_model, n_heads, dropout,
            headwise_attn_output_gate=headwise_attn_output_gate,
            elementwise_attn_output_gate=elementwise_attn_output_gate,
            use_qk_norm=use_qk_norm,
            qkv_bias=qkv_bias,
            rms_norm_eps=rms_norm_eps,
            use_rope=use_rope,
        )
        self.drop1 = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, hidden_act=hidden_act)
        self.drop2 = nn.Dropout(dropout)

    def _forward_impl(self, x, mask, position_ids, geom, attn_bias):
        if self.use_adaln and geom is not None:
            h = self.norm1(x, geom)
        else:
            h = self.norm1(x)
        x = x + self.drop1(self.attn(h, h, h, mask, position_ids, attn_bias))
        if self.use_adaln and geom is not None:
            x = x + self.drop2(self.ffn(self.norm2(x, geom)))
        else:
            x = x + self.drop2(self.ffn(self.norm2(x)))
        return x

    def forward(self, x, mask=None, position_ids=None, geom=None, attn_bias=None):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, mask, position_ids, geom, attn_bias,
                use_reentrant=False,
            )
        return self._forward_impl(x, mask, position_ids, geom, attn_bias)

class MultiStageGatedEncoder(nn.Module):
    """Multi-stage Gated encoder.

    Pipeline:
    1. Stage 0: (B, seq_len, input_dim) → (B, seq_len, embed_dim) → Gated Attention
    2. Stage 1-N: Projection (embed_dim halved) → channel expansion → Gated Attention

    Supports GeomAdaLN, GeomAttnBias, multi-stage geometry injection.
    """

    def __init__(self, input_dim, embed_dim=1024, num_stages=3,
                 num_heads=16, d_ff=2048, dropout=0.1,
                 max_channels=8, norm_type='rms',
                 headwise_attn_output_gate=False,
                 elementwise_attn_output_gate=False,
                 use_qk_norm=False,
                 qkv_bias=False,
                 rms_norm_eps=1e-8,
                 hidden_act='gelu',
                 use_rope: bool = False,
                 use_adaln: bool = False,
                 use_geom_attn_bias: bool = False,
                 use_multistage_geom: bool = False,
                 geom_dim: int = 2,
                 use_checkpoint: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_stages = num_stages
        self.max_channels = max_channels
        self.use_rope = use_rope
        self.use_adaln = use_adaln
        self.use_geom_attn_bias = use_geom_attn_bias
        self.use_multistage_geom = use_multistage_geom
        self.geom_dim = geom_dim
        self.use_checkpoint = use_checkpoint

        self.initial_proj = nn.Linear(input_dim, embed_dim)
        self.initial_norm = get_norm_layer(norm_type, embed_dim, eps=rms_norm_eps)

        self.stage0_attn = GatedTransformerEncoderBlock(
            embed_dim, num_heads, d_ff, dropout,
            headwise_attn_output_gate=headwise_attn_output_gate,
            elementwise_attn_output_gate=elementwise_attn_output_gate,
            use_qk_norm=use_qk_norm,
            qkv_bias=qkv_bias,
            norm_type=norm_type,
            rms_norm_eps=rms_norm_eps,
            hidden_act=hidden_act,
            use_rope=use_rope,
            use_adaln=use_adaln,
            geom_dim=geom_dim,
            use_checkpoint=use_checkpoint,
        )

        self.down_stages = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.stage_geom_projs = nn.ModuleList()
        self.stage_geom_gammas = nn.ParameterList()

        if use_multistage_geom:
            self.stage_geom_projs.append(nn.Linear(embed_dim, embed_dim))
            self.stage_geom_gammas.append(nn.Parameter(torch.tensor(1.0)))

        current_channels = 1
        current_embed_dim = embed_dim

        for _stage in range(1, num_stages + 1):
            next_channels = min(current_channels * 2, max_channels)
            next_embed_dim = current_embed_dim // 2

            proj = nn.Linear(current_embed_dim, next_embed_dim)
            channel_expand = nn.Conv2d(current_channels, next_channels,
                                       kernel_size=(1, 3), stride=1, padding=(0, 1))

            attn = GatedTransformerEncoderBlock(
                next_embed_dim, num_heads, d_ff, dropout,
                headwise_attn_output_gate=headwise_attn_output_gate,
                elementwise_attn_output_gate=elementwise_attn_output_gate,
                use_qk_norm=use_qk_norm,
                qkv_bias=qkv_bias,
                norm_type=norm_type,
                rms_norm_eps=rms_norm_eps,
                hidden_act=hidden_act,
                use_rope=use_rope,
                use_adaln=use_adaln,
                geom_dim=geom_dim,
                use_checkpoint=use_checkpoint,
            )

            self.down_stages.append(nn.ModuleDict({
                'proj': proj,
                'channel_expand': channel_expand
            }))
            self.down_attns.append(attn)

            if use_multistage_geom:
                self.stage_geom_projs.append(nn.Linear(embed_dim, next_embed_dim))
                self.stage_geom_gammas.append(nn.Parameter(torch.tensor(1.0)))

            current_channels = next_channels
            current_embed_dim = next_embed_dim

        self.final_norm = get_norm_layer(norm_type, current_embed_dim, eps=rms_norm_eps)

    def forward(self, x, skip_initial_proj=False, geom=None, geom_embed=None, attn_bias=None, position_ids=None):
        B, seq_len, _ = x.shape
        features = []

        if not skip_initial_proj:
            x = self.initial_proj(x)
            x = self.initial_norm(x)

        pos0 = None
        if self.use_rope:
            if position_ids is not None:
                pid = position_ids.to(device=x.device, dtype=torch.long)
                if pid.dim() == 1:
                    pid = pid.unsqueeze(0).expand(B, -1)
                pos0 = pid
            else:
                pos0 = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, -1)

        x = self.stage0_attn(x, position_ids=pos0, geom=geom, attn_bias=attn_bias)

        if self.use_multistage_geom and geom_embed is not None:
            inj = self.stage_geom_projs[0](geom_embed.to(dtype=x.dtype))
            x = x + self.stage_geom_gammas[0] * inj

        features.append(x)

        current_channels = 1
        current_embed_dim = self.embed_dim

        for stage_idx, (down_stage, attn) in enumerate(zip(self.down_stages, self.down_attns)):
            B, tokens, embed_dim = x.shape

            x = down_stage['proj'](x)
            next_embed_dim = x.shape[-1]

            x_4d = x.reshape(B, seq_len, current_channels, next_embed_dim)
            x_4d = x_4d.permute(0, 2, 1, 3)
            x_4d = down_stage['channel_expand'](x_4d)
            next_channels = x_4d.shape[1]
            x_4d = x_4d.permute(0, 2, 1, 3)
            x = x_4d.reshape(B, seq_len * next_channels, next_embed_dim)

            stage_geom = geom
            stage_attn_bias = attn_bias
            if geom is not None and next_channels > 1:
                stage_geom = expand_geom_for_tokens(geom, seq_len, next_channels)
            if attn_bias is not None and next_channels > 1:
                stage_attn_bias = expand_attn_bias_for_tokens(attn_bias, seq_len, next_channels)

            pos1 = None
            if self.use_rope:
                if position_ids is not None and next_channels > 1:
                    pid = position_ids.to(device=x.device, dtype=torch.long)
                    if pid.dim() == 1:
                        pid = pid.unsqueeze(0)
                    pid = pid.repeat_interleave(next_channels, dim=1)
                    pos1 = pid.expand(B, -1)
                elif position_ids is not None:
                    pid = position_ids.to(device=x.device, dtype=torch.long)
                    if pid.dim() == 1:
                        pid = pid.unsqueeze(0).expand(B, -1)
                    pos1 = pid
                else:
                    ntok = x.shape[1]
                    pos1 = torch.arange(ntok, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            x = attn(x, position_ids=pos1, geom=stage_geom, attn_bias=stage_attn_bias)

            if self.use_multistage_geom and geom_embed is not None:
                gs = stage_idx + 1
                proj_out = self.stage_geom_projs[gs](geom_embed.to(dtype=x.dtype))
                if next_channels > 1:
                    proj_out = expand_geom_for_tokens(proj_out, seq_len, next_channels)
                x = x + self.stage_geom_gammas[gs] * proj_out

            features.append(x)
            current_channels = next_channels
            current_embed_dim = next_embed_dim

        x = self.final_norm(x)
        return features, x

class MultiStageGatedDecoder(nn.Module):
    """Multi-stage Gated decoder (symmetric to encoder).

    Pipeline:
    1. Stage N → N-1: Channel compression + projection upsampling + skip + Gated Attention
    2. ...
    3. Stage 0: Final output projection

    Supports GeomAdaLN, GeomAttnBias, multi-stage geometry injection.
    """

    def __init__(self, input_dim, embed_dim=1024, num_stages=3,
                 num_heads=16, d_ff=2048, dropout=0.1,
                 max_channels=8, norm_type='rms',
                 headwise_attn_output_gate=False,
                 elementwise_attn_output_gate=False,
                 use_qk_norm=False,
                 qkv_bias=False,
                 rms_norm_eps=1e-8,
                 hidden_act='gelu',
                 use_rope: bool = False,
                 use_adaln: bool = False,
                 use_geom_attn_bias: bool = False,
                 use_multistage_geom: bool = False,
                 geom_dim: int = 2,
                 use_checkpoint: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_stages = num_stages
        self.max_channels = max_channels
        self.use_rope = use_rope
        self.use_adaln = use_adaln
        self.use_geom_attn_bias = use_geom_attn_bias
        self.use_multistage_geom = use_multistage_geom
        self.geom_dim = geom_dim
        self.use_checkpoint = use_checkpoint

        self.up_stages = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.skip_projs = nn.ModuleDict()
        self.dec_geom_projs = nn.ModuleList()
        self.dec_geom_gammas = nn.ParameterList()

        current_channels = max_channels
        current_embed_dim = embed_dim // (2 ** num_stages)

        for stage in range(num_stages, 0, -1):
            prev_channels = max(1, current_channels // 2)
            prev_embed_dim = current_embed_dim * 2

            channel_reduce = nn.Conv2d(current_channels, prev_channels,
                                       kernel_size=(1, 3), stride=1, padding=(0, 1))
            proj = nn.Linear(current_embed_dim, prev_embed_dim)

            attn = GatedTransformerEncoderBlock(
                prev_embed_dim, num_heads, d_ff, dropout,
                headwise_attn_output_gate=headwise_attn_output_gate,
                elementwise_attn_output_gate=elementwise_attn_output_gate,
                use_qk_norm=use_qk_norm,
                qkv_bias=qkv_bias,
                norm_type=norm_type,
                rms_norm_eps=rms_norm_eps,
                hidden_act=hidden_act,
                use_rope=use_rope,
                use_adaln=use_adaln,
                geom_dim=geom_dim,
                use_checkpoint=use_checkpoint,
            )

            self.up_stages.append(nn.ModuleDict({
                'channel_reduce': channel_reduce,
                'proj': proj
            }))
            self.up_attns.append(attn)

            if use_multistage_geom:
                self.dec_geom_projs.append(nn.Linear(embed_dim, prev_embed_dim))
                self.dec_geom_gammas.append(nn.Parameter(torch.tensor(1.0)))

            current_channels = prev_channels
            current_embed_dim = prev_embed_dim

        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim, input_dim),
            get_norm_layer(norm_type, input_dim, eps=rms_norm_eps)
        )

    def forward(self, x, skip_features, geom=None, geom_embed=None, attn_bias=None, position_ids=None):
        B = x.shape[0]
        use_geom = self.use_adaln or self.use_geom_attn_bias or self.use_multistage_geom

        current_channels = self.max_channels
        current_embed_dim = self.embed_dim // (2 ** self.num_stages)

        if len(skip_features) > 0:
            seq_len = skip_features[0].shape[1]
        else:
            seq_len = x.shape[1] // current_channels

        for stage_idx, (up_stage, attn) in enumerate(zip(self.up_stages, self.up_attns)):
            skip_idx = len(skip_features) - 1 - stage_idx - 1
            skip_feat = skip_features[skip_idx] if skip_idx >= 0 else None

            B, tokens, embed_dim = x.shape

            x_4d = x.reshape(B, seq_len, current_channels, embed_dim)
            x_4d = x_4d.permute(0, 2, 1, 3)
            x_4d = up_stage['channel_reduce'](x_4d)
            prev_channels = x_4d.shape[1]
            x_4d = x_4d.permute(0, 2, 1, 3)
            x = x_4d.reshape(B, seq_len * prev_channels, embed_dim)

            x = up_stage['proj'](x)
            prev_embed_dim = x.shape[-1]

            if skip_feat is not None:
                if skip_feat.shape == x.shape:
                    x = x + skip_feat
                else:
                    if skip_feat.shape[-1] != prev_embed_dim:
                        skip_proj_key = f'skip_proj_{stage_idx}'
                        if skip_proj_key not in self.skip_projs:
                            skip_proj = nn.Linear(skip_feat.shape[-1], prev_embed_dim)
                            torch.nn.init.xavier_uniform_(skip_proj.weight, gain=1.0)
                            if skip_proj.bias is not None:
                                nn.init.constant_(skip_proj.bias, 0)
                            skip_proj = skip_proj.to(x.device)
                            self.add_module(skip_proj_key, skip_proj)
                            self.skip_projs[skip_proj_key] = skip_proj
                        skip_feat = self.skip_projs[skip_proj_key](skip_feat)
                    if skip_feat.shape[1] != x.shape[1]:
                        if skip_feat.shape[1] < x.shape[1]:
                            pad_size = x.shape[1] - skip_feat.shape[1]
                            skip_feat = F.pad(skip_feat, (0, 0, 0, pad_size))
                        else:
                            skip_feat = skip_feat[:, :x.shape[1], :]
                    x = x + skip_feat

            stage_geom = geom
            stage_attn_bias = attn_bias
            if use_geom and geom is not None and prev_channels > 1:
                stage_geom = expand_geom_for_tokens(geom, seq_len, prev_channels)
            if attn_bias is not None and prev_channels > 1:
                stage_attn_bias = expand_attn_bias_for_tokens(attn_bias, seq_len, prev_channels)

            pos_d = None
            if self.use_rope:
                if position_ids is not None and prev_channels > 1:
                    pid = position_ids.to(device=x.device, dtype=torch.long)
                    if pid.dim() == 1:
                        pid = pid.unsqueeze(0)
                    pid = pid.repeat_interleave(prev_channels, dim=1)
                    pos_d = pid.expand(B, -1)
                elif position_ids is not None:
                    pid = position_ids.to(device=x.device, dtype=torch.long)
                    if pid.dim() == 1:
                        pid = pid.unsqueeze(0).expand(B, -1)
                    pos_d = pid
                else:
                    ntok = x.shape[1]
                    pos_d = torch.arange(ntok, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            x = attn(x, position_ids=pos_d, geom=stage_geom, attn_bias=stage_attn_bias)

            if self.use_multistage_geom and geom_embed is not None:
                dec_proj_out = self.dec_geom_projs[stage_idx](geom_embed.to(dtype=x.dtype))
                if prev_channels > 1:
                    dec_proj_out = expand_geom_for_tokens(dec_proj_out, seq_len, prev_channels)
                x = x + self.dec_geom_gammas[stage_idx] * dec_proj_out

            current_channels = prev_channels
            current_embed_dim = prev_embed_dim

        B, tokens, embed_dim = x.shape

        if tokens > seq_len:
            channels = tokens // seq_len
            x = x.reshape(B, seq_len, channels, embed_dim)
            x = x.mean(dim=2)

        x = self.final_proj[0](x)
        return x

def remap_legacy_classifier_state_dict(state_dict):
    """Compat: old cls_head.* -> pick_logits_head.*"""
    return {
        k.replace("cls_head.", "pick_logits_head.", 1) if k.startswith("cls_head.") else k: v
        for k, v in state_dict.items()
    }

class SpatialFirstBreakTransformerGeom(nn.Module):
    """Multi-stage backbone with Gated Attention + first-break picking head.

    Three core geometry innovations (each toggled independently):
      (1) GeomMLP: g = MLP(δx, δz),  x ← x + γ·g
      (2) GeomAdaLN: γ_c, β_c = AdaLNMod(g),  x ← (1+γ_c)⊙RMSNorm(x)+β_c
      (3) GeomAttnBias: d_ij=‖g_i−g_j‖₂, B_ij=−d_ij/τ, A=softmax(QK^⊤/√d_k+B)

    Auxiliary: use_multistage_geom, use_global_trace_context.
    """

    def __init__(
        self,
        input_dim, d_model=512, n_heads=8, n_stages=3, d_ff=2048, dropout=0.1,
        output_dim=None, norm_type="rms", max_channels=8,
        headwise_attn_output_gate=False, elementwise_attn_output_gate=False,
        use_qk_norm=False, qkv_bias=False, rms_norm_eps=1e-8, hidden_act="gelu",
        coord_dim=4, coord_max_freq=1.0, pos_encoding: str = "coord",
        use_geom_mlp: bool = False, geom_mlp_hidden: int = 256,
        use_adaln: bool = False, use_geom_attn_bias: bool = False,
        use_multistage_geom: bool = False,
        geom_attn_bias_temperature: float = 0.3,
        adaln_hidden: int = 256,
        use_global_trace_context: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.norm_type = norm_type
        if pos_encoding not in ("coord", "rope", "coord_rope", "none"):
            raise ValueError("pos_encoding must be 'coord', 'rope', 'coord_rope', or 'none'")
        self.pos_encoding = pos_encoding
        use_coord = pos_encoding in ("coord", "coord_rope")
        use_rope = pos_encoding in ("rope", "coord_rope")
        self.use_coord_encoding = use_coord
        self.use_rope = use_rope

        self.coord_encoding = None
        if use_coord:
            self.coord_encoding = AbsoluteCoordinateEncoding(d_model, coord_dim=coord_dim, max_freq=coord_max_freq)

        self.use_geom_mlp = use_geom_mlp
        self.use_adaln = use_adaln and use_geom_mlp
        self.use_geom_attn_bias = use_geom_attn_bias and use_geom_mlp
        self.use_multistage_geom = use_multistage_geom and use_geom_mlp
        self.geom_attn_bias_temperature = geom_attn_bias_temperature
        self.use_global_trace_context = bool(use_global_trace_context)

        self.encoder = MultiStageGatedEncoder(
            input_dim=input_dim, embed_dim=d_model, num_stages=n_stages,
            num_heads=n_heads, d_ff=d_ff, dropout=dropout,
            max_channels=max_channels, norm_type=norm_type,
            headwise_attn_output_gate=headwise_attn_output_gate,
            elementwise_attn_output_gate=elementwise_attn_output_gate,
            use_qk_norm=use_qk_norm, qkv_bias=qkv_bias,
            rms_norm_eps=rms_norm_eps, hidden_act=hidden_act,
            use_rope=use_rope, use_adaln=self.use_adaln,
            use_geom_attn_bias=self.use_geom_attn_bias,
            use_multistage_geom=self.use_multistage_geom,
            geom_dim=2, use_checkpoint=use_checkpoint,
        )

        if output_dim is None:
            output_dim = input_dim

        self.decoder = MultiStageGatedDecoder(
            input_dim=output_dim, embed_dim=d_model, num_stages=n_stages,
            num_heads=n_heads, d_ff=d_ff, dropout=dropout,
            max_channels=max_channels, norm_type=norm_type,
            headwise_attn_output_gate=headwise_attn_output_gate,
            elementwise_attn_output_gate=elementwise_attn_output_gate,
            use_qk_norm=use_qk_norm, qkv_bias=qkv_bias,
            rms_norm_eps=rms_norm_eps, hidden_act=hidden_act,
            use_rope=use_rope, use_adaln=self.use_adaln,
            use_geom_attn_bias=self.use_geom_attn_bias,
            use_multistage_geom=self.use_multistage_geom,
            geom_dim=2, use_checkpoint=use_checkpoint,
        )
        final_dim = output_dim if output_dim is not None else input_dim

        self.pick_logits_head = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.GELU(),
            nn.Linear(final_dim, final_dim),
        )

        self.dropout = nn.Dropout(dropout)

        if use_geom_mlp:
            h = int(geom_mlp_hidden)
            self.geom_mlp = nn.Sequential(
                nn.Linear(2, h), nn.GELU(),
                nn.Linear(h, h * 2), nn.GELU(),
                nn.Linear(h * 2, d_model),
            )
            self.geom_gamma = nn.Parameter(torch.tensor(1.0))
        else:
            self.geom_mlp = None
            self.geom_gamma = None

        if self.use_global_trace_context:
            self.global_trace_context_mlp = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model),
            )
        else:
            self.global_trace_context_mlp = None

        self.initialize_weights()
        if self.use_global_trace_context and self.global_trace_context_mlp is not None:
            nn.init.zeros_(self.global_trace_context_mlp[-1].weight)
            nn.init.zeros_(self.global_trace_context_mlp[-1].bias)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            pass
        elif isinstance(m, nn.Parameter):
            if m.data.abs().max() > 100:
                m.data.clamp_(-100, 100)

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(remap_legacy_classifier_state_dict(state_dict), strict=strict)

    def forward(self, x, coords=None, mask=None, geom2=None):
        x = self.encoder.initial_proj(x)
        x = self.encoder.initial_norm(x)
        if self.use_coord_encoding and self.coord_encoding is not None and coords is not None:
            x = self.coord_encoding(x, coords)

        geom_embed = None
        attn_bias = None
        if self.use_geom_mlp and geom2 is not None:
            geom_embed = self.geom_mlp(geom2.to(dtype=x.dtype))
            x = x + self.geom_gamma * geom_embed

        x = self.dropout(x)

        if self.use_global_trace_context and self.global_trace_context_mlp is not None:
            gctx = self.global_trace_context_mlp(x.mean(dim=1, keepdim=True))
            x = x + gctx

        if self.use_geom_attn_bias and geom2 is not None:
            attn_bias = compute_geom_attention_bias(geom2, temperature=self.geom_attn_bias_temperature)

        geom_raw = geom2 if (self.use_adaln or self.use_geom_attn_bias) else None

        features, latent = self.encoder(x, skip_initial_proj=True, geom=geom_raw,
                                        geom_embed=geom_embed, attn_bias=attn_bias)
        x = self.decoder(latent, features, geom=geom_raw, geom_embed=geom_embed, attn_bias=attn_bias)
        return self.pick_logits_head(x)

def create_spatial_first_break_transformer_geom(
    input_dim, d_model=512, n_heads=8, n_stages=3, d_ff=2048, dropout=0.1,
    output_dim=None, norm_type="rms", max_channels=8,
    headwise_attn_output_gate=False, elementwise_attn_output_gate=False,
    use_qk_norm=False, qkv_bias=False, rms_norm_eps=1e-8, hidden_act="gelu",
    use_coord_encoding=True, coord_dim=4, coord_max_freq=1.0,
    pos_encoding=None, use_geom_mlp: bool = False, geom_mlp_hidden: int = 256,
    use_adaln: bool = False, use_geom_attn_bias: bool = False,
    use_multistage_geom: bool = False,
    geom_attn_bias_temperature: float = 0.3, adaln_hidden: int = 256,
    use_global_trace_context: bool = False, use_checkpoint: bool = False,
):
    """Build a spatially-aware 5D first-break picking Transformer (geometry-enhanced)."""
    if output_dim is None:
        output_dim = input_dim
    if pos_encoding is None:
        pos_encoding = "coord" if use_coord_encoding else "none"
    return SpatialFirstBreakTransformerGeom(
        input_dim=input_dim, d_model=d_model, n_heads=n_heads, n_stages=n_stages,
        d_ff=d_ff, dropout=dropout, output_dim=output_dim,
        norm_type=norm_type, max_channels=max_channels,
        headwise_attn_output_gate=headwise_attn_output_gate,
        elementwise_attn_output_gate=elementwise_attn_output_gate,
        use_qk_norm=use_qk_norm, qkv_bias=qkv_bias,
        rms_norm_eps=rms_norm_eps, hidden_act=hidden_act,
        coord_dim=coord_dim, coord_max_freq=coord_max_freq,
        pos_encoding=pos_encoding,
        use_geom_mlp=use_geom_mlp, geom_mlp_hidden=geom_mlp_hidden,
        use_adaln=use_adaln, use_geom_attn_bias=use_geom_attn_bias,
        use_multistage_geom=use_multistage_geom,
        geom_attn_bias_temperature=geom_attn_bias_temperature,
        adaln_hidden=adaln_hidden,
        use_global_trace_context=use_global_trace_context,
        use_checkpoint=use_checkpoint,
    )

