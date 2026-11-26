import torch
import torch.nn.functional as F


class SafeGroupNorm(torch.nn.Module):
    def __init__(self, norm, k=32.0):
        super().__init__()
        self.num_groups = norm.num_groups
        self.weight = norm.weight
        self.bias = norm.bias
        self.inv = 1.0 / k
        self.eps = max(norm.eps / (k * k), 1e-7)

    def forward(self, x):
        return F.group_norm(x * self.inv, self.num_groups, self.weight, self.bias, self.eps)


class SafeLayerNorm(torch.nn.Module):
    def __init__(self, norm, k=32.0):
        super().__init__()
        self.shape = norm.normalized_shape
        self.weight = norm.weight
        self.bias = norm.bias
        self.inv = 1.0 / k
        self.eps = max(norm.eps / (k * k), 1e-7)

    def forward(self, x):
        return F.layer_norm(x * self.inv, self.shape, self.weight, self.bias, self.eps)


def make_norms_safe(model, ks=None):
    count = 0
    for name, module in list(model.named_modules()):
        replacement = None
        k = float(ks.get(name, 32.0)) if ks else 32.0
        if isinstance(module, torch.nn.GroupNorm):
            replacement = SafeGroupNorm(module, k)
        elif isinstance(module, torch.nn.LayerNorm):
            replacement = SafeLayerNorm(module, k)
        if replacement is None or not name:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, replacement)
        count += 1
    return count


class SafeAttnProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        bsz = query.shape[0]
        head_dim = key.shape[-1] // attn.heads
        query = query.view(bsz, -1, attn.heads, head_dim).transpose(1, 2) * attn.scale
        key = key.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        probs = torch.matmul(query, key.transpose(-1, -2)).softmax(dim=-1)
        hidden_states = torch.matmul(probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(bsz, -1, attn.heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor
