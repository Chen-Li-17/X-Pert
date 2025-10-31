import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many


def exists(val):
    return val is not None

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class MLP(torch.nn.Module):
    
    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)
    
class MLP_gene(nn.Module):
    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        sizes: e.g. [in_dim, h1, h2, ..., out_dim]
        last_layer_act: "linear" | "relu" | "sigmoid" | "tanh" | nn.Module
        """
        super().__init__()
        assert len(sizes) >= 2, "sizes must have at least input and output dim"

        layers = []
        n_layers = len(sizes) - 1

        for i in range(n_layers):
            in_dim, out_dim = sizes[i], sizes[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))

            is_last = i == n_layers - 1
            if not is_last:                              # hidden layer
                if batch_norm:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU())
            else:                                        # output layer
                act = last_layer_act.lower() if isinstance(last_layer_act, str) else last_layer_act
                if act == "relu":
                    layers.append(nn.ReLU())
                elif act == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif act == "tanh":
                    layers.append(nn.Tanh())
                elif act == "linear" or act is None:
                    pass
                elif isinstance(act, nn.Module):
                    layers.append(act)
                else:
                    raise ValueError(f"Unsupported last_layer_act: {last_layer_act}")

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)

class PerceiverAttention_mask(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents, mask):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        # add mask
        b, h, t, n1, n2 = sim.shape
        mask = torch.cat((mask, torch.zeros(mask.shape[0], latents.shape[-2], device=mask.device)), dim=-1)
        mask = repeat(mask, 'b n -> b h t n1 n', h=h, t=t, n1=n1)
        sim = sim.masked_fill(mask == 1, float('-inf'))


        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_media_embeds = 4,
        ff_mult = 4,
        mask_mode = True,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        # self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))

        self.layers = nn.ModuleList([])
        if mask_mode:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PerceiverAttention_mask(dim = dim, dim_head = dim_head, heads = heads),
                    FeedForward(dim = dim, mult = ff_mult)
                ]))

        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                    FeedForward(dim = dim, mult = ff_mult)
                ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        # times = x.shape[1]
        # x = x + self.media_pos_emb[:times]

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents, mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)
    


class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        only_attend_immediate_media = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        # whether for text to only attend to immediate preceding image, or all images

        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(
        self,
        x,
        media,
        media_locations = None
    ):
        b, t, m = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, 'b t n d -> b (t n) d')

        k, v = self.to_kv(media).chunk(2, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        if exists(media_locations):
            text_time = media_locations.cumsum(dim = -1) # at each boolean of True, increment the time counter (relative to media time)
            media_time = torch.arange(t, device = x.device) + 1

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(rearrange(text_time, 'b i -> b 1 i 1'), repeat(media_time, 'j -> 1 1 1 (j m)', m = m))
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
            attn = attn.masked_fill(text_without_media_mask, 0.)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        only_attend_immediate_media = True,
        attn_gate_mode = None,
        use_scgpt_layer = True,
        use_scgpt_input = True,
        cross_norm = False,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(dim = dim, dim_head = dim_head, heads = heads, only_attend_immediate_media = only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult = ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

        self.attn_gate_mode = attn_gate_mode
        self.use_scgpt_layer = use_scgpt_layer
        self.use_scgpt_input = use_scgpt_input
        self.cross_norm = cross_norm

        if cross_norm:
            # 两个 Post-Norm
            self.ln_attn = nn.LayerNorm(dim, elementwise_affine=True)
            self.ln_ff   = nn.LayerNorm(dim, elementwise_affine=True)

    def forward(
        self,
        x,
        media,                  # media tensor, encoded by perceiver resample - (batch, time, latents, dim)
        media_locations = None  # boolean tensor indicating positions of media - (batch, sequence)
    ):
        # - use scgpt transformer encoders and the embeddings
        if self.use_scgpt_layer:
            if self.attn_gate_mode:
                x = self.attn(x, media, media_locations = media_locations) * self.attn_gate.tanh() + x
                if self.cross_norm:
                    x = self.ln_attn(x)   # Post-LN
                x = self.ff(x) * self.ff_gate.tanh()  + x
                if self.cross_norm:
                    x = self.ln_ff(x)     # Post-LN
            else:
                x = self.attn(x, media, media_locations = media_locations) + x
                if self.cross_norm:
                    x = self.ln_attn(x)   # Post-LN
                x = self.ff(x)  + x
                if self.cross_norm:
                    x = self.ln_ff(x)     # Post-LN


        else:
            if self.use_scgpt_input:
                x = self.attn(x, media, media_locations = media_locations) * self.attn_gate.tanh() + x
                if self.cross_norm:
                    x = self.ln_attn(x)   # Post-LN
                x = self.ff(x) * self.ff_gate.tanh()  + x
                if self.cross_norm:
                    x = self.ln_ff(x)     # Post-LN
            else:
                x = self.attn(x, media, media_locations = media_locations)
                x = self.ff(x) * self.ff_gate.tanh()  + x
                if self.cross_norm:
                    x = self.ln_ff(x)     # Post-LN
        return x
