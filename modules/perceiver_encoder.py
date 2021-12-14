import numpy as np
import torch
import torch.nn as nn

from torch import einsum
import torch.nn.functional as F
from torch.nn.modules import transformer

from einops import rearrange, repeat
from einops.layers.torch import Reduce

from functools import wraps


import models.gnn as gnn


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


# helper classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class PerceiverNodeEncoder(nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("transformer")
        group.add_argument(
            "--d_model", type=int, default=128, help="transformer d_model."
        )
        group.add_argument("--nhead", type=int, default=4, help="transformer heads")
        group.add_argument(
            "--dim_feedforward",
            type=int,
            default=512,
            help="transformer feedforward dim",
        )
        group.add_argument("--transformer_dropout", type=float, default=0.3)
        group.add_argument("--transformer_activation", type=str, default="relu")
        group.add_argument("--num_encoder_layers", type=int, default=4)
        group.add_argument(
            "--max_input_len",
            default=1000,
            help="The max input length of transformer input",
        )
        group.add_argument(
            "--transformer_norm_input", action="store_true", default=False
        )

        group.add_argument("--latent_dim", type=int, default=8)
        group.add_argument("--self_per_cross_attn", type=int, default=2)
        group.add_argument("--num_latents", type=int, default=8)

    def __init__(self, args):
        super().__init__()

        self.d_model = args.d_model
        self.num_layer = args.num_encoder_layers
        # Creating Transformer Encoder Model

        self.latents = nn.Parameter(torch.randn(args.num_latents, args.latent_dim))
        get_cross_attn = lambda: PreNorm(
            args.latent_dim,
            Attention(
                args.latent_dim,
                args.d_model,
                heads=args.nhead,
                dim_head=args.d_model // args.nhead,
                dropout=args.transformer_dropout,
            ),
            # context_dim=gnn_embed_dim,
        )
        get_cross_ff = lambda: PreNorm(
            args.latent_dim,
            FeedForward(args.latent_dim, dropout=args.transformer_dropout),
        )
        get_latent_attn = lambda: PreNorm(
            args.latent_dim,
            Attention(
                args.latent_dim,
                heads=args.nhead,
                dim_head=args.d_model // args.nhead,
                dropout=args.transformer_dropout,
            ),
        )
        get_latent_ff = lambda: PreNorm(
            args.latent_dim,
            FeedForward(args.latent_dim, dropout=args.transformer_dropout),
        )

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(
            cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff)
        )

        self.layers = nn.ModuleList([])
        for i in range(args.num_encoder_layers):
            # should_cache = i > 0 and weight_tie_layers
            should_cache = False
            cache_args = {"_cache": should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(args.self_per_cross_attn):
                self_attns.append(
                    nn.ModuleList(
                        [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                    )
                )

            self.layers.append(
                nn.ModuleList(
                    [
                        get_cross_attn(**cache_args),
                        get_cross_ff(**cache_args),
                        self_attns,
                    ]
                )
            )

        self.max_input_len = args.max_input_len

        self.norm_input = None
        if args.transformer_norm_input:
            self.norm_input = nn.LayerNorm(args.d_model)
        self.cls_embedding = None
        if args.graph_pooling == "cls":
            self.cls_embedding = nn.Parameter(
                torch.randn([1, 1, args.d_model], requires_grad=True)
            )

    def forward(self, padded_h_node, src_padding_mask):
        """
        padded_h_node: n_b x B x h_d
        src_key_padding_mask: B x n_b
        """

        # (S, B, h_d), (B, S)

        b = padded_h_node.shape[1]

        x = repeat(self.latents, "n d -> b n d", b=b)

        for cross_attn, cross_ff, self_attns in self.layers:
            x = (
                cross_attn(
                    x, context=padded_h_node.permute((1, 0, 2)), mask=src_padding_mask
                )
                + x
            )
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        transformer_out = x.permute((1, 0, 2))
        # (S, B, h_d)

        return transformer_out, src_padding_mask
