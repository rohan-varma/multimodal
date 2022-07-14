# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from torchmultimodal.modules.layers.attention import FullAttention, MultiHeadAttention
from torchmultimodal.modules.layers.mlp import MLP


class SiLU(nn.Module):
    r"""Sigmoind Linear Unit

    .. math:: \text{SiLU}(x) = x * \sigma(1.702 * x)

    where :math:`\sigma(x)` is the cumulative distribution function for Logistic Distribution.

    Approximation of the exact GeLU for greater forward speed. Note that this is different from
    ``torch.nn.SiLU`` by the coefficient ``1.702`` from the paper:
    `"Gaussian error linear units"<https://arxiv.org/pdf/1606.08415.pdf>`_.
    """

    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        shape,  # TODO: Retire shape after caching is implemented
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_hidden_layers,
        dropout,
        attention_dropout,
    ):
        super().__init__()
        self.pre_attention_norm = nn.LayerNorm(hidden_size)
        self.post_attention_dropout = nn.Dropout(dropout)
        self.attention = MultiHeadAttention(
            shape,
            hidden_size,
            hidden_size,
            num_attention_heads,
            num_hidden_layers,
            causal=True,
            attention_module=FullAttention(shape, True, attention_dropout),
        )
        self.pre_mlp_norm = nn.LayerNorm(hidden_size)
        self.post_mlp_dropout = nn.Dropout(dropout)
        self.mlp_block = MLP(
            in_dim=hidden_size,
            out_dim=hidden_size,
            hidden_dims=intermediate_size,
            dropout=0.0,
            activation=SiLU,
        )

    def forward(self, x, cache=None):
        h = self.pre_attention_norm(x)
        if self.training:
            # By default the checkpoint API stashes and restores the RNG state during each checkpointed
            # segment such that checkpointed passes making use of RNG (e.g., through dropout, batch norm)
            # have deterministic outputs as compared to non-checkpointed passes. This can incur a moderate
            # performance hit which we mitigate by checkpointing either before and after the layer that
            # requires RNG.
            if cache is not None:
                warnings.warn(
                    "Using `cache` is incompatible with gradient checkpointing. Setting `cache=None`..."
                )
                cache = None

            def create_custom_forward(module):
                # Specifies what should the checkpoint API run in forward pass
                def custom_forward(*inputs):
                    return module(*inputs, cache)

                return custom_forward

            checkpoint(create_custom_forward(self.attention), h, h, h)

        else:
            h = self.attention(h, h, h, cache)

        h = self.post_attention_dropout(h)
        x = x + h

        h = self.pre_mlp_norm(x)
        if self.training:
            h = checkpoint(self.mlp_block, h)
        else:
            h = self.mlp_block(h)

        h = self.post_mlp_dropout(h)
        x = x + h

        return x


# TODO: Remove mask generation from FullAttention
#   Add util to generate mask, attention_mask
#   Add mask as an argument to `forward`
# TODO: Retire shape in attention modules
# TODO: Implement init_cache method to initialize cache
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_hidden_layers,
        dropout,
        attention_dropout,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    shape=(total_len),
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_hidden_layers=num_hidden_layers,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for i in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        cache=None,
    ):
        all_hidden_states = []
        all_attentions = []

        for layer in self.layers:
            all_hidden_states.append(hidden_states)
            layer_outputs = layer(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
            all_attentions.append(layer_outputs[1])

        all_hidden_states.append(hidden_states)

        all_hidden_states = tuple(all_hidden_states)
        all_attentions = tuple(all_attentions)

        return tuple(
            v
            for v in [hidden_states, all_hidden_states, all_attentions]
            if v is not None
        )
