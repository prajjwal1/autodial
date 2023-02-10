#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension

See https://arxiv.org/abs/1910.13461.

The BART agent can be instantiated as simply `-m bart`,
however it is recommended to specify `--init-model zoo:bart/bart_large/model`
or `-mf zoo:bart/bart_large/model` to ensure correct dictionaries are saved.
"""
from __future__ import annotations


#  from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np

import torch
import torch.cuda
import torch.nn.functional as F
from torch import nn

#  import parlai.utils.fsdp as fsdp_utils
from parlai.agents.bart.convert_fairseq_to_parlai import ConversionScript
from parlai.agents.transformer.modules import (
    TransformerDecoder,
    TransformerEncoder,
    create_embeddings,
)
from parlai.agents.transformer.modules.modular import swappable
from parlai.agents.transformer.transformer import (
    #  _check_positional_embeddings,
    add_common_cmdline_args,
)

#  from parlai.core.agents import compare_init_model_opts
#  from parlai.core.message import Message
from parlai.core.torch_generator_agent import SearchBlocklist
from parlai.core.metrics import AverageMetric, FairseqBleuMetric, SumMetric
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import (
    Batch,
    DictionaryAgent,
    History,
    Output,
    TorchAgent,
)
from parlai.core.torch_generator_agent import (
    BeamSearch,
    DelayedBeamSearch,
    GreedySearch,
    NucleusSampling,
    PPLMetric,
    SearchBlocklist,
    TopKSampling,
    TorchGeneratorModel,
    TorchGeneratorAgent,
)

#  from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.metrics import ExactMatchMetric, F1Metric
#  from parlai.utils.distributed import is_distributed, sync_parameters
from parlai.utils.fp16 import FP16SafeCrossEntropy
#  from parlai.utils.io import PathManager
from parlai.utils.logging import logging
from parlai.utils.misc import AttrDict, recursive_getattr, warn_once
from parlai.utils.torch import (
    #  PipelineHelper,
    argsort,
    neginf,
    #  total_parameters,
    #  trainable_parameters,
)
#  from parlai.utils.typing import TShared
from parlai.zoo.bart.build import BART_ARGS, CONVERSION_ARGS, download
from parlai.agents.bart.bart import BartAgent

DecoderIncrState = Dict[int, Dict[str, Dict[str, torch.tensor]]]

from parlai.core.core_utils import (
    NashBargainingLoss,
    MultiTaskBatch,
    OutputGenerator,
    OutputClassifier,
    GlobalModelOutput,
    set_requires_grad
)
from parlai.agents.transformer.modules import (
    create_position_codes,
    get_n_positions_from_options,
    LAYER_NORM_EPS,
    MultiHeadAttention,
    TransformerFFN,
)
from parlai.core.params import default
from parlai.utils.torch import PipelineHelper
from parlai.utils.fsdp import fsdp_wrap
from parlai.nn.checkpoint import checkpoint_wrapper

from parlai.agents.transformer.modules import (
    create_position_codes,
    get_n_positions_from_options,
    LAYER_NORM_EPS,
    MultiHeadAttention,
    TransformerFFN,
)
from abc import ABC


################
# Decoder #
################

DecoderIncrState = Dict[int, Dict[str, Dict[str, torch.Tensor]]]

# Prefix Changes
class BaseTransformerDecoder(nn.Module, ABC):
    """
    Implements functionality common to all transformer decoder variants. Not intended to
    be instantiated directly.

    For a (Vaswani 2017) style encoder-decoder transformer, use ``TransformerDecoder``. For a GPT-style decoder-only transformer, use ``TransformerDecoderOnly``.

    Subclasses are required to implement ``forward``. In your ``forward`` implementation, you can call ``forward_embedding`` to get embeddings for the input tokens and ``forward_layers`` to pass those embeddings sequentially through each layer.

    Subclasses can optionally override ``__init__``, ``build_layer``, and
    ``build_layers`` to customize subcomponents. In particular, ``build_layer`` can be used to instantiate heterogeneous layers (e.g. every other layer being a different type).
    """

    def __init__(
        self,
        opt: Opt,
        embedding: nn.Embedding,
        dictionary: DictionaryAgent,
        n_positions: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.opt = opt
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]

        self.embedding_size = opt['embedding_size']
        self.ffn_size = opt['ffn_size']
        self.n_layers = (
            opt['n_decoder_layers']
            if opt.get('n_decoder_layers', -1) > 0
            else opt['n_layers']
        )
        self.n_heads = opt['n_heads']
        self.dim = self.embedding_size
        self.activation = opt.get('activation', 'relu')
        self.variant = opt.get('variant', 'aiayn')

        self.embeddings_scale = opt.get('embeddings_scale', True)
        self.dropout = nn.Dropout(p=opt.get('dropout', 0.0))  # --dropout

        self.n_positions = default(n_positions, get_n_positions_from_options(opt))
        self.out_dim = self.embedding_size
        assert (
            self.embedding_size % self.n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        if (
            self.variant == 'xlm'
            or self.variant == 'prelayernorm'
            or self.variant == 'bart'
        ):
            self.norm_embeddings = torch.nn.LayerNorm(self.dim, eps=LAYER_NORM_EPS)
            if self.variant == 'xlm':
                warn_once(
                    'DEPRECATED: XLM should only be used for backwards compatibility, '
                    'as it involves a less-stable layernorm operation.'
                )
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(self.n_positions, self.embedding_size)
        if not opt.get('learn_positional_embeddings', False):
            create_position_codes(
                self.n_positions,
                self.embedding_size,
                out=self.position_embeddings.weight,
            )
        else:
            nn.init.normal_(
                self.position_embeddings.weight, 0, self.embedding_size**-0.5
            )

        # build the model
        self.layers = self.build_layers()

    def build_layers(self) -> nn.ModuleList:
        """
        Instantiates all layers. Called only once during __init__.

        Additional setup common to all layers, such as checkpoint wrapping, can be done
        here.
        """
        layers = nn.ModuleList()
        for i in range(self.n_layers):
            layer = self.build_layer(index=i)
            if self.opt.get('checkpoint_activations'):
                layer = checkpoint_wrapper(layer)
            layers.append(fsdp_wrap(layer))  # type: ignore
        return layers

    def build_layer(self, index: int) -> BaseTransformerDecoderLayer:
        """
        Instantiate a single layer. Called n_layers times during __init__.

        :param int index:
            Index of current layer.
        """
        return BaseTransformerDecoderLayer(  # type: ignore
            self.opt,
            attention_dropout=self.opt.get('attention_dropout', 0.0),
            relu_dropout=self.opt.get('relu_dropout', 0.0),
            dropout=self.opt.get('dropout', 0.0),
            activation=self.activation,
            variant=self.variant,
        )

    def forward(
        self,
        input: torch.Tensor,
        encoder_state: Tuple[torch.Tensor, torch.Tensor],
        incr_state: Optional[DecoderIncrState] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        raise NotImplementedError

    def forward_embedding(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Embed tokens prior to feeding into transformer.

        :param LongTensor[batch, seqlen] input:
            The target input IDs
        :param LongTensor[batch, seqlen] positions:
            Positions for input IDs. If None, computes defaults.
        :param LongTensor[batch, seqlen] segments:
            Segment IDs for extra embedding features. If None, not used.

        :return (tensor, mask):
            embedded input and mask
        """
        tensor = self.embeddings(input)

        if self.opt["prefix_seq_len"]:

            positions = positions + self.opt["prefix_seq_len"]
            tensor = self.embeddings(input)

        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        if self.variant == 'xlm':
            tensor = self.norm_embeddings(tensor)
        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if self.variant == 'bart':
            tensor = self.norm_embeddings(tensor)

        return tensor

    def forward_layers(
        self, tensor: torch.Tensor, *extra_args, incr_state: DecoderIncrState, **kwargs
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Forward pass of decoder layers.

        :param tensor:
            embedded input tensor for the decoder
        :param extra_args:
            any number of positional arguments to be passed to each layer
        :param incr_state:
            Dict mapping layer_idx to incremental state
        :param kwargs:
            any number of keyword (named) arguments to be passed to each layer

        :return (tensor, new_incr_state):
            return encoding after applying decoder layers, as well
            as new incremental decoding state.
        """
        new_incr_state = {}
        if getattr(self.layers, 'is_model_parallel', False):
            tensor, new_incr_state = self._apply_model_parallel(
                tensor, *extra_args, incr_state=incr_state
            )
        else:
            for idx, layer in enumerate(self.layers):
                #  print("inside decoder ", idx)
                #  print("tensor ", tensor.shape)
                if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
                    if "past_key_values" in kwargs and idx in kwargs["past_key_values"]:
                        #  print("using this ")
                        past_key_values = kwargs["past_key_values"][idx]
                        batch_size, seq_len = tensor.size(0), tensor.size(1)
                        new_mask = torch.zeros(batch_size, seq_len, self.opt["prefix_seq_len"]).cuda()
                        past_key_values.mask = new_mask
                else:
                    past_key_values = None

                if "inference_mode" in kwargs:
                    #  print(kwargs['inference_mode'], kwargs['decoding_idx'])
                    if kwargs['decoding_idx'] == 1:
                        batch_size, seq_len = tensor.size(0), tensor.size(1)
                        new_mask = torch.zeros(batch_size, 1, 8).cuda()
                        past_key_values = {"prev_mask": new_mask}
                else:
                    past_key_values = None
                tensor, new_incr_state[idx] = layer(
                    tensor, *extra_args, incr_state=incr_state.get(idx), past_key_values=past_key_values
                )

        return tensor, new_incr_state

    def _apply_model_parallel(
        self, tensor: torch.Tensor, *extra_args, incr_state: DecoderIncrState
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Pipeline application of model parallelism.
        """
        chunks = PipelineHelper.split((tensor, *extra_args, incr_state))
        work_items = PipelineHelper.schedule_work_items(self.layers, chunks)

        new_incr_state = {i: [] for i, _ in enumerate(self.layers)}

        for chunk_idx, layer_nos, next_device in work_items:
            s_tensor, *s_extra_args, s_incr_state = chunks[chunk_idx]
            for layer_no in layer_nos:
                s_tensor, nis = self.layers[layer_no](
                    s_tensor, *s_extra_args, incr_state=s_incr_state.get(layer_no)
                )
                new_incr_state[layer_no].append(nis)
            # don't move incr state, it's always on the correct device
            s_layer_args = PipelineHelper.chunk_to(
                (s_tensor, *s_extra_args), next_device
            )
            chunks[chunk_idx] = (*s_layer_args, s_incr_state)

        tensor_out = PipelineHelper.join([c[0] for c in chunks])
        new_incr_state = {
            layer_no: PipelineHelper.join(pieces)
            for layer_no, pieces in new_incr_state.items()
        }

        return tensor_out, new_incr_state

class BaseTransformerDecoderLayer(nn.Module, ABC):
    """
    Implements functionality common to all transformer decoder layer variants. Subclass
    this if you'd like to modify the behavior of any layer in a transformer decoder.

    While this code is functional, it is not intended to be instantiated directly. If
    this functionality is desired as-is, use TransformerDecoderOnlyLayer instead to gain
    the ability to swap self-attention and feedforward classes at instantiation.
    """

    def __init__(
        self,
        opt: Opt,
        n_heads: int = None,
        embedding_size: int = None,
        ffn_size: int = None,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = 'relu',
        variant: str = 'aiayn',
        **kwargs,
    ):
        super().__init__()

        n_heads = default(n_heads, opt['n_heads'])
        embedding_size = default(embedding_size, opt['embedding_size'])
        ffn_size = default(ffn_size, opt['ffn_size'])

        self.opt = opt
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.variant = variant
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = self.build_self_attention(
            n_heads=n_heads, dim=embedding_size, dropout=attention_dropout
        )
        self.norm1 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.ffn = self.build_feedforward(
            dim=embedding_size,
            dim_hidden=ffn_size,
            relu_dropout=relu_dropout,
            activation=activation,
        )
        self.norm3 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

    def build_self_attention(
        self, n_heads: int = None, dim: int = None, dropout: float = 0
    ) -> MultiHeadAttention:
        return MultiHeadAttention(
            opt=self.opt, n_heads=n_heads, dim=dim, dropout=dropout
        )

    def build_feedforward(
        self,
        dim: int = None,
        dim_hidden: int = None,
        relu_dropout: float = 0,
        activation: str = 'relu',
    ) -> TransformerFFN:
        return TransformerFFN(
            opt=self.opt,
            dim=dim,
            dim_hidden=dim_hidden,
            relu_dropout=relu_dropout,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        *extra_args,
        incr_state: Optional[DecoderLayerIncrState] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecoderLayerIncrState]:
        """
        Forward pass.

        The incremental state is a dict with values for self-attention states.
        """
        if incr_state is None:
            incr_state = {}

        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        if self.variant == 'prelayernorm':
            x = self.norm1(x)


        # don't peak into the future!
        x, final_self_attn_incr_state = self.self_attention(
            query=x,
            mask=decoder_mask,
            incr_state=incr_state.get('self_attn'),
            static_kv=False,
            **kwargs,
        )[:2]
        x = self.dropout(x)  # --dropout
        x = x + residual
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm1(x)

        # finally the ffn
        residual = x
        if self.variant == 'prelayernorm':
            x = self.norm3(x)
        x = self.ffn(x, **kwargs)
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm3(x)

        return x, {'self_attn': final_self_attn_incr_state}

    def reorder_incremental_state(
        self, incremental_state: DecoderLayerIncrState, inds: torch.Tensor
    ) -> Dict[str, dict]:
        """
        Reorder all incremental-state tensors for this layer.
        """
        attn_types = {'self_attn': self.self_attention}
        return {
            attn_type: attn.reorder_incremental_state(
                incremental_state[attn_type], inds
            )
            for attn_type, attn in attn_types.items()
        }

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask

@swappable(
    self_attention=MultiHeadAttention,
    encoder_attention=MultiHeadAttention,
    feedforward=TransformerFFN,
)
class TransformerDecoderLayer(BaseTransformerDecoderLayer):
    """
    Implements a single Transformer decoder layer with cross (encoder) attention as in
    [Vaswani, 2017](https://arxiv.org/abs/1706.03762).

    Decoder layers are similar to encoder layers but:

    1. Self-attention is limited in a causal (auto-regressive) manner.
    2. Attend over all of the encoder states.
    """

    def __init__(
        self,
        opt: Opt,
        n_heads: int = None,
        embedding_size: int = None,
        ffn_size: int = None,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = 'relu',
        variant: str = 'aiayn',
        **kwargs,
    ):
        super().__init__(
            opt=opt,
            n_heads=n_heads,
            embedding_size=embedding_size,
            ffn_size=ffn_size,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
            dropout=dropout,
            activation=activation,
            variant=variant,
            **kwargs,
        )

        n_heads = default(n_heads, opt['n_heads'])
        embedding_size = default(embedding_size, opt['embedding_size'])

        self.encoder_attention = self.swappables.encoder_attention(  # type: ignore
            opt=self.opt, n_heads=n_heads, dim=embedding_size, dropout=attention_dropout
        )
        self.norm2 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

    def build_self_attention(
        self, n_heads: int = None, dim: int = None, dropout: float = 0
    ) -> MultiHeadAttention:
        """
        Overridden to allow swapping out of the attention class at instantiation.
        """
        return self.swappables.self_attention(  # type: ignore
            opt=self.opt, n_heads=n_heads, dim=dim, dropout=dropout
        )

    def build_feedforward(
        self,
        dim: int = None,
        dim_hidden: int = None,
        relu_dropout: float = 0,
        activation: str = 'relu',
    ) -> TransformerFFN:
        """
        Overridden to allow swapping out of the feedforward class at instantiation.
        """
        return self.swappables.feedforward(  # type: ignore
            opt=self.opt,
            dim=dim,
            dim_hidden=dim_hidden,
            relu_dropout=relu_dropout,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        incr_state: Optional[DecoderLayerIncrState] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecoderLayerIncrState]:
        """
        Forward pass.

        The incremental state is a dict with values for self- and encoder-attention
        states.
        """

        if incr_state is None:
            incr_state = {}

        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        if self.variant == 'prelayernorm':
            x = self.norm1(x)

        # don't peak into the future!
        x, final_self_attn_incr_state = self.self_attention(
            query=x,
            mask=decoder_mask,
            incr_state=incr_state.get('self_attn'),
            static_kv=False,
            **kwargs,
        )[:2]
        x = self.dropout(x)  # --dropout
        x = x + residual
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm1(x)

        residual = x
        # encoder_attn_layer_norm norm 2
        if self.variant == 'prelayernorm':
            x = self.norm2(x)
        x, final_encoder_attn_incr_state = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask,
            incr_state=incr_state.get('encoder_attn'),
            static_kv=True,
            **kwargs,
        )[:2]
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm2(x)

        # finally the ffn
        residual = x
        if self.variant == 'prelayernorm':
            x = self.norm3(x)
        x = self.ffn(x, **kwargs)
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm3(x)

        new_incr_state = {
            'self_attn': final_self_attn_incr_state,
            'encoder_attn': final_encoder_attn_incr_state,
        }
        return x, new_incr_state

    def reorder_incremental_state(
        self, incremental_state: DecoderLayerIncrState, inds: torch.Tensor
    ) -> DecoderLayerIncrState:
        """
        Reorder all incremental-state tensors for this layer.
        """
        attn_types = {
            'self_attn': self.self_attention,
            'encoder_attn': self.encoder_attention,
        }
        return {
            attn_type: attn.reorder_incremental_state(
                incremental_state[attn_type], inds
            )
            for attn_type, attn in attn_types.items()
        }


@swappable(layer=TransformerDecoderLayer)
class TransformerDecoder(BaseTransformerDecoder):
    """
    Transformer Decoder module.

    For documentation on parameters that are take directly from opt,
    see parlai/agents/transformer/transformer.py

    :param opt: ParlAI-parsed options.
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def build_layer(self, index: int) -> BaseTransformerDecoderLayer:
        """
        Instantiate a single layer. Called n_layers times during __init__.

        Overridden to allow swapping out of the layer class at instantiation.

        :param int index:
            Index of current layer.
        """
        return self.swappables.layer(  # type: ignore
            self.opt,
            attention_dropout=self.opt.get('attention_dropout', 0.0),
            relu_dropout=self.opt.get('relu_dropout', 0.0),
            dropout=self.opt.get('dropout', 0.0),
            activation=self.activation,
            variant=self.variant,
        )

    def forward(
        self,
        input: torch.Tensor,
        encoder_state: Tuple[torch.Tensor, torch.Tensor],
        incr_state: Optional[DecoderIncrState] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        encoder_output, encoder_mask = encoder_state

        seq_len = input.size(1)
        positions = torch.arange(
            seq_len, dtype=torch.long, device=input.device
        ).unsqueeze(0)

        if incr_state is not None:
            # We're doing incremental decoding, so select only the most recent position
            input = input[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        else:
            incr_state = {}

        tensor = self.forward_embedding(input, positions, **kwargs)

        tensor = self.dropout(tensor)  # --dropout

        tensor, new_incr_state = self.forward_layers(
            tensor, encoder_output, encoder_mask, incr_state=incr_state, **kwargs
        )

        if self.variant == 'prelayernorm':
            tensor = self.norm_embeddings(tensor)

        return tensor, new_incr_state

###########################
# ENCODER MODULES #
###########################


@swappable(self_attention=MultiHeadAttention, feedforward=TransformerFFN)
class TransformerEncoderLayer(nn.Module):
    """
    Implements a single Transformer encoder layer.
    """

    def __init__(
        self,
        opt: Opt,
        n_heads: int = None,
        embedding_size: int = None,
        ffn_size: int = None,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = 'relu',
        variant: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        n_heads = default(n_heads, opt['n_heads'])
        embedding_size = default(embedding_size, opt['embedding_size'])
        ffn_size = default(ffn_size, opt['ffn_size'])

        self.opt = opt
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.activation = activation
        self.variant = variant
        self.attention = self.swappables.self_attention(  # type: ignore
            opt=self.opt,
            n_heads=n_heads,
            dim=embedding_size,
            dropout=attention_dropout,  # --attention-dropout
        )
        self.norm1 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.ffn = self.swappables.feedforward(  # type: ignore
            opt=self.opt,
            dim=embedding_size,
            dim_hidden=ffn_size,
            relu_dropout=relu_dropout,
            activation=self.activation,
        )
        self.norm2 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, tensor: torch.Tensor, mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        residual = tensor
        if self.variant == 'prelayernorm':
            tensor = self.norm1(tensor)
        attended_tensor = self.attention(tensor, mask=mask, **kwargs)[0]
        tensor = residual + self.dropout(attended_tensor)
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            tensor = self.norm1(tensor)
        residual = tensor
        if self.variant == 'prelayernorm':
            tensor = self.norm2(tensor)
        tensor = residual + self.dropout(self.ffn(tensor))
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            tensor = self.norm2(tensor)
        tensor *= mask.unsqueeze(-1).type_as(tensor)
        return tensor

class PrefixEmbedding(torch.nn.Module):
    '''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''

    def __init__(self, config):
        super().__init__()
        self.prefix_embedding = nn.Embedding(config["prefix_seq_len"], config["embedding_size"])

        self.prefix_weights = torch.nn.Sequential(
            torch.nn.Linear(
                config['embedding_size'], config["prefix_mid_dim"]
            ),
            torch.nn.Tanh(),
            nn.Linear(config['prefix_mid_dim'], config['prefix_mid_dim']),
            nn.Tanh(),
            torch.nn.Linear(
                config['prefix_mid_dim'],
                config["num_decoder_layers"]*config["embedding_size"]
            ),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
            #  module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, prefix_tokens: torch.Tensor):
        prefix_embeds = self.prefix_embedding(prefix_tokens)
        past_key_values = self.prefix_weights(prefix_embeds)
        #  print(past_key_values.shape)
        #  prefix_embeds = prefix_embeds.view(
            #  prefix_embeds.size(0), -1, 1024
        #  )
        return past_key_values

# forward embeddings is here
# Positions embeddings are switched
@swappable(layer=TransformerEncoderLayer)
class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.

    For documentation on parameters that are take directly from opt,
    see parlai/agents/transformer/transformer.py

    :param opt: ParlAI-parsed options.
    :param vocabulary_size: Count of tokens/words in the dictionary.
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param str reduction_type: Type of reduction at the end of the encoder.
    :param int n_positions: Size of the position embeddings matrix.
    :param int n_segments: Number of segments/lang/sentence embeddings.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    """

    def __init__(
        self,
        opt: Opt,
        vocabulary_size: int,
        embedding: Optional[nn.Embedding] = None,
        padding_idx: int = 0,
        reduction_type: str = 'mean',
        n_positions: Optional[int] = None,
        n_segments: Optional[int] = None,
        embeddings_scale: Optional[bool] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = None,
        variant: Optional[str] = None,
        output_scaling: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        self.opt = opt
        self.embedding_size = opt['embedding_size']
        self.ffn_size = opt['ffn_size']
        self.n_layers = (
            opt['n_encoder_layers']
            if opt.get('n_encoder_layers', -1) > 0
            else opt['n_layers']
        )
        self.n_heads = opt['n_heads']
        self.dim = self.embedding_size
        self.embeddings_scale = default(
            embeddings_scale, opt.get('embeddings_scale', False)
        )
        self.reduction_type = reduction_type
        self.padding_idx = padding_idx
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout_frac = default(dropout, opt.get('dropout', 0.0))
        self.dropout = nn.Dropout(p=self.dropout_frac)
        self.activation = default(activation, opt.get('activation', 'relu'))
        self.variant = default(variant, opt.get('variant', 'aiayn'))
        self.n_segments = default(n_segments, opt.get('n_segments', 0))

        self.n_positions = default(n_positions, get_n_positions_from_options(opt))
        self.out_dim = self.embedding_size
        assert (
            self.embedding_size % self.n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        # check input formats:
        if embedding is not None:
            assert (
                self.embedding_size is None
                or self.embedding_size == embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            raise AssertionError(
                "This code should not execute. Left here in case we want to enable it."
            )
            assert self.padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, self.embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, self.embedding_size**-0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(self.n_positions, self.embedding_size)
        if not opt.get('learn_positional_embeddings', False):
            create_position_codes(
                self.n_positions,
                self.embedding_size,
                out=self.position_embeddings.weight,
            )
        else:
            nn.init.normal_(
                self.position_embeddings.weight, 0, self.embedding_size**-0.5
            )

        # embedding normalization
        if (
            self.variant == 'xlm'
            or self.variant == 'prelayernorm'
            or self.variant == 'bart'
        ):
            self.norm_embeddings = torch.nn.LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        if self.n_segments >= 1:
            self.segment_embeddings = nn.Embedding(self.n_segments, self.dim)
            nn.init.normal_(self.segment_embeddings.weight, 0, self.dim**-0.5)

        # build the model
        self.layers = self.build_layers()
        self.output_scaling = default(output_scaling, opt.get('output_scaling', 1.0))

    def build_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layer = self.swappables.layer(  # type: ignore
                self.opt,
                attention_dropout=self.opt.get('attention_dropout', 0.0),
                relu_dropout=self.opt.get('relu_dropout', 0.0),
                dropout=self.dropout_frac,
                variant=self.variant,
                activation=self.activation,
            )
            if self.opt.get('checkpoint_activations'):
                layer = checkpoint_wrapper(layer)
            layers.append(fsdp_wrap(layer))
        return layers

    def forward_embedding(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Embed tokens prior to feeding into transformer.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen]:
            If provided, additionally adds ``segments`` as extra embedding features.

        :return (tensor, mask):
            return embedded input and mask
        """
        mask = input != self.padding_idx
        if self.opt["prefix_seq_len"]:
            # Shift the positions because we are added prefix tokens
            positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0) + self.opt["prefix_seq_len"]
            tensor = self.embeddings(input)

        if positions is None:
            positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
            tensor = self.embeddings(input)

        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)

        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        position_embs = self.position_embeddings(positions).expand_as(tensor)
        tensor = tensor + position_embs

        if self.n_segments >= 1:
            if segments is None:
                segments = torch.zeros_like(input)  # type: ignore
            tensor = tensor + self.segment_embeddings(segments)

        return tensor, mask

    def forward_layers(
        self, tensor: torch.Tensor, mask: torch.BoolTensor, **kwargs
    ) -> torch.Tensor:
        """
        Apply transformer layers to input.

        :param tensor:
            embedded input
        :param mask:
            mask of input

        :return tensor:
            return embedding after applying transformer layers
        """
        if getattr(self.layers, 'is_model_parallel', False):
            # factored out for readability. It is equivalent to the other
            # condition
            tensor = self._apply_model_parallel(tensor, mask)
        else:
            for i in range(self.n_layers):
                #  print("encoder idx: ", i)
                if self.opt["prefix_seq_len"] and i in  kwargs["prefix_output"]:
                    past_key_values = kwargs["prefix_output"][i]
                else:
                    past_key_values = None
                tensor = self.layers[i](tensor, mask, past_key_values=past_key_values)

        return tensor

    def reduce_output(
        self, tensor: torch.Tensor, mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, Optional[torch.BoolTensor]]:
        """
        Reduce transformer output at end of forward pass.

        :param tensor:
            encoded input
        :param mask:
            mask for encoded input

        :return (tensor, mask):
            returns the reduced tensor, and mask if appropriate
        """
        tensor *= self.output_scaling
        if self.reduction_type == 'first':
            return tensor[:, 0, :], None
        elif self.reduction_type == 'max':
            return tensor.max(dim=1)[0], None
        elif self.reduction_type == 'mean':
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(tensor)
            output = tensor.sum(dim=1) / divisor
            return output, None
        elif self.reduction_type is None or 'none' in self.reduction_type:
            return tensor, mask
        else:
            raise ValueError(
                "Can't handle --reduction-type {}".format(self.reduction_type)
            )

    def forward(  # type: ignore
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.BoolTensor]]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen] segments:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        # embed input
        tensor, mask = self.forward_embedding(input, positions, segments)

        if self.variant == 'xlm' or self.variant == 'bart':
            tensor = self.norm_embeddings(tensor)

        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor *= mask.unsqueeze(-1).type_as(tensor)

        # apply transformer layers
        # Me: Changes this to pass **kwargs
        tensor = self.forward_layers(tensor, mask, **kwargs)

        if self.variant == 'prelayernorm':
            tensor = self.norm_embeddings(tensor)

        # reduce output
        tensor, out_mask = self.reduce_output(tensor, mask)
        if out_mask is not None:
            return tensor, out_mask
        else:
            return tensor

    def _apply_model_parallel(self, tensor, mask):
        """
        Pipeline application of model parallelism.
        """
        chunks = PipelineHelper.split((tensor, mask))
        work_items = PipelineHelper.schedule_work_items(self.layers, chunks)

        for chunk_idx, layer_nos, next_device in work_items:
            s_tensor, s_mask = chunks[chunk_idx]
            for layer_no in layer_nos:
                s_tensor = self.layers[layer_no](s_tensor, s_mask)
            chunks[chunk_idx] = PipelineHelper.chunk_to((s_tensor, s_mask), next_device)

        tensor_out, mask_out = PipelineHelper.join(chunks)
        return tensor_out

@swappable(encoder=TransformerEncoder)
class TransformerGeneratorModel(TorchGeneratorModel):
    def __init__(self, opt: Opt, dictionary: DictionaryAgent, **kwargs):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx, **kwargs)
        self.embeddings = create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )
        self.encoder = self.build_encoder(
            opt,
            dictionary,
            self.embeddings,
            self.pad_idx,
            reduction_type=None,
            encoder_class=self.swappables.encoder,  # type: ignore
        )


    @classmethod
    def build_encoder(
        cls,
        opt,
        dictionary,
        embedding=None,
        padding_idx=None,
        reduction_type='mean',
        encoder_class: Type[TransformerEncoder] = TransformerEncoder,
        **kwargs,
    ) -> TransformerEncoder:
        return encoder_class(
            opt=opt,
            embedding=embedding,
            vocabulary_size=len(dictionary),
            padding_idx=padding_idx,
            reduction_type=reduction_type,
            **kwargs,
        )

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        if mask is not None:
            mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Reorder the decoder incremental state.

        See ``TorchGeneratorModel.reorder_decoder_incremental_state`` for a description.

        Here, incremental_state is a dict whose keys are layer indices and whose values
        are dicts containing the incremental state for that layer.
        """
        return {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(self.decoder.layers)
        }

    def output(self, tensor):
        """
        Compute output logits.
        """
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        # compatibility with fairseq: fairseq sometimes reuses BOS tokens and
        # we need to force their probability of generation to be 0.
        output[:, :, self.start_idx] = neginf(output.dtype)
        return output


class PrefixEncoder(TransformerGeneratorModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Freeze the embedding
        if self.opt["freeze_encoder"]:
            warn_once("Encoder is Frozen")
            # Freeze the encoder layers
            self.encoder.layers.apply(lambda m: set_requires_grad(m, False))
            self.encoder.norm_embeddings.apply(lambda m: set_requires_grad(m, False))
            self.encoder.position_embeddings.apply(lambda m: set_requires_grad(m, False))
            self.encoder.embeddings.apply(lambda m: set_requires_grad(m, False))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.opt["num_decoder_layers"] = 12 # match_n_layer = 12 # decoder layers
        self.opt["num_decoder_attention_heads"] = 16 # match_n_head"] = 16 # num_decoder_attention_heads
        self.opt["embedding_size"] = 1024 #n_embd = 1024
        self.opt["match_n_embed"] = self.opt["embedding_size"] // 16
        self.opt["prefix_mid_dim"] = 512
        self.opt["n_layer"] = 12
        self.opt["d_model"] = 1024

        if self.opt["prefix_seq_len"]:

            if self.opt["deep_prefix_tuning"]:
                self.prefix_embedding = nn.ModuleList([
                    PrefixEmbedding(self.opt).to(self.device) for _ in range(12)
                ])

                self.prefix_tokens = [torch.arange(
                    self.opt["prefix_seq_len"], device=self.device
                ) for _ in range(12)]
            else:
                self.prefix_embedding = PrefixEmbedding(self.opt).to(self.device)
                self.prefix_tokens = torch.arange(
                    self.opt["prefix_seq_len"], device=self.device
                )


    def _prompt_preprocess(self, batch_size, idx=None):
        if self.opt["deep_prefix_tuning"]:
            prefix_tokens = self.prefix_tokens[idx].unsqueeze(0).expand(batch_size, -1)
            past_key_values = self.prefix_embedding[idx](prefix_tokens)
        else:
            prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
            past_key_values = self.prefix_embedding(prefix_tokens)

        # parlai: dim_per_gead = 64
        past_key_values = past_key_values.view(
            batch_size,
            self.opt["prefix_seq_len"],
            self.opt["num_decoder_layers"]*2,
            self.opt["num_decoder_attention_heads"],
            self.opt["embedding_size"]//32
        )
        # Shape: [batch_size, prefix_seq_len, num_decoder_layers*2, num_decoder_attention_heads, embeddding_size//32]
        #        [8, 8, 12*2, 16, 32]
        # This is what Huggingface expects # batch_size, num_head, seq_len-1, embedding_size

        #  past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        prev_key = past_key_values[0].contiguous()
        prev_value = past_key_values[1].contiguous()

        prev_key = prev_key.view(
            batch_size*self.opt["num_decoder_attention_heads"],
            -1,
            self.opt["embedding_size"] // self.opt["num_decoder_attention_heads"]
        )
        prev_value = prev_value.view(
            batch_size*self.opt["num_decoder_attention_heads"],
            -1,
            self.opt["embedding_size"] // self.opt["num_decoder_attention_heads"]
        )

        assert prev_key.size(1) == self.opt["prefix_seq_len"]
        assert prev_value.size(1) == self.opt["prefix_seq_len"]

        #  prev_key = past_key_values[0].view(batch_size, -1, 64).contiguous()
        #  prev_value = past_key_values[0].view(batch_size, -1, 64).contiguous()
        # Shape [2, 8, 16, 8, 32]
        # as per parlai, this can become [bs*n_heads, -1, dim_per_head]
        # Shape[8*16, -1, 64]

        #  print(past_key_values[0].shape)

        prefix_mask = torch.zeros([batch_size, 1, self.opt["prefix_seq_len"]]).to(self.device)


        # We only want to add prefix tokens for layer 0
        output = PrefixOutput(
            prev_key = prev_key,
            prev_value = prev_value,
            mask = prefix_mask)
        return output

    def get_prompt(self, batch_size):
        #  return self._prompt_preprocess(batch_size, 0)
        if self.opt["deep_prefix_tuning"]:
            return {idx: self._prompt_preprocess(batch_size, idx) for idx in range(12)}
        else:
            return self._get_prompt_preprocess(batch_size)

class PrefixDecoder(TransformerDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.opt["freeze_decoder"]:
            warn_once("Decoder is frozen")
            # Freeze the decoder layers
            self.layers.apply(lambda m: set_requires_grad(m, False))
            self.norm_embeddings.apply(lambda m: set_requires_grad(m, False))
            self.position_embeddings.apply(lambda m: set_requires_grad(m, False))
            self.embeddings.apply(lambda m: set_requires_grad(m, False))

            self.layers[10].encoder_attention.out_lin.apply(lambda m: set_requires_grad(m, True))
            self.layers[10].encoder_attention.out_lin.apply(lambda m: set_requires_grad(m, True))
            self.layers[11].encoder_attention.out_lin.apply(lambda m: set_requires_grad(m, True))


        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.opt["num_decoder_layers"] = 12 # match_n_layer = 12 # decoder layers
        self.opt["num_decoder_attention_heads"] = 16 # match_n_head"] = 16 # num_decoder_attention_heads
        self.opt["embedding_size"] = 1024 #n_embd = 1024
        self.opt["match_n_embed"] = self.opt["embedding_size"] // 16
        self.opt["prefix_mid_dim"] = 512
        self.opt["n_layer"] = 12
        self.opt["d_model"] = 1024

        if self.opt["prefix_seq_len"]:

            if self.opt["deep_prefix_tuning"]:
                self.prefix_embedding = nn.ModuleList([
                    PrefixEmbedding(self.opt).to(self.device) for _ in range(12)
                ])

                self.prefix_tokens = [torch.arange(
                    self.opt["prefix_seq_len"], device=self.device
                ) for _ in range(12)]
            else:
                self.prefix_embedding = PrefixEmbedding(self.opt).to(self.device)
                self.prefix_tokens = torch.arange(
                    self.opt["prefix_seq_len"], device=self.device
                )

    def _prompt_preprocess(self, batch_size, idx=None):
        if self.opt["deep_prefix_tuning"]:
            prefix_tokens = self.prefix_tokens[idx].unsqueeze(0).expand(batch_size, -1)
            past_key_values = self.prefix_embedding[idx](prefix_tokens)
        else:
            prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
            past_key_values = self.prefix_embedding(prefix_tokens)

        # parlai: dim_per_gead = 64
        past_key_values = past_key_values.view(
            batch_size,
            self.opt["prefix_seq_len"],
            self.opt["num_decoder_layers"]*2,
            self.opt["num_decoder_attention_heads"],
            self.opt["embedding_size"]//32
        )
        # Shape: [batch_size, prefix_seq_len, num_decoder_layers*2, num_decoder_attention_heads, embeddding_size//32]
        #        [8, 8, 12*2, 16, 32]
        # This is what Huggingface expects # batch_size, num_head, seq_len-1, embedding_size

        #  past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        prev_key = past_key_values[0].contiguous()
        prev_value = past_key_values[1].contiguous()

        prev_key = prev_key.view(
            batch_size*self.opt["num_decoder_attention_heads"],
            -1,
            self.opt["embedding_size"] // self.opt["num_decoder_attention_heads"]
        )
        prev_value = prev_value.view(
            batch_size*self.opt["num_decoder_attention_heads"],
            -1,
            self.opt["embedding_size"] // self.opt["num_decoder_attention_heads"]
        )

        assert prev_key.size(1) == self.opt["prefix_seq_len"]
        assert prev_value.size(1) == self.opt["prefix_seq_len"]

        #  prev_key = past_key_values[0].view(batch_size, -1, 64).contiguous()
        #  prev_value = past_key_values[0].view(batch_size, -1, 64).contiguous()
        # Shape [2, 8, 16, 8, 32]
        # as per parlai, this can become [bs*n_heads, -1, dim_per_head]
        # Shape[8*16, -1, 64]

        #  print(past_key_values[0].shape)

        prefix_mask = torch.zeros([batch_size, 1, self.opt["prefix_seq_len"]]).to(self.device)


        # We only want to add prefix tokens for layer 0
        output = PrefixOutput(
            prev_key = prev_key,
            prev_value = prev_value,
            mask = prefix_mask)
        return output

    def get_prompt(self, batch_size):
        #  return self._prompt_preprocess(batch_size, 0)
        if self.opt["deep_prefix_tuning"]:
            return {idx: self._prompt_preprocess(batch_size, idx) for idx in range(12)}
        else:
            return self._get_prompt_preprocess(batch_size)

# get_prompt is called here in forward() decoder and generate()
class TransformerDecoderGenerator(PrefixDecoder):
    def __init__(self, opt, *args, **kwargs):
        super().__init__(opt, *args, **kwargs)
        #  self.opt = opt
        self.criterion = self.build_criterion()
        self.show_token_details = opt.get(
            'verbose', False
        ) or 'token_losses' in opt.get('display_add_fields', '')
        self.skip_generation = opt.get('skip_generation', False)
        self.rank_candidates = opt['rank_candidates']
        self.compute_tokenized_bleu = opt.get('compute_tokenized_bleu', False)
        label_truncate = opt.get('label_truncate') or opt.get('truncate')
        self.label_truncate = label_truncate if label_truncate >= 0 else None
        self.beam_size = opt.get('beam_size', 1)
        self.beam_min_length = opt.get('beam_min_length', 1)
        self.beam_context_block_ngram = opt.get('beam_context_block_ngram', -1)
        self.beam_block_ngram = opt.get('beam_block_ngram', -1)
        self.temperature = opt.get('temperature', 1.0)
        assert self.temperature > 0, '--temperature must be greater than 0'
        self.beam_block_list: Optional[SearchBlocklist] = None

    def forward(self, encoder_states, ys):
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(
            1, 0, seqlen - 1
        )  # performs trimming as per seq_len, [16, 79]
        if (ys[:, 0] == self.opt.START_IDX).any():
            raise AssertionError(
                "The Beginning of Sentence token is automatically added to the "
                "label in decode_forced, but you included it in the label. This means "
                "your model will have a double BOS token, which is probably not what "
                "you intended."
            )
        inputs = self._get_initial_forced_decoder_input(
            bsz, inputs
        )  # [16, 79]

        # training time
        #  incr_state = self.get_prompt(bsz)
        if self.opt["prefix_seq_len"]:
            past_key_values = self.get_prompt(bsz)
        else:
            past_key_values = None

        incr_state = None

        latent, _ = super().forward(inputs, encoder_states, incr_state=incr_state,
                                   past_key_values=past_key_values)

        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        #  print("logits: ", logits.shape)

        # see this, this has been changed for prefix tuning
        if logits.size(1) != ys.size(1):
            logits = logits[:, 1:, :]
            preds = preds[:, 1:]

        #  print("here: ", logits.shape)
        logits_view = logits.squeeze(1)
        #  print(logits.shape)
        logits_view = logits.reshape(-1, logits.size(-1))

        #  print(logits_view.shape, ys.shape)

        loss = self.criterion(logits_view, ys.view(-1))
        loss = loss.view(logits.shape[:-1]).sum(dim=1)

        # target tokens compute, not sure
        notnull = ys.ne(self.opt.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)

        return OutputGenerator(
            logits=logits,
            preds=preds,
            loss=loss,
            target_tokens=target_tokens,
            labels=ys,
            notnull=notnull,
            encoder_states=encoder_states,
        )

    def build_criterion(self):
        if not self.opt.fp16:
            return torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        else:
            return FP16SafeCrossEntropy(ignore_index=0, reduction='none')

    def _get_initial_forced_decoder_input(
        self, bsz: int, inputs: torch.LongTensor
    ):
        """
        Return initial input to the decoder.

        Override TGA._get_initial_forced_decoder_input to seed EOS BOS.

        :param bsz:
            batchsize
        :param inputs:
            inputs to decode

        :return initial_input:
        /torch.cat            initial input for the decoder.
        """
        #  end_idx = [self.opt.END_IDX]*8 + [self.opt.END_IDX, self.opt.START_IDX]
        tens = (
            torch.LongTensor([self.opt.END_IDX, self.opt.START_IDX])
            .to(inputs)
            .detach()
            .expand(bsz, 2)
        )
        return torch.cat([tens, inputs], 1)

    def reorder_decoder_incremental_state(
        self,
        incremental_state: Dict[str, Any],
        inds: Union[List[int], torch.LongTensor],
    ) -> Optional[Dict[str, Any]]:
        """
        Incremental state is weird to handle when we seed decoder with two inputs
        initially.
        """
        # we only have this method called when it's actually being used
        assert incremental_state is not None
        assert len(incremental_state) > 0

        for incr_state_l in incremental_state.values():
            assert 'self_attn' in incr_state_l
            assert 'prev_mask' in incr_state_l['self_attn']
            self_attn_mask = incr_state_l['self_attn']['prev_mask']
            # check this is on the very first run with incremental state
            if self_attn_mask.ndim == 3 and tuple(
                self_attn_mask.shape[1:]
            ) == (2, 2):
                # cut off the inappropriate incremental state
                incr_state_l['self_attn']['prev_mask'] = self_attn_mask[
                    :, -1:, :
                ]

        return {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(self.layers)
        }

    def _v2t(self, vec):
        """
        Convert token indices to string of tokens.
        """
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            if i == self.opt.END_IDX:
                break
            elif i != self.opt.START_IDX:
                new_vec.append(i)
        return self.opt.dict.vec2txt(new_vec)

    def output(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute output logits.

        Override standard TGM output to _not_ prevent generation of BOS.
        """
        # tensor.shape -> [batch_size, variable seq len, embedding_size] [16, n, 1024]
        # embeddings.weight.shape -> [50264 (vocab_size), 1024]
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        # output.shape -> [16, n, 50264]

        return output

    def get_prefix_tokens(
        self, batch: MultiTaskBatch
    ) -> Optional[torch.LongTensor]:
        return None

    def evaluate(self, batch, model_output):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None and batch.image is None:
            return
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        cand_scores = None
        token_losses = None
        text_token_info = None

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            #  loss, model_output = self.compute_loss(batch, return_output=True)
            if self.show_token_details:
                token_losses = self._construct_label_token_losses(
                    batch.label_vec, model_output
                )

        beam_preds_scores = None
        preds = None
        if self.skip_generation:
            warn_once("--skip-generation true produces limited metrics")
        else:
            maxlen = self.label_truncate or 256
            prefix_tokens = self.get_prefix_tokens(batch)
            beam_preds_scores, beams = self._generate(
                batch,
                model_output.encoder_states,
                self.beam_size,
                maxlen,
                prefix_tokens=prefix_tokens,
            )
            preds, _, _ = zip(*beam_preds_scores)

            # bsz x beamsize
            beam_texts: List[List[Tuple[str, float]]] = []
            beam_texts_token_info: List[List[List[Tuple]]] = []
            for beam in beams:
                beam_texts.append([])
                if self.show_token_details:
                    beam_texts_token_info.append([])

                for (
                    tokens,
                    score,
                    token_metadata,
                ) in beam.get_rescored_finished():
                    try:
                        if self.show_token_details:
                            beam_texts_token_info[-1].append(
                                self._construct_generated_token_details(
                                    tokens, token_metadata
                                )
                            )
                        beam_texts[-1].append(
                            (self._v2t(tokens), score.item())
                        )
                    except KeyError:
                        logging.error("Decoding error: %s", tokens)
                        continue

        cand_choices = None
        cand_scores = None
        if self.rank_candidates:
            cand_choices, cand_scores = self.rank_eval_label_candidates(
                batch, bsz
            )

        text = (
            [self._v2t(pred_data[0]) for pred_data in beam_preds_scores]
            if beam_preds_scores is not None
            else None
        )

        if self.show_token_details and beam_preds_scores is not None:
            text_token_info = []
            for beam_text_token_info in beam_texts_token_info:
                text_token_info.append(beam_text_token_info[0])

        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
        retval = Output(
            text,
            cand_choices,
            token_losses=token_losses,
            cand_scores=cand_scores,
        )

        if not self.skip_generation:
            retval.beam_texts = beam_texts
            retval.beam_texts_token_info = beam_texts_token_info
            retval.text_token_info = text_token_info
        return retval

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        if mask is not None:
            mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def _construct_label_token_losses(self, labels, model_output):
        # Get non-aggregated losses
        scores, _, _ = model_output
        score_view = scores.reshape(-1, scores.size(-1))
        losses = self.criterion(score_view, labels.view(-1)).view(
            len(labels), -1
        )

        # Zip decoded tokens with losses
        token_losses = []
        for i, label in enumerate(labels):
            token_losses.append(
                list(
                    zip(
                        [self.dict[token] for token in label.tolist()],
                        losses[i].tolist(),
                    )
                )
            )
        return token_losses

    def _generate(
        self,
        batch: Batch,
        encoder_states: torch.LongTensor,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
    ):
        """
        Generate an output with beam search.

        Depending on the options, this may perform greedy/topk/nucleus generation.

        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence
        :param prefix_tokens:
            if given, a tensor of tokens that must begin the decoded sequence.

        :return:
            tuple (beam_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score, token_metadata) tuples for each sample in
              Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """
        #  model = self.model
        #  if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        #  model = self.model.module
        #  encoder_states = model.encoder(*self._encoder_input(batch))
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            assert batch.label_vec is not None, "need label_vec for _generate"
            dev = batch.label_vec.device

        bsz = batch.batchsize
        if batch.text_vec is not None:
            batchsize = batch.batchsize
            batch_context_list = self._get_batch_context(batch).tolist()
            beams = [
                self._treesearch_factory(dev, verbose=self.show_token_details)
                .set_batch_context(batch_context_list, batch_idx)
                .set_block_list(self.beam_block_list)
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [
                self._treesearch_factory(dev, verbose=self.show_token_details)
                for _ in range(bsz)
            ]

        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, beam_size, dev)

        inds = (
            torch.arange(bsz)
            .to(dev)
            .unsqueeze(1)
            .repeat(1, beam_size)
            .view(-1)
        )
        encoder_states = self.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        if self.opt["prefix_seq_len"]:
            past_key_values = self.get_prompt(bsz)
        else:
            past_key_values = None
        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            # This is being done so that conditioning happens on prefix tokens once
            # after which incr_state will take care of maintaining the state
            if _ts != 0:
                past_key_values = None
            #  past_key_values = None
            score, incr_state = super().forward(
                decoder_input,
                encoder_states,
                incr_state=incr_state,
                past_key_values=past_key_values,
            )
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = self.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            if self.temperature != 1.0:
                score.div_(self.temperature)
            # force to fp32 to avoid overflow issues during search calculations
            score = F.log_softmax(score, dim=-1, dtype=torch.float32)  # type: ignore
            if prefix_tokens is not None and _ts < prefix_tokens.size(1):
                # generate prefix_tokens for every timestep that they exist
                # achieve by setting score of all other tokens to be -inf
                prefix_toks = prefix_tokens[:, _ts]
                prefix_mask = torch.ones_like(score, dtype=torch.bool)
                prefix_mask[
                    :, :, prefix_toks
                ] = False  # everything except prefix toks should be neginf
                score[prefix_mask] = neginf(score.dtype)
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = self.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = self._get_next_decoder_input(
                decoder_input, selection, incr_state_inds
            )
            #  print(decoder_input.shape)

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(  # type: ignore
                batch, n_best_beam_preds_scores
            )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [
            n_best_list[0] for n_best_list in n_best_beam_preds_scores
        ]

        return beam_preds_scores, beams

    def _get_batch_context(self, batch):
        """
        Version of TGA._get_context() that operates on full batches for speed.
        """
        if self.beam_context_block_ngram <= 0:
            # We aren't context blocking, return empty tensor of the correct size
            return torch.zeros(batch.batchsize, 0, dtype=torch.long)

        ctxt = batch.text_vec
        if self.beam_block_full_context:
            ctxt = batch.full_text_vec
        return ctxt

    def _get_next_decoder_input(
        self,
        prev_input: torch.LongTensor,
        selection: torch.LongTensor,
        incr_state_inds: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Return next decoder input.

        :param prev_input:
            previous input to decoder
        :param selection:
            token selections for current timestep
        :param inds:
            incremental state indices

        :return decoder input:
            return decoder input for next timestep
        """
        prev_input = torch.index_select(prev_input, 0, incr_state_inds)
        decoder_input = torch.cat([prev_input, selection], dim=-1)
        return decoder_input

    def _add_generation_metrics(self, batch, preds):
        """
        Can be overridden to allow for some metrics on the generations calculated at
        eval.
        """
        self.record_local_metric(
            'gen_n_toks',
            AverageMetric.many([p.size(0) for p in preds], [1] * len(preds)),
        )

    def _compute_fairseq_bleu(self, batch: Batch, preds):
        """
        Compute BLEU score between text and label, using the FAIRSeq BLEU Scorer.

        :param batch:
            Batch of observations
        :param texts:
            list of string predictions
        """
        all_results = []
        label_vec = batch.label_vec
        assert label_vec is not None, "label_vec must exist for fairseq bleu"
        for i, t in enumerate(preds):
            result = FairseqBleuMetric.compute_many(
                t,
                label_vec[i].unsqueeze(0),
                pad_idx=self.NULL_IDX,
                end_idx=self.END_IDX,
                unk_idx=self.dict[self.dict.unk_token],
            )
            if result is None:
                return
            all_results.append(result)

        bleu_scores = list(zip(*all_results))
        for k in range(4):
            self.record_local_metric(f'fairseq_bleu{k + 1}', bleu_scores[k])

    def rank_eval_label_candidates(self, batch, batchsize):
        """
        Rank label_candidates during eval_step.

        Can be overridden to allow for different ways of ranking candidates. Must have
        `--rank-candidates` set to True. By default, we roughly compute PPL to rank the
        candidates.
        """
        # compute roughly ppl to rank candidates
        cand_choices = []
        cand_choices_scores = []
        encoder_states = self.model.encoder(*self._encoder_input(batch))
        for i in range(batchsize):
            num_cands = len(batch.candidate_vecs[i])
            enc = self.model.reorder_encoder_states(
                encoder_states, [i] * num_cands
            )
            cands, _ = self._pad_tensor(batch.candidate_vecs[i], is_label=True)
            cands = cands.to(batch.text_vec.device)
            scores, _ = self.model.decode_forced(enc, cands)
            score_view = scores.reshape(num_cands * cands.size(1), -1)
            cand_losses = F.cross_entropy(
                score_view, cands.view(-1), reduction='none'
            ).view(num_cands, cands.size(1))
            # now cand_losses is cands x seqlen size, but we still need to
            # check padding and such
            mask = (cands != self.NULL_IDX).float()
            cand_scores = (cand_losses * mask).sum(dim=1) / (
                mask.sum(dim=1) + 1e-9
            )
            sorted_scores, ordering = cand_scores.sort()
            cand_choices.append([batch.candidates[i][o] for o in ordering])
            cand_choices_scores.append(sorted_scores.tolist())

        return cand_choices, cand_choices_scores

    def _treesearch_factory(self, device, verbose=False):
        method = self.opt.get('inference', 'greedy')
        beam_size = self.opt.get('beam_size', 1)
        if method == 'greedy':
            return GreedySearch(
                beam_size,
                min_length=0,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.opt.NULL_IDX,
                bos_token=self.opt.START_IDX,
                eos_token=self.opt.END_IDX,
                device=device,
                verbose=verbose,
            )
        elif method == 'beam':
            return BeamSearch(
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.opt.NULL_IDX,
                bos_token=self.opt.START_IDX,
                eos_token=self.opt.END_IDX,
                device=device,
                verbose=verbose,
            )
        elif method == 'delayedbeam':
            return DelayedBeamSearch(
                self.opt['topk'],
                self.opt['beam_delay'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.opt.NULL_IDX,
                bos_token=self.opt.START_IDX,
                eos_token=self.opt.END_IDX,
                device=device,
                verbose=verbose,
            )
        elif method == 'topk':
            return TopKSampling(
                self.opt['topk'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.opt.NULL_IDX,
                bos_token=self.opt.START_IDX,
                eos_token=self.opt.END_IDX,
                device=device,
                verbose=verbose,
            )
        elif method == 'nucleus':
            return NucleusSampling(
                self.opt['topp'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.opt.NULL_IDX,
                bos_token=self.opt.START_IDX,
                eos_token=self.opt.END_IDX,
                device=device,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Can't use inference method {method}")

    def _construct_generated_token_details(self, tokens, tokens_metadata):
        tokens_as_txt = [self.dict[int(token)] for token in tokens]
        return list(zip(tokens_as_txt, tokens_metadata))

    def _get_initial_decoder_input(
        self, bsz: int, beam_size: int, dev: torch.device
    ) -> torch.LongTensor:
        """
        Override to seed decoder with EOS BOS token.
        """
        return (
            torch.LongTensor([self.opt.END_IDX, self.opt.START_IDX])  # type: ignore
            .expand(bsz * beam_size, 2)
            .to(dev)
        )


class MultiTaskBartModel(PrefixEncoder):
    """
    BART Model.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, **kwargs):
        self.opt = opt
        self.opt.dict = dictionary
        self._task_specific_init()
        super().__init__(opt, dictionary, **kwargs)
        self.build_decoder(opt, self.embeddings, dictionary, **kwargs)

    def _task_specific_init(self):
        self.opt.domain_act_list = [
            'None',
            'Taxi-Request',
            'Police-Inform',
            'Hotel-Inform',
            'Hotel-Request',
            'Police-Request',
            'Hospital-Request',
            'Hospital-Inform',
            'general-greet',
            'Restaurant-Request',
            'Attraction-Inform',
            'Restaurant-Inform',
            'Taxi-Inform',
            'Attraction-Request',
            'general-bye',
            'Train-Inform',
            'general-thank',
            'Train-Request',
        ]
        self.opt.entity_list = [
            'none',
            'Attraction-Inform_none',
            'Attraction-Inform_type',
            'Attraction-Inform_area',
            'Attraction-Inform_name',
            'Attraction-Inform_entrancefee',
            'Attraction-Request_phone',
            'Attraction-Request_postcode',
            'Attraction-Request_entrancefee',
            'Attraction-Request_name',
            'Attraction-Request_address',
            'Attraction-Request_type',
            'Attraction-Request_area',
            'Attraction-Request_parking',
            'general-bye_none',
            'general-thank_none',
            'general-greet_none',
            'Restaurant-Inform_booktime',
            'Restaurant-Inform_bookday',
            'Restaurant-Request_ref',
            'Restaurant-Request_address',
            'Restaurant-Request_phone',
            'Restaurant-Request_pricerange',
            'Restaurant-Request_postcode',
            'Restaurant-Request_name',
            'Restaurant-Request_area',
            'Restaurant-Inform_none',
            'Restaurant-Inform_food',
            'Restaurant-Inform_pricerange',
            'Restaurant-Inform_bookpeople',
            'Restaurant-Inform_area',
            'Restaurant-Inform_name',
            'Restaurant-Request_food',
            'Hotel-Inform_none',
            'Hotel-Inform_choice',
            'Hotel-Inform_area',
            'Hotel-Inform_bookpeople',
            'Hotel-Inform_internet',
            'Hotel-Inform_bookday',
            'Hotel-Inform-bookpeople',
            'Hotel-Inform_bookstay',
            'Hotel-Inform_parking',
            'Hotel-Inform_pricerange',
            'Hotel-Inform_name',
            'Hotel-Inform_stars',
            'Hotel-Inform_type',
            'Hotel-Request_pricerange',
            'Hotel-Request_parking',
            'Hotel-Request_address',
            'Hotel-Request_name',
            'Hotel-Request_type',
            'Hospital-Inform_none',
            'Hospital-Inform_department',
            'Hospital-Request_phone',
            'Hospital-Request_name',
            'Hospital-Request_postcode',
            'Hospital-Request_address',
            'Hotel-Request_stars',
            'Hotel-Request_ref',
            'Hotel-Request_area',
            'Hotel-Request_internet',
            'Hotel-Request_phone',
            'Hotel-Request_postcode',
            'Train-Inform_none',
            'Train-Inform_day',
            'Train-Inform_departure',
            'Train-Inform_arriveby',
            'Train-Inform_leaveat',
            'Train-Inform_destination',
            'Train-Inform_bookpeople',
            'Train-Inform_price',
            'Train-Request_ref',
            'Train-Request_name',
            'Train-Request_price',
            'Taxi-Request_name',
            'Train-Request_trainid',
            'Train-Request_duration',
            'Train-Request_leaveat',
            'Train-Request_arriveby',
            'Taxi-Inform_departure',
            'Taxi-Inform_none',
            'Taxi-Inform_destination',
            'Taxi-Inform_leaveat',
            'Taxi-Inform_arriveby',
            'Taxi-Inform_bookpeople',
            'Taxi-Request_phone',
            'Taxi-Request_type',
            'Police-Inform_none',
            'Police-Request_name',
            'Police-Request_address',
            'Police-Request_phone',
            'Police-Request_postcode',
            'Police-Request_department',
        ]

        self.opt.domain_act_dict_label2idx = {
            v: k for k, v in enumerate(self.opt.domain_act_list)
        }
        self.opt.domain_act_dict_idx2label = {
            k: v for k, v in enumerate(self.opt.domain_act_list)
        }
        self.opt.entity_dict_idx2label = {
            k: v for k, v in enumerate(self.opt.entity_list)
        }
        self.opt.entity_dict_label2idx = {
            v: k for k, v in enumerate(self.opt.entity_list)
        }

        self.opt.fp16 = self.opt['fp16']
        self.opt.NULL_IDX = 0
        self.opt.START_IDX = 1
        self.opt.END_IDX = 2

    def build_decoder(self, opt, embedding, dictionary, **kwargs):
        if not self.opt['disable_classification_decoder']:
            self.classification_decoder_1 = TransformerDecoderClassifier(opt=opt)
        else:
            warn_once("Classification Decoder is frozen")
            self.classification_decoder_1 = None
        if not self.opt['disable_pretrained_decoder']:
            self.decoder = TransformerDecoderGenerator(
                opt, embedding, dictionary
            )
        else:
            self.decoder = None

    def forward(
        self,
        *xs,
        ys_dst=None,
        ys_dialog_act=None,
        prev_enc=None,
        maxlen=None,
        bsz=None,
    ):
        assert (
            ys_dst is not None
        ), "Greedy decoding in TGModel.forward no longer supported."
        self.longest_label = max(self.longest_label, ys_dst.size(1))

        # use cached encoding if available
        bsz = ys_dst.size(0)
        if self.opt["prefix_seq_len"]:
            prefix_output = self.get_prompt(bsz)
        else:
            prefix_output = None

        encoder_states = (
            prev_enc if prev_enc is not None else self.encoder(*xs, prefix_output=prefix_output)
        )

        # use teacher forcing

        # This is being done to take into account that only some decoders
        # might be enabled
        generative_model_output = None
        classification_model_output_1 = None

        if self.decoder is not None:
            generative_model_output = self.decoder(encoder_states, ys_dst)

        if self.classification_decoder_1 is not None:
            classification_model_output_1 = self.classification_decoder_1(
                encoder_states[0], ys_dialog_act
            )  # encoder state is a tuple, classifier needs only the first element

        global_model_output = GlobalModelOutput(
            classification_decoder_1=classification_model_output_1,
            decoder=generative_model_output,
        )
        return global_model_output



from parlai.core.nash import NashMTL
class PromptBartAgent(BartAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('Multi Task Bart Args')
        group.add_argument(
            '--not_load_decoder_pretrained_weights',
            type='bool',
            default=False,
            help='whether to use pre-trained weights of original BART decoder',
        )
        group.add_argument('--fp16', type=str, default=True, help='use fp16')
        parser.add_argument(
            '--disable_classification_decoder',
            type=bool,
            default=False,
            help='disable classification decoder',
        )
        parser.add_argument(
            '--disable_pretrained_decoder',
            type=bool,
            default=False,
            help='whether to use pre-trained weights of original BART decoder',
        )
        parser.add_argument(
            '--freeze_encoder',
            type=bool,
            default=False,
        )
        parser.add_argument(
            '--freeze_decoder',
            type=bool,
            default=False
        )
        parser.add_argument(
            '--prefix_seq_len',
            type=int,
            default=None
        )
        parser.add_argument(
            '--deep_prefix_tuning',
            type=bool,
            default=False
        )

        parser.add_argument(
            '--loss_method',
            type=str,
            default=None
        )

        return parser

    def build_model(self) -> MultiTaskBartModel:
        """
        Build and return model.
        """
        model = MultiTaskBartModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def compute_loss(self, batch, return_output=False):
        if batch.label_vec_dst is None:
            raise ValueError('Cannot compute loss without a label.')
        global_model_output = self.model(
            *self._model_input(batch),
            ys_dst=batch.label_vec_dst,
            ys_dialog_act=batch.label_vec_dialog_act,
        )
        loss_decoders = []

        if global_model_output.classification_decoder_1 is not None:
            output = global_model_output.classification_decoder_1

            self.record_local_metric(
                'loss_classification_decoder_1',
                AverageMetric.many([output.loss] * len(batch.valid_indices)),
            )
            loss_decoders.append(output.loss)

        if global_model_output.decoder is not None:
            output = global_model_output.decoder
            notnull = output.notnull
            correct = ((batch.label_vec == output.preds) * notnull).sum(dim=-1)
            self.record_local_metric(
                'loss_decoder',
                AverageMetric.many(output.loss, output.target_tokens),
            )
            # perplexity
            self.record_local_metric(
                'ppl', PPLMetric.many(output.loss, output.target_tokens)
            )
            # token-wise accuracy
            self.record_local_metric(
                'token_acc', AverageMetric.many(correct, output.target_tokens)
            )
            # utterance-wise exact match
            self.record_local_metric(
                'token_em', AverageMetric.many(correct == output.target_tokens)
            )
            loss_decoder = output.loss.sum()
            loss_decoder = (loss_decoder) / (output.target_tokens.sum())
            loss_decoders.append(loss_decoder)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        #  loss_weights = F.softmax(torch.randn(2), dim=-1).to(device)
        #  loss = (loss_weights[0] *loss_decoders[0] + loss_weights[1]*loss_decoders[1])
        if self.opt["loss_method"] == "sum":
            loss = sum(loss_decoders)
        if self.opt["loss_method"] == "individual":
            loss = loss_decoders
        #  loss_fn = NashMTL(n_tasks=2, device=device)
        #  loss_output = loss_fn.get_loss(
            #  loss_decoders,
            #  shared_parameters=[
                #  self.model.prefix_embedding.parameters(),
                #  self.model.prefix_embedding.parameters()
            #  ],
            #  task_specific_parameters=[
                #  self.model.classification_decoder_1.parameters(),
                #  self.model.decoder.prefix_embedding.parameters()
            #  ]
        #  )
        #  loss = loss_output[0] # loss_output[1] are the weights)

#          self.record_local_metric(
            #  'Combined Loss',
            #  AverageMetric.many([loss] * (len(batch.valid_indices))),
#          )
        # actually do backwards loss
        #  loss = loss.sum()
        #  loss /= target_tokens.sum()  # average loss per token
        if return_output:
            return (loss, global_model_output)
        else:
            return loss

    def load_state_dict(self, state_dict):
        output = self.model.load_state_dict(state_dict, strict=False)
        if len(output.unexpected_keys) > 1:
            warn_once(
                "The weights seems to have keys which cannot be loaded, this is unexpected if you want to load pre-trained weights, training will terminate now"
            )
            if self.opt['disable_pretrained_decoder']:
                warn_once(
                    "Decoder is not being loaded up with pre-trained weights"
                )
            else:
                warn_once(
                    "If training needs to be performed without loading decoder pre-trained weights, pass the --not_load_decoder_pretrained_weights"
                )
                exit(0)
        if len(output.missing_keys) > 1:
            warn_once(
                "New keys have been added to existing model weights, this is expected if you want to make modifications to model architecture which is supposed to load with pre-trained weights"
            )

    def _set_label_vec(self, obs, add_start, add_end, label_truncate):
        if "dialog_act" in obs:
            obs = super()._set_label_vec(
                obs, add_start, add_end, label_truncate
            )
            # there can be multiple dialog acts, we take the first one
            dialog_act_entry = obs["dialog_act"]
            domain_act_list, entity_list = [], []

            for domain_act_, values in dialog_act_entry.items():
                domain_act_list.append(domain_act_)
                for entity_, slot_value in values:
                    entity_list.append(entity_)

            if not entity_list:
                entity_list.append('none')

            domain_act_indices = [
                self.opt.domain_act_dict_label2idx[x] for x in domain_act_list
            ]
            entity_indices = [
                self.opt.entity_dict_label2idx[x] for x in entity_list
            ]

            domain_act_multi_hot_label = 0
            for x in domain_act_indices:
                one_hot = F.one_hot(
                    torch.tensor(x), len(self.opt.domain_act_dict_label2idx)
                )
                domain_act_multi_hot_label += one_hot

            entity_multi_hot_label = 0

            for x in entity_indices:
                one_hot = F.one_hot(
                    torch.tensor(x), len(self.opt.entity_dict_label2idx)
                )
                entity_multi_hot_label += one_hot

            if self.use_cuda:
                domain_act_multi_hot_label = domain_act_multi_hot_label.cuda()
                entity_multi_hot_label = entity_multi_hot_label.cuda()

            obs["domain_act_vec"] = domain_act_multi_hot_label
            obs["entity_vec"] = entity_multi_hot_label

        return obs

    def batchify(self, obs_batch, sort=False):
        """
        Manage dialog act labels. Adds them in Batch namedtuple
        """
        if len(obs_batch) == 0:
            return Batch(batchsize=0)

        valid_obs = [
            (i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)
        ]

        if len(valid_obs) == 0:
            return Batch(batchsize=0)

        valid_inds, exs = zip(*valid_obs)

        # TEXT
        xs = x_lens = context_original_lengths = None
        context_truncate_rate = context_truncated_lengths = None
        if any(ex.get('text_vec') is not None for ex in exs):
            if any('context_original_length' in ex for ex in exs):
                context_truncate_rate = torch.LongTensor(
                    [ex.get('context_truncate_rate', 0) for ex in exs]
                )
                context_original_lengths = torch.LongTensor(
                    [ex.get('context_original_length', 0) for ex in exs]
                )
            if any('context_truncated_length' in ex for ex in exs):
                context_truncated_lengths = torch.LongTensor(
                    [ex.get('context_truncated_length', 0) for ex in exs]
                )
            _xs = [ex.get('text_vec', self.EMPTY) for ex in exs]
            xs, x_lens = self._pad_tensor(_xs)
            if sort:
                sort = False  # now we won't sort on labels
                xs, x_lens, valid_inds, exs = argsort(
                    x_lens, xs, x_lens, valid_inds, exs, descending=True
                )

        # LABELS
        labels_avail = any('labels_vec' in ex for ex in exs)
        some_labels_avail = labels_avail or any(
            'eval_labels_vec' in ex for ex in exs
        )

        ys_dst = y_dst_lens = labels_dst = label_dst_original_lengths = None
        label_dst_truncate_rate = label_dst_truncated_lengths = None
        if some_labels_avail:
            if any('label_original_length' in ex for ex in exs):
                label_dst_truncate_rate = torch.LongTensor(
                    [ex.get('label_truncate_rate', 0) for ex in exs]
                )
                label_dst_original_lengths = torch.LongTensor(
                    [ex.get('label_original_length', 0) for ex in exs]
                )
            if any('label_truncated_length' in ex for ex in exs):
                label_dst_truncated_lengths = torch.LongTensor(
                    [ex.get('label_truncated_length', 0) for ex in exs]
                )
            field = 'labels' if labels_avail else 'eval_labels'

            domain_act_field = 'domain_act'
            entity_field = 'entity'

            # generative labels
            label_vecs_dst = [ex.get(field + '_vec', self.EMPTY) for ex in exs]
            labels_dst = [ex.get(field + '_choice') for ex in exs]

            # classifier labels
            label_vecs_domain_act = torch.stack(
                [ex.get(domain_act_field + '_vec', self.EMPTY) for ex in exs]
            )
            label_vecs_entity = torch.stack(
                [ex.get(entity_field + '_vec', self.EMPTY) for ex in exs]
            )

            labels_domain_act = [
                ex.get(domain_act_field + '_choice') for ex in exs
            ]
            labels_entity = [ex.get(entity_field + '_choice') for ex in exs]

            y_dst_lens = [y.shape[0] for y in label_vecs_dst]
            ys_dst, y_dst_lens = self._pad_tensor(
                label_vecs_dst, is_label=True
            )

            if sort and xs is None:
                (
                    ys_dst,
                    valid_inds,
                    label_vecs_dst,
                    labels_dst,
                    y_dst_lens,
                ) = argsort(
                    y_dst_lens,
                    ys_dst,
                    valid_inds,
                    label_vecs_dst,
                    labels_dst,
                    y_dst_lens,
                    descending=True,
                )

        # LABEL_CANDIDATES
        cands, cand_vecs = None, None
        if any('label_candidates_vecs' in ex for ex in exs):
            cands = [ex.get('label_candidates', None) for ex in exs]
            cand_vecs = [ex.get('label_candidates_vecs', None) for ex in exs]

        # IMAGE
        imgs = None
        if any('image' in ex for ex in exs):
            imgs = [ex.get('image', None) for ex in exs]

        # reward
        rewards = None
        if any('reward' in ex for ex in exs):
            rewards = torch.Tensor([ex.get('reward', 0) for ex in exs])

        # make sure we're only passing around tensors
        valid_inds = torch.LongTensor(valid_inds)

        is_training = any('labels' in obs for obs in obs_batch)

        return MultiTaskBatch(
            batchsize=len(valid_inds),
            is_training=is_training,
            text_vec=xs,
            label_vec_dst=ys_dst,
            labels_dst=labels_dst,
            label_vec_dialog_act=label_vecs_entity,
            labels_dialog_act=labels_entity,
            valid_indices=valid_inds,
            candidates=cands,
            candidate_vecs=cand_vecs,
            image=imgs,
            rewards=rewards,
            observations=exs if self.is_debug else None,
            _context_original_length=context_original_lengths,
            _context_truncate_rate=context_truncate_rate,
            _context_truncated_length=context_truncated_lengths,
            _label_original_length=label_dst_original_lengths,
            _label_truncate_rate=label_dst_truncate_rate,
            _label_truncated_length=label_dst_truncated_lengths,
        )

    def eval_step(self, batch):
        """
        Depending on which decoder is active, runs evaluation accordingly
        """

        # for classification
        self.model.eval()
        global_model_output = self.model(
            *self._model_input(batch),
            ys_dst=batch.label_vec_dst,
            ys_dialog_act=batch.label_vec_dialog_act,
        )  # this uses forward of BartModel

        output = {}

        if global_model_output.classification_decoder_1:
            model_output = global_model_output.classification_decoder_1
            model_output = self.model.classification_decoder_1.evaluate(
                model_output
            )

            res = []

            for pred, label in zip(
                model_output.predictions, model_output.labels
            ):
                if pred == label:
                    res.append(1)
                else:
                    res.append(0)

            self.record_local_metric(
                'accuracy_entity',
                AverageMetric.many(res),
            )

        if global_model_output.decoder:
            model_output = self.model.decoder.evaluate(
                batch, global_model_output.decoder
            )
            for k, v in model_output.items():
                output[
                    k
                ] = v  # text, text_candidates, token_losses, cand_scores, beam_texts, beam_texts_token_info, text_token_info

        return output

    def train_step(self, batch):
        self._cache_dummy_batch(batch)
        self.model.train()
        self.zero_grad()

        try:
            loss = self.compute_loss(batch)
            if isinstance(loss, list):
                self.backward(loss[0], retain_graph=True)
                self.backward(loss[1])
            else:
                self.backward(loss)
            self.update_params()
            oom_sync=False
        except RuntimeError as e:
            if 'out_of_memory' in str(e):
                oom_sync = True
                logging.error(
                    'Ran out of memory, skipping batch. '
                    'if this happens frequently, decrease batchsize or '
                    'truncate the inputs to the model.'
                )
                self.global_metrics.add('skipped_batches', SumMetric(1))
            else:
                raise e

        if oom_sync:
            self._fake_forward_backward_pass()
### ALL GOOD
####
class TransformerPooler(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.dense = nn.Linear(opt['embedding_size'], opt['embedding_size'])
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class TransformerDecoderClassifier(nn.Module):
    def __init__(self, opt: Opt, **kwargs):
        super().__init__()
        self.opt = opt
        self.num_domain_act = len(self.opt.domain_act_dict_label2idx)
        self.num_entities = len(self.opt.entity_dict_label2idx)
        self.build_model()
        self.pooler = TransformerPooler(opt)
        self.criterion = self.build_criterion()

    def build_model(self):
        self.non_linear = F.relu

        dim = self.opt['embedding_size']
        dim_hidden = self.opt['ffn_size']

        self.lin1 = nn.Linear(dim, dim_hidden // 4)
        self.lin2 = nn.Linear(dim_hidden // 4, dim_hidden // 8)
        self.lin3 = nn.Linear(dim_hidden // 8, self.num_entities)
        #  self.lin2 = nn.Linear(dim_hidden, self.num_domain_act)
        #  self.lin3 = nn.Linear(dim_hidden, self.num_entities)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.xavier_uniform_(self.lin3.weight)

    def build_criterion(self):
        return torch.nn.BCEWithLogitsLoss(reduction='mean')
        #  if not self.opt.fp16: return torch.nn.CrossEntropyLoss(ignore_index=self.opt.NULL_IDX, reduction='none')
        #  else: return FP16SafeCrossEntropy(ignore_index=self.opt.NULL_IDX, reduction='none')

    def forward(self, encoder_state, ys_dialog_act):
        ys_entity = ys_dialog_act

        x = self.pooler(encoder_state)
        logits = self.lin3(
            self.non_linear(self.lin2(self.non_linear(self.lin1(x))))
        )
        loss = self.criterion(logits, ys_entity.float())

        output = OutputClassifier(logits=logits, loss=loss, labels=ys_entity)
        return output

    def evaluate(self, model_output):
        output_entity = torch.sigmoid(model_output.logits)

        predicted_entity = np.round(output_entity.detach().cpu()).int()

        def get_accuracy_generative(x, y):
            assert x.shape == y.shape
            predicted_labels, entity_labels = [], []
            for _x, _y in zip(x, y):
                batch_predictions = []
                batch_gt = []
                _pred_indices = torch.where(_x == 1)[0].tolist()
                _gt_indices = torch.where(_y == 1)[0].tolist()

                # this is being done because _pred_indices might not be equal to _gt_indices
                for gt_indice in _gt_indices:
                    gt_val_str = self.opt.entity_dict_idx2label[gt_indice]
                    gt_val_str = " ".join(gt_val_str.split("-"))
                    gt_val_str = " ".join(gt_val_str.split("_"))

                    batch_gt.append(gt_val_str)

                for pred_indice in _pred_indices:
                    predicted_val_str = self.opt.entity_dict_idx2label[
                        pred_indice
                    ]
                    predicted_val_str = " ".join(predicted_val_str.split("-"))
                    predicted_val_str = " ".join(predicted_val_str.split("_"))

                    batch_predictions.append(predicted_val_str)

                predicted_labels.append(', '.join(batch_predictions))
                entity_labels.append(', '.join(batch_gt))

            return predicted_labels, entity_labels

        detached_entity_labels = model_output.labels.detach().cpu()

        predictions, labels = get_accuracy_generative(
            predicted_entity, detached_entity_labels
        )

        model_output.predictions, model_output.labels = predictions, labels

        return model_output

@dataclass
class PrefixOutput:
    prev_key: torch.Tensor
    prev_value: torch.Tensor
    mask: torch.Tensor

