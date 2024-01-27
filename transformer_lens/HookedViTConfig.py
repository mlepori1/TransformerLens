"""
@author: alexatartaglini

Hooked ViT Config.

Module with a dataclass for storing the configuration of a
:class:`transformer_lens.HookedViT` model.
"""
from __future__ import annotations

import logging
import pprint
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from transformer_lens import utils

SUPPORTED_ACTIVATIONS = [
    "relu",
    "gelu",
    "silu",
    "gelu_new",
    "solu_ln",
    "gelu_fast",
    "quick_gelu",
]


@dataclass
class HookedViTConfig:
    """
    Configuration class to store the configuration of a HookedViT model.

    See further_comments.md for more details on the more complex arguments.

    Args:
        d_model (int): The dimensionality of the embeddings.
        d_head (int): The dimensionality of each attention head.
        n_layers (int): The number of transformer blocks (one block = one attn layer AND one MLP layer).
        n_heads (int): The number of attention heads. If not
            specified, will be set to d_model // d_head. (This is represented by a default value of -1)
        d_mlp (int, *optional*): The dimensionality of the feedforward mlp
            network. Defaults to 4 * d_model, and in an attn-only model is None.
        act_fn (str, *optional*): The activation function to use. Always
            lowercase. Supports ['relu', 'gelu', 'silu', 'gelu_new', 'solu_ln',
            'gelu_fast', 'quick_gelu']. Must be set unless using an attn-only model.
        eps (float): The epsilon value to use for layer normalization. Defaults
            to 1e-5
        use_attn_result (bool): whether to explicitly calculate the amount
            each head adds to the residual stream (with a hook) and THEN add it
            up, vs just calculating the sum. This can be very memory intensive
            for large models, so defaults to False
        use_split_qkv_input (bool): whether to explicitly calculate the input of
            each head separately, with a hook. Defaults to false to save memory.
        use_hook_mlp_in (bool): whether to use a hook to get the input to the
            MLP layer. Defaults to false to save memory.
        use_attn_in (bool): whether to explicitly calculate the input of each
            attention head separately, with a hook. Defaults to false to save memory
        use_attn_scale (bool): whether to scale the attention weights by
            1/sqrt(d_head)
        model_name (str): the name of the model, used to load
            weights from HuggingFace or initialized to "custom" if not passed
        original_architecture (str, *optional*): the family of the model, used
        to help load
            weights from HuggingFace or initialized to "custom" if not passed
        from_checkpoint (bool): Whether the model weights were
            loaded from a checkpoint (only applies to pretrained models)
        checkpoint_index (int, *optional*): The index of the
            checkpoint loaded (only applies to pretrained models).
        checkpoint_label_type (str, *optional*): Whether
            checkpoints are labelled by the number of steps or number of tokens.
        checkpoint_value (int, *optional*): The value of the
            checkpoint label (whether of steps or tokens).
        init_mode (str): the initialization mode to use for the
            weights. Only relevant for custom models, ignored for pre-trained.
            Currently the only supported mode is 'gpt2', where biases are
            initialized to 0 and weights are standard normals of range
            initializer_range.
        normalization_type (str, *optional*): the type of normalization to use.
            Options are None (no normalization), 'LN' (use LayerNorm, including weights
            & biases) and 'LNPre' (use LayerNorm, but no weights & biases).
            Defaults to LN
        device(str): The device to use for the model. Defaults to 'cuda' if
            available, else 'cpu'. Must be 'cuda' if `n_devices` > 1.
        n_devices (int): The number of devices to use for the model. Defaults to 1. Layers are loaded
            to support "pipeline parallelism", where each device is responsible for a subset of the layers.
        attn_only (bool): Whether to only use attention layers, no feedforward
            layers. Defaults to False
        seed (int, *optional*): The seed to use for the model.
            Used to set sources of randomness (Python, PyTorch and
            NumPy) and to initialize weights. Defaults to None. We recommend setting a seed, so your experiments are reproducible.
        initializer_range (float): The standard deviation of the normal used to
            initialise the weights, initialized to 0.8 / sqrt(d_model) .
        init_weights (bool): Whether to initialize the weights. Defaults to
            True. If False, does not initialize weights.
        positional_embedding_type (str): The positional embedding used. Options
            are 'standard' (ie GPT-2 style, absolute, randomly initialized learned positional
            embeddings, directly added to the residual stream), 'rotary'
            (described here: https://blog.eleuther.ai/rotary-embeddings/ ) and
            'shortformer' (GPT-2 style absolute & learned, but rather than being
            added to the residual stream they're only added to the inputs to the
            keys and the queries (ie key = W_K(res_stream + pos_embed), but
            values and MLPs don't get any positional info)). Sinusoidal are not
            currently supported. Defaults to 'standard'.
        num_labels (int, *optional*): Number of labels to use in the last layer added to the model,
            typically for a classification task. Defaults to 1 (binary classification).
        id2label (Dict[int, str], *optional*): a map from index (for instance prediction index, or target index) to label.
        label2id: (Dict[str, int], *optional*): A map from label to index for the model.
        n_params (int, *optional*): The number of (hidden weight)
            parameters in the model. This is automatically calculated and not
            intended to be set by the user. (Non embedding parameters, because
            the [scaling laws paper](https://arxiv.org/pdf/2001.08361.pdf) found
            that that was a more meaningful number. Ignoring biases and layer
            norms, for convenience)
        use_hook_tokens (bool): Will add a hook point on the token input to
            HookedTransformer.forward, which lets you cache or intervene on the tokens.
            Defaults to False.
        dtype (torch.dtype, *optional*): The model's dtype. Defaults to torch.float32.
        post_embedding_ln (bool): Whether to apply layer normalization after embedding the tokens. Defaults
            to False.
        image_size (int, *optional*): The size (resolution) of each image. Defaults to 224.
        patch_size (int, *optional*): The size (resolution) of each patch. Defaults to 16.
        num_channels (int, *optional*): The number of input channels. Defaults to 3.
        is_clip (bool, *optional*): Whether we're loading from a CLIP model
        force_projection_bias (bool, *optional*): Whether to force the visual projection to have a bias term

    """

    n_layers: int
    d_model: int
    d_head: int
    model_name: str = "custom"
    n_heads: int = -1
    d_mlp: Optional[int] = None
    act_fn: Optional[str] = None
    eps: float = 1e-12
    use_attn_result: bool = False
    use_attn_scale: bool = True
    use_split_qkv_input: bool = False
    use_hook_mlp_in: bool = False
    use_attn_in: bool = False
    original_architecture: Optional[str] = None
    from_checkpoint: bool = False
    checkpoint_index: Optional[int] = None
    checkpoint_label_type: Optional[str] = None
    checkpoint_value: Optional[int] = None
    init_mode: str = "vit"
    normalization_type: Optional[str] = "LN"
    device: Optional[str] = None
    n_devices: int = 1
    attn_only: bool = False
    seed: Optional[int] = None
    initializer_range: float = -1.0
    init_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    positional_embedding_type: str = "standard"
    num_labels: int = 1
    id2label: Dict[int, str] = None
    label2id: Dict[str, int] = None
    n_params: Optional[int] = None
    use_hook_tokens: bool = False
    parallel_attn_mlp: bool = False
    dtype: torch.dtype = torch.float32
    attention_dir: str = "bidirectional"
    use_local_attn: bool = False
    gated_mlp: bool = False
    default_prepend_bos: bool = False
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    n_ctx: int = 196
    is_clip: bool = False
    force_projection_bias: bool = False

    def __post_init__(self):
        assert (
            self.image_size % self.patch_size == 0
        ), f"patch_size {self.patch_size} is invalid for image_size {self.image_size}"

        if self.n_heads == -1:
            self.n_heads = self.d_model // self.d_head

            if not self.d_model % (self.d_head) == 0:
                logging.warning(
                    f"d_model {self.d_model} is not divisible by d_head {self.d_head}. n_heads was inferred to be {self.n_heads}, rounding down the ratio."
                )

        if self.seed is not None:
            self.set_seed_everywhere(self.seed)

        if not self.attn_only:
            if self.d_mlp is None:
                # For some reason everyone hard codes in this hyper-parameter!
                self.d_mlp = self.d_model * 4
            assert (
                self.act_fn is not None
            ), "act_fn must be specified for non-attn-only models"
            assert (
                self.act_fn in SUPPORTED_ACTIVATIONS
            ), f"act_fn={self.act_fn} must be one of {SUPPORTED_ACTIVATIONS}"

        if self.initializer_range < 0:
            # Roughly copy the GPT-2 value, but proportional to sqrt(1/d_model)
            self.initializer_range = 0.8 / np.sqrt(self.d_model)

        # The number of parameters in attention layers (ignoring biases and layer norm). 4 because W_Q, W_K, W_V and W_O
        self.n_params = self.n_layers * (
            (self.d_model * self.d_head * self.n_heads * 4)
        )

        if not self.attn_only:
            # Number of parameters in MLP layers (ignoring biases and layer norm). 2 because W_in and W_out
            self.n_params += self.n_layers * self.d_model * self.d_mlp * 2

        if self.device is None:
            self.device = utils.get_device()

        if self.n_devices > 1:
            assert (
                torch.cuda.device_count() >= self.n_devices
            ), f"Not enough CUDA devices to support n_devices {self.n_devices}"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> HookedViTConfig:
        """
        Instantiates a `HookedViTConfig` from a Python dictionary of
        parameters.
        """
        return cls(**config_dict)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return "HookedViTConfig:\n" + pprint.pformat(self.to_dict())

    def set_seed_everywhere(self, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
