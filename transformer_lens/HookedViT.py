"""
@author: alexatartaglini

Hooked ViT.

Closely follows the implementation of :class:`transformer_lens.HookedEncoder` with a few 
vision-specific alterations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union, cast, overload

import torch
from einops import repeat
from jaxtyping import Float, Int
from torch import nn
from transformers import AutoTokenizer
from typing_extensions import Literal

import transformer_lens.loading_from_pretrained as loading
from transformer_lens import ActivationCache, FactoredMatrix, HookedViTConfig
from transformer_lens.components import TransformerBlock, ViTEmbed, ViTHead
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.utilities import devices


class HookedEncoder(HookedRootModule):
    """
    TODO: rewrite
    This class implements a BERT-style encoder using the components in ./components.py, with HookPoints on every interesting activation. It inherits from HookedRootModule.

    Limitations:
    - The current MVP implementation supports only the masked language modelling (MLM) task. Next sentence prediction (NSP), causal language modelling, and other tasks are not yet supported.
    - Also note that model does not include dropouts, which may lead to inconsistent results from training or fine-tuning.

    Like HookedTransformer, it can have a pretrained Transformer's weights loaded via `.from_pretrained`. There are a few features you might know from HookedTransformer which are not yet supported:
        - There is no preprocessing (e.g. LayerNorm folding) when loading a pretrained model
        - The model only accepts tokens as inputs, and not strings, or lists of strings
    """

    def __init__(self, cfg, tokenizer=None, move_to_device=True, **kwargs):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedViTConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedViTConfig object. If you want to load a pretrained model, use HookedViT.from_pretrained() instead."
            )
        self.cfg = cfg

        assert (
            self.cfg.n_devices == 1
        ), "Multiple devices not supported for HookedViT"

        self.embed = ViTEmbed(self.cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.cfg) for _ in range(self.cfg.n_layers)]
        )
        self.classifier_head = ViTHead(cfg)

        self.hook_full_embed = HookPoint()

        if move_to_device:
            self.to(self.cfg.device)

        self.setup()

    @overload
    def forward(
        self,
        input: Int[torch.Tensor, "batch num_channels height width"],
        return_type: Literal["logits"],
        token_type_ids: Optional[Int[torch.Tensor, "batch pos"]] = None,
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos num_labels"]:
        ...

    @overload
    def forward(
        self,
        input: Int[torch.Tensor, "batch num_channels height width"],
        return_type: Literal[None],
        token_type_ids: Optional[Int[torch.Tensor, "batch pos"]] = None,
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Optional[Float[torch.Tensor, "batch pos num_labels"]]:
        ...

    def forward(
        self,
        input: Int[torch.Tensor, "batch num_channels height width"],
        return_type: Optional[str] = "logits"
    ) -> Optional[Float[torch.Tensor, "batch pos num_labels"]]:
        """Input must be a batch of images. 

        return_type Optional[str]: The type of output to return. Can be one of: None (return nothing, don't calculate logits), or 'logits' (return logits).
        """

        ims = input

        if ims.device.type != self.cfg.device:
            ims = ims.to(self.cfg.device)

        resid = self.hook_full_embed(self.embed(ims))

        for block in self.blocks:
            resid = block(resid)
        logits = self.classifier_head(resid)

        if return_type is None:
            return

        return logits

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[True] = True, **kwargs
    ) -> Tuple[Float[torch.Tensor, "batch pos num_labels"], ActivationCache]:
        ...

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[False] = False, **kwargs
    ) -> Tuple[Float[torch.Tensor, "batch pos num_labels"], Dict[str, torch.Tensor]]:
        ...

    def run_with_cache(
        self,
        *model_args,
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[
        Float[torch.Tensor, "batch pos num_labels"],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """
        Wrapper around run_with_cache in HookedRootModule. If return_cache_object is True, this will return an ActivationCache object, with a bunch of useful HookedTransformer specific methods, otherwise it will return a dictionary of activations as in HookedRootModule. This function was copied directly from HookedTransformer.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(
                cache_dict, self, has_batch_dim=not remove_batch_dim
            )
            return out, cache
        else:
            return out, cache_dict

    def to(
        self,
        device_or_dtype: Union[torch.device, str, torch.dtype],
        print_details: bool = True,
    ):
        return devices.move_to_and_update_config(self, device_or_dtype, print_details)

    def cuda(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cuda")

    def cpu(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cpu")

    def mps(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("mps")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        checkpoint_index: Optional[int] = None,
        checkpoint_value: Optional[int] = None,
        hf_model=None,
        device: Optional[str] = None,
        tokenizer=None,
        move_to_device=True,
        dtype=torch.float32,
        **from_pretrained_kwargs,
    ) -> HookedEncoder:
        """Loads in the pretrained weights from huggingface. Currently supports loading weight from HuggingFace BertForMaskedLM. Unlike HookedTransformer, this does not yet do any preprocessing on the model."""
        logging.warning(
            "Support for BERT in TransformerLens is currently experimental, until such a time when it has feature "
            "parity with HookedTransformer and has been tested on real research tasks. Until then, backward "
            "compatibility is not guaranteed. Please see the docs for information on the limitations of the current "
            "implementation."
            "\n"
            "If using BERT for interpretability research, keep in mind that BERT has some significant architectural "
            "differences to GPT. For example, LayerNorms are applied *after* the attention and MLP components, meaning "
            "that the last LayerNorm in a block cannot be folded."
        )

        assert not (
            from_pretrained_kwargs.get("load_in_8bit", False)
            or from_pretrained_kwargs.get("load_in_4bit", False)
        ), "Quantization not supported"

        if "torch_dtype" in from_pretrained_kwargs:
            dtype = from_pretrained_kwargs["torch_dtype"]

        official_model_name = loading.get_official_model_name(model_name)

        cfg = loading.get_pretrained_model_config(
            official_model_name,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            fold_ln=False,
            device=device,
            n_devices=1,
            dtype=dtype,
            **from_pretrained_kwargs,
        )

        state_dict = loading.get_pretrained_state_dict(
            official_model_name, cfg, hf_model, dtype=dtype, **from_pretrained_kwargs
        )

        model = cls(cfg, tokenizer, move_to_device=False)

        model.load_state_dict(state_dict, strict=False)

        if move_to_device:
            model.to(cfg.device)

        print(f"Loaded pretrained model {model_name} into HookedTransformer")

        return model

    @property
    def W_E(self) -> Float[torch.Tensor, "d_vocab d_model"]:
        """
        Convenience to get the embedding matrix
        """
        return self.embed.embed.projection.weight

    @property
    def W_pos(self) -> Float[torch.Tensor, "n_ctx d_model"]:
        """
        Convenience function to get the positional embedding. Only works on models with absolute positional embeddings!
        """
        return self.embed.pos_embed.data

    @property
    def W_E_pos(self) -> Float[torch.Tensor, "d_vocab+n_ctx d_model"]:
        """
        Concatenated W_E and W_pos. Used as a full (overcomplete) basis of the input space, useful for full QK and full OV circuits.
        """
        return torch.cat([self.W_E, self.W_pos], dim=0)

    @property
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the key weights across all layers"""
        return torch.stack(
            [block.attn.W_K for block in self.blocks], dim=0
        )

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the query weights across all layers"""
        return torch.stack(
            [block.attn.W_Q for block in self.blocks], dim=0
        )

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the value weights across all layers"""
        return torch.stack(
            [block.attn.W_V for block in self.blocks], dim=0
        )

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stacks the attn output weights across all layers"""
        return torch.stack(
            [block.attn.W_O for block in self.blocks], dim=0
        )

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        """Stacks the MLP input weights across all layers"""
        return torch.stack(
            [block.mlp.W_in for block in self.blocks], dim=0
        )

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        """Stacks the MLP output weights across all layers"""
        return torch.stack(
            [block.mlp.W_out for block in self.blocks], dim=0
        )

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the key biases across all layers"""
        return torch.stack(
            [block.attn.b_K for block in self.blocks], dim=0
        )

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the query biases across all layers"""
        return torch.stack(
            [block.attn.b_Q for block in self.blocks], dim=0
        )

    @property
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the value biases across all layers"""
        return torch.stack(
            [block.attn.b_V for block in self.blocks], dim=0
        )

    @property
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the attn output biases across all layers"""
        return torch.stack(
            [block.attn.b_O for block in self.blocks], dim=0
        )

    @property
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        """Stacks the MLP input biases across all layers"""
        return torch.stack(
            [block.mlp.b_in for block in self.blocks], dim=0
        )

    @property
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the MLP output biases across all layers"""
        return torch.stack(
            [block.mlp.b_out for block in self.blocks], dim=0
        )

    @property
    def QK(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    def OV(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
        return FactoredMatrix(self.W_V, self.W_O)

    def all_head_labels(self) -> List[str]:
        return [
            f"L{l}H{h}"
            for l in range(self.cfg.n_layers)
            for h in range(self.cfg.n_heads)
        ]
