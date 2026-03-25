# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class GDNFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('experimental_attention_variant', optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title='experimental-attention-variant')
        group.add_argument("--experimental-attention-variant", type=str, default=None,
                           choices=['gated_delta_net', 'dsa'],
                           help="Experimental attention variant (e.g., dsa for DeepSeek Sparse Attention).")

    def register_patches(
            self,
            patch_manager: MindSpeedPatchesManager,
            args: Namespace
    ):
        if getattr(args, self.feature_name, None) == 'gated_delta_net':
            from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule
            patch_manager.register_patch('fla.ops.gated_delta_rule.chunk_gated_delta_rule', torch_chunk_gated_delta_rule)

            from mindspeed.core.ssm.gated_delta_net import GatedDeltaNet
            patch_manager.register_patch('megatron.core.ssm.gated_delta_net.GatedDeltaNet', GatedDeltaNet)