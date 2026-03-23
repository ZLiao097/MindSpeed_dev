# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class GDNFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__(
            'gated_delta_net',
            optimization_level=2
        )

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--gated-delta-net', action='store_true', default=False,
                           help='Enable gated delta net feature.')

    def register_patches(
            self,
            patch_manager: MindSpeedPatchesManager,
            args: Namespace
    ):
        if getattr(args, self.feature_name, None):
            from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule
            patch_manager.register_patch('fla.ops.gated_delta_rule.chunk_gated_delta_rule', torch_chunk_gated_delta_rule)

            from mindspeed.core.ssm.gated_delta_net import GatedDeltaNet
            patch_manager.register_patch('megatron.core.ssm.gated_delta_net.GatedDeltaNet', GatedDeltaNet)