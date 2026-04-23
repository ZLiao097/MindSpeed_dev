# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import json
import torch.nn as nn


def patch_module_type_registry():
    """使用 AutoMapping.register_module_type 注册 MindSpeed 模块类型"""
    from megatron.bridge.models.conversion.param_mapping import AutoMapping
    
    AutoMapping.register_module_type("MindSpeedTEColumnParallelGroupedLinear", "column")
    AutoMapping.register_module_type("MindSpeedTELayerNormColumnParallelLinear", "column")
    AutoMapping.register_module_type("MindSpeedTERowParallelGroupedLinear", "row")
    AutoMapping.register_module_type("PTNorm", "replicated")


def _detect_parallelism_type_wrapper(original_func):
    """wrapper 包装 _detect_parallelism_type 方法，修改条件判断支持 MindSpeedTELayerNormColumnParallelLinear"""
    from megatron.bridge.models.conversion.utils import is_modelopt_dynamic_module
    
    def wrapper(self, module: nn.Module) -> str:
        if is_modelopt_dynamic_module(module):
            module_type = module.get_original_cls_by_level(level=0).__name__
        else:
            module_type = type(module).__name__
        
        if module_type == "TELayerNormColumnParallelLinear" or module_type == "MindSpeedTELayerNormColumnParallelLinear":
            if self.megatron_param and (
                self.megatron_param.endswith("layer_norm_weight") or self.megatron_param.endswith("layer_norm_bias")
            ):
                return "replicated"
            return "column"
        
        for parallelism, types in self._MODULE_TYPE_REGISTRY.items():
            if module_type in types:
                return parallelism
        
        if hasattr(module, "tensor_model_parallel"):
            if not module.tensor_model_parallel:
                return "replicated"
            partition_dim = getattr(module, "partition_dim", None)
            if partition_dim == 0:
                return "column"
            elif partition_dim == 1:
                return "row"
        
        if any(norm in module_type for norm in ["Norm", "Normalization"]):
            return "replicated"
        
        if module_type == "TELinear":
            if module.parallel_mode == "column":
                return "column"
            elif module.parallel_mode == "row":
                return "row"
            else:
                return "replicated"
        
        known_types = {p: sorted(list(t)) for p, t in self._MODULE_TYPE_REGISTRY.items()}
        raise ValueError(
            f"Cannot determine parallelism type for module '{module_type}' "
            f"at weight '{self.megatron_param}'.\n"
            f"Please use an explicit mapping type (e.g., ColumnParallelMapping) "
            f"or register the module type using:\n"
            f"  AutoMapping.register_module_type('{module_type}', 'column|row|replicated')\n\n"
            f"Currently known module types:\n{json.dumps(known_types, indent=2)}"
        )
    
    return wrapper