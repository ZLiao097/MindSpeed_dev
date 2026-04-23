# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import json
import torch.nn as nn
from functools import wraps

_MODULE_TYPE_REGISTRY_PATCHED = False


def _ensure_module_type_registry_patched():
    """延迟注册 MindSpeed 模块类型，仅在第一次调用 wrapper 时执行"""
    global _MODULE_TYPE_REGISTRY_PATCHED
    if _MODULE_TYPE_REGISTRY_PATCHED:
        return
    try:
        from megatron.bridge.models.conversion.param_mapping import AutoMapping
        
        AutoMapping.register_module_type("MindSpeedTEColumnParallelGroupedLinear", "column")
        AutoMapping.register_module_type("MindSpeedTELayerNormColumnParallelLinear", "column")
        AutoMapping.register_module_type("MindSpeedTERowParallelGroupedLinear", "row")
        AutoMapping.register_module_type("PTNorm", "replicated")
        _MODULE_TYPE_REGISTRY_PATCHED = True
    except ImportError:
        pass


def _detect_parallelism_type_wrapper(original_func):
    """wrapper 包装 _detect_parallelism_type 方法，修改条件判断支持 MindSpeedTELayerNormColumnParallelLinear"""
    def wrapper(self, module: nn.Module) -> str:
        _ensure_module_type_registry_patched()
        
        try:
            from megatron.bridge.models.conversion.utils import is_modelopt_dynamic_module
            
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
        except ImportError:
            return original_func(self, module)
    
    return wrapper


def transformer_config_finalize_wrapper(fn):
    """wrapper 为 Megatron-Bridge TransformerConfig.finalize() 注入完整参数"""
    @wraps(fn)
    def wrapper(self):
        from mindspeed.args_utils import get_full_args
        
        args = get_full_args()
        if args is not None:
            for key, value in vars(args).items():
                if not hasattr(self, key):
                    setattr(self, key, value)
        return fn(self)
    
    return wrapper