from functools import wraps
import math
import sys
import types
import torch


def _create_megatron_ssm_module_at_import_time():
    """Create dummy megatron.core.ssm.mamba_hybrid_layer_allocation module at import time.
    
    This must happen BEFORE any megatron.bridge imports to prevent static import failures.
    """
    target_module = 'megatron.core.ssm.mamba_hybrid_layer_allocation'
    
    if target_module in sys.modules:
        return
    
    class Symbols:
        """Symbols for different layer types."""
        MAMBA = "M"
        ATTENTION = "*"
        MLP = "-"
        MOE = 'E'
        VALID = {MAMBA, ATTENTION, MLP, MOE}
    
    def get_hybrid_total_layer_count(hybrid_layer_pattern):
        if hybrid_layer_pattern is None:
            return 0
        return len([c for c in hybrid_layer_pattern if c not in ('-', '*')])
    
    class ParsedHybridPattern:
        def __init__(self, pattern=None):
            self.main_pattern = pattern or []
            self.mtp_pattern = None
            self.mtp_num_depths = 0
    
    def parse_hybrid_pattern(hybrid_layer_pattern):
        if hybrid_layer_pattern is None:
            return ParsedHybridPattern()
        pattern = [c for c in hybrid_layer_pattern if c in Symbols.VALID]
        return ParsedHybridPattern(pattern)
    
    def allocate_layers(total_layers_count, target_attention_ratio=0.0, target_mlp_ratio=0.0, override_pattern=None):
        if override_pattern:
            layer_type_list = list(override_pattern)
        else:
            layer_type_list = [Symbols.MAMBA] * total_layers_count
        return layer_type_list
    
    def get_layer_maps_from_layer_type_list(layer_type_list):
        layer_types = [Symbols.ATTENTION, Symbols.MAMBA, Symbols.MLP, Symbols.MOE]
        layer_maps = {layer_type: {} for layer_type in layer_types}
        for global_layer_idx, layer_type in enumerate(layer_type_list):
            if layer_type in layer_maps:
                layer_map = layer_maps[layer_type]
                local_layer_idx = len(layer_map)
                layer_map[global_layer_idx] = local_layer_idx
        return [layer_maps[layer_type] for layer_type in layer_types]
    
    dummy_module = types.ModuleType(target_module)
    dummy_module.Symbols = Symbols
    dummy_module.get_hybrid_total_layer_count = get_hybrid_total_layer_count
    dummy_module.parse_hybrid_pattern = parse_hybrid_pattern
    dummy_module.allocate_layers = allocate_layers
    dummy_module.get_layer_maps_from_layer_type_list = get_layer_maps_from_layer_type_list
    dummy_module.ParsedHybridPattern = ParsedHybridPattern
    dummy_module.__name__ = target_module
    dummy_module.__file__ = 'mindspeed_dummy_module.py'
    sys.modules[target_module] = dummy_module


_create_megatron_ssm_module_at_import_time()


def create_dummy_modelopt_modules():
    """Create dummy modelopt modules to prevent import errors in NPU environment.
    
    In NPU environment, importing modelopt.torch.quantization triggers triton module loading,
    which calls torch.cuda.get_device_capability() that returns None, causing TypeError.
    This function pre-registers dummy modules in sys.modules to bypass the real imports.
    """
    dummy_modules = [
        'modelopt',
        'modelopt.torch',
        'modelopt.torch.quantization',
        'modelopt.torch.quantization.utils',
        'modelopt.torch.quantization.nn',
        'modelopt.torch.quantization.nn.modules',
        'modelopt.torch.quantization.nn.modules.quant_module',
        'modelopt.torch.quantization.tensor_quant',
        'modelopt.torch.quantization.triton',
    ]
    
    for module_name in dummy_modules:
        if module_name not in sys.modules:
            sys.modules[module_name] = types.ModuleType(module_name)
    
    def is_quantized(model):
        return False
    
    def QuantInputBase(*args, **kwargs):
        pass
    
    def QuantModuleRegistry(*args, **kwargs):
        pass
    
    def QUANT_DESC_8BIT_PER_TENSOR(*args, **kwargs):
        pass
    
    sys.modules['modelopt.torch.quantization.utils'].is_quantized = is_quantized
    sys.modules['modelopt.torch.quantization.nn.modules.quant_module'].QuantInputBase = QuantInputBase
    sys.modules['modelopt.torch.quantization.nn.modules.quant_module'].QuantModuleRegistry = QuantModuleRegistry
    sys.modules['modelopt.torch.quantization.tensor_quant'].QUANT_DESC_8BIT_PER_TENSOR = QUANT_DESC_8BIT_PER_TENSOR


def version_wrapper(fn):
    @wraps(fn)
    def wrapper(name, *args, **kwargs):
        return '2.2.0' if name == 'transformer-engine' else fn(name, *args, **kwargs)

    return wrapper


def multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
    return op(noop_flag_buffer, tensor_lists, *args)


def multi_tensor_l2norm(overflow_buf, tensor_lists, per_parameter):
    total_norm = 0.0
    norm_type = 2.0
    ret_per_tensor = [] if per_parameter else None
    for grads_for_norm in tensor_lists:
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += grad_norm ** norm_type
        if per_parameter:
            ret_per_tensor.append(total_norm.clone())
    if not tensor_lists:
        grad_norm = torch.cuda.FloatTensor([0])
        total_norm = grad_norm ** norm_type
    return total_norm ** (1 / norm_type), ret_per_tensor


def multi_tensor_scale(overflow_buf, tensor_lists, scale):
    if len(tensor_lists) != 2:
        raise AssertionError('The size of tensor list must be 2, but got {}'.format(len(tensor_lists)))
    if len(tensor_lists[0]) != len(tensor_lists[1]):
        raise AssertionError('The size of tensor list must be same, but got {} and {}'
                             .format(len(tensor_lists[0]), len(tensor_lists[1])))
    with torch.no_grad():
        for i in range(len(tensor_lists[0])):
            tensor_lists[1][i].copy_(tensor_lists[0][i] * scale)


def type_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if isinstance(res, str):
            res = res.replace('npu', 'cuda')
        return res

    return wrapper


def ensure_contiguous_wrapper(fn):
    @wraps(fn)
    def wrapper(tensor, *args, **kwargs):
        tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
        return fn(tensor, *args, **kwargs)

    return wrapper


def lcm(a, b):
    return (a * b) // math.gcd(a, b)


def dummy_function(*args, **kwargs):
    pass


def torch_all_reduce_double_dtype_bypass_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if torch.is_tensor(args[0]) and args[0].dtype == torch.double:
            args = list(args)
            args[0] = args[0].float()
            handle = fn(*args, **kwargs)
            if handle is not None:
                handle.wait()
            args[0] = args[0].double()
            return None

        return fn(*args, **kwargs)

    return wrapper


def dummy_compile(*args, **kwargs):
    if len(args) > 0 and callable(args[0]):
        def wrapper(*fn_args, **fn_kwargs):
            return args[0](*fn_args, **fn_kwargs)
        return wrapper
    else:
        def compile_wrapper(fn):
            def wrapper(*fn_args, **fn_kwargs):
                return fn(*fn_args, **fn_kwargs)
            return wrapper
        return compile_wrapper
