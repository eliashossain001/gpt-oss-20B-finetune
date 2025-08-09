
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.verbose = False
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.accumulated_cache_size_limit = 128
torch._dynamo.config.traceable_tensor_subclasses = set()
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.allowed_functions_module_string_ignorelist = {'torch.testing', 'torch._decomp', 'torch.distributions', 'torch._prims', 'torch._refs'}
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.optimize_ddp = True
torch._dynamo.config.do_not_emit_runtime_asserts = True
torch._dynamo.config._ddp_optimization_mode = ['ddp_optimizer', 'python_reducer', 'python_reducer_without_compiled_forward', 'no_optimization']
torch._dynamo.config.numpy_default_float = 'float32'
torch._dynamo.config.inline_inbuilt_nn_modules = True
torch._dynamo.config._save_config_ignore = {'constant_functions', 'repro_after', 'skipfiles_inline_module_allowlist', 'repro_level'}
torch._dynamo.config.reorderable_logging_functions = set()
torch._dynamo.config.ignore_logger_methods = set()
torch._dynamo.config._autograd_backward_strict_mode_banned_ops = ['stride', 'requires_grad', 'storage_offset', 'layout', 'data', 'is_coalesced', 'is_complex', 'is_conj', 'is_contiguous', 'is_cpu', 'is_cuda', 'is_distributed', 'is_floating_point', 'is_inference', 'is_ipu', 'is_leaf', 'is_maia', 'is_meta', 'is_mkldnn', 'is_mps', 'is_mtia', 'is_neg', 'is_nested', 'is_nonzero', 'is_pinned', 'is_quantized', 'is_same_size', 'is_set_to', 'is_shared', 'is_signed', 'is_sparse', 'is_sparse_csr', 'is_vulkan', 'is_xla', 'is_xpu']
torch._dynamo.config.compiled_autograd = False
torch._dynamo.config.compiled_autograd_kwargs_override = {}
torch._inductor.config.debug = False
torch._inductor.config.disable_progress = True
torch._inductor.config.verbose_progress = False
torch._inductor.config.dce = True
torch._inductor.config.memory_planning = True
torch._inductor.config.memory_pool = 'none'
torch._inductor.config.epilogue_fusion = True
torch._inductor.config.efficient_conv_bn_eval_fx_passes = True
torch._inductor.config.group_fusion = False
torch._inductor.config.pre_grad_fusion_options = {}
torch._inductor.config.post_grad_fusion_options = {}
torch._inductor.config.dynamic_scale_rblock = True
torch._inductor.config.fx_passes_numeric_check = {'pre_grad': False, 'precision': 0.0001, 'num_iterations': 1, 'requires_optimizer': True}
torch._inductor.config.reorder_for_compute_comm_overlap_passes = ['reorder_compute_for_overlap', 'sink_waits', 'raise_comms']
torch._inductor.config.max_autotune = False
torch._inductor.config.max_autotune_pointwise = False
torch._inductor.config.max_autotune_gemm = False
torch._inductor.config.max_autotune_gemm_backends = 'ATEN,TRITON,CPP'
torch._inductor.config.autotune_fallback_to_aten = True
torch._inductor.config.autotune_multi_device = True
torch._inductor.config.coordinate_descent_tuning = False
torch._inductor.config.aggressive_fusion = False
torch._inductor.config.combo_kernels = False
torch._inductor.config.benchmark_combo_kernel = False
torch._inductor.config.combo_kernel_foreach_dynamic_shapes = False
torch._inductor.config.emulate_precision_casts = False
torch._inductor.config._fuse_ddp_communication_passes = ['fuse_ddp_with_concat_op', 'schedule_comm_wait']
torch._inductor.config.compile_threads = 1
torch._inductor.config.shape_padding = True
torch._inductor.config.freezing = False
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.cooperative_reductions = False
torch._inductor.config.triton.multi_kernel = 0
torch._inductor.config.triton.use_block_ptr = False
torch._inductor.config.triton.enable_persistent_tma_matmul = False
torch._inductor.config.aot_inductor.metadata = {}
torch._inductor.config.aot_inductor.presets = {}
torch._inductor.config.cuda.compile_opt_level = '-O1'
torch._inductor.config.cuda.enable_cuda_lto = True
torch._inductor.config.cuda.use_fast_math = True
torch._inductor.config.rocm.arch = []
torch._inductor.config.rocm.ck_supported_arch = ['gfx90a', 'gfx940', 'gfx941', 'gfx942']
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.trace.graph_diagram = False
torch._inductor.config._save_config_ignore = ['trace.upload_tar', 'joint_custom_pre_pass', 'joint_custom_post_pass', 'pre_grad_custom_pass']
torch._inductor.config._cache_config_ignore_prefix = ['trace', 'cuda.cutlass_dir', 'worker_start_method', 'compile_threads', 'post_grad_custom_post_pass', 'post_grad_custom_pre_pass', 'always_complex_memory_overlap_TESTING_ONLY']
torch._inductor.config.external_matmul = []
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.6.0+cu124
# torch cuda version: 12.4
# torch git version: 2236df1770800ffea5697b11b0bb0d910b2e59e1


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA A100 80GB PCIe : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_4, primals_2, sym_size_int_5, mm, amax, log, mm_1, amax_1, log_1, mm_2, amax_2, log_2, mm_3, amax_3, log_3, mm_4, amax_4, log_4, mm_5, amax_5, log_5, mm_6, amax_6, log_6, convert_element_type_28, ne_43, where_14, permute_8, ne_45, where_16, ne_47, where_18, ne_49, where_20, ne_51, where_22, ne_53, where_24, ne_55, where_26, tangents_1):
        div_1 = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_28);  tangents_1 = convert_element_type_28 = None
        full = torch.ops.aten.full.default([sym_size_int_5, 201088], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  sym_size_int_5 = None
        scatter = torch.ops.aten.scatter.value(full, 1, where_14, -1.0);  full = where_14 = None
        full_default_2 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15 = torch.ops.aten.where.self(ne_43, div_1, full_default_2);  ne_43 = None
        mul_111 = torch.ops.aten.mul.Tensor(scatter, where_15);  scatter = where_15 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        sub_62 = torch.ops.aten.sub.Tensor(convert_element_type_26, amax_6);  convert_element_type_26 = amax_6 = None
        sub_63 = torch.ops.aten.sub.Tensor(sub_62, log_6);  sub_62 = log_6 = None
        exp_7 = torch.ops.aten.exp.default(sub_63);  sub_63 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(mul_111, [1], True)
        mul_112 = torch.ops.aten.mul.Tensor(exp_7, sum_22);  exp_7 = sum_22 = None
        sub_66 = torch.ops.aten.sub.Tensor(mul_111, mul_112);  mul_111 = mul_112 = None
        convert_element_type_29 = torch.ops.prims.convert_element_type.default(sub_66, torch.bfloat16);  sub_66 = None
        mm_7 = torch.ops.aten.mm.default(convert_element_type_29, permute_8);  convert_element_type_29 = None
        add_19 = primals_4 + 7
        sub_8 = add_19 - 1;  add_19 = None
        floordiv = sub_8 // 7;  sub_8 = None
        full_default_18 = torch.ops.aten.full.default([floordiv, 201088], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  floordiv = None
        scatter_1 = torch.ops.aten.scatter.value(full_default_18, 1, where_16, -1.0);  where_16 = None
        where_17 = torch.ops.aten.where.self(ne_45, div_1, full_default_2);  ne_45 = None
        mul_113 = torch.ops.aten.mul.Tensor(scatter_1, where_17);  scatter_1 = where_17 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        sub_56 = torch.ops.aten.sub.Tensor(convert_element_type_22, amax_5);  convert_element_type_22 = amax_5 = None
        sub_57 = torch.ops.aten.sub.Tensor(sub_56, log_5);  sub_56 = log_5 = None
        exp_8 = torch.ops.aten.exp.default(sub_57);  sub_57 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(mul_113, [1], True)
        mul_114 = torch.ops.aten.mul.Tensor(exp_8, sum_23);  exp_8 = sum_23 = None
        sub_67 = torch.ops.aten.sub.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        convert_element_type_32 = torch.ops.prims.convert_element_type.default(sub_67, torch.bfloat16);  sub_67 = None
        mm_8 = torch.ops.aten.mm.default(convert_element_type_32, permute_8);  convert_element_type_32 = None
        scatter_2 = torch.ops.aten.scatter.value(full_default_18, 1, where_18, -1.0);  where_18 = None
        where_19 = torch.ops.aten.where.self(ne_47, div_1, full_default_2);  ne_47 = None
        mul_115 = torch.ops.aten.mul.Tensor(scatter_2, where_19);  scatter_2 = where_19 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        sub_50 = torch.ops.aten.sub.Tensor(convert_element_type_18, amax_4);  convert_element_type_18 = amax_4 = None
        sub_51 = torch.ops.aten.sub.Tensor(sub_50, log_4);  sub_50 = log_4 = None
        exp_9 = torch.ops.aten.exp.default(sub_51);  sub_51 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(mul_115, [1], True)
        mul_116 = torch.ops.aten.mul.Tensor(exp_9, sum_24);  exp_9 = sum_24 = None
        sub_68 = torch.ops.aten.sub.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(sub_68, torch.bfloat16);  sub_68 = None
        mm_9 = torch.ops.aten.mm.default(convert_element_type_35, permute_8);  convert_element_type_35 = None
        scatter_3 = torch.ops.aten.scatter.value(full_default_18, 1, where_20, -1.0);  where_20 = None
        where_21 = torch.ops.aten.where.self(ne_49, div_1, full_default_2);  ne_49 = None
        mul_117 = torch.ops.aten.mul.Tensor(scatter_3, where_21);  scatter_3 = where_21 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        sub_44 = torch.ops.aten.sub.Tensor(convert_element_type_14, amax_3);  convert_element_type_14 = amax_3 = None
        sub_45 = torch.ops.aten.sub.Tensor(sub_44, log_3);  sub_44 = log_3 = None
        exp_10 = torch.ops.aten.exp.default(sub_45);  sub_45 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_117, [1], True)
        mul_118 = torch.ops.aten.mul.Tensor(exp_10, sum_25);  exp_10 = sum_25 = None
        sub_69 = torch.ops.aten.sub.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
        convert_element_type_38 = torch.ops.prims.convert_element_type.default(sub_69, torch.bfloat16);  sub_69 = None
        mm_10 = torch.ops.aten.mm.default(convert_element_type_38, permute_8);  convert_element_type_38 = None
        scatter_4 = torch.ops.aten.scatter.value(full_default_18, 1, where_22, -1.0);  where_22 = None
        where_23 = torch.ops.aten.where.self(ne_51, div_1, full_default_2);  ne_51 = None
        mul_119 = torch.ops.aten.mul.Tensor(scatter_4, where_23);  scatter_4 = where_23 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        sub_38 = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_2);  convert_element_type_10 = amax_2 = None
        sub_39 = torch.ops.aten.sub.Tensor(sub_38, log_2);  sub_38 = log_2 = None
        exp_11 = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(mul_119, [1], True)
        mul_120 = torch.ops.aten.mul.Tensor(exp_11, sum_26);  exp_11 = sum_26 = None
        sub_70 = torch.ops.aten.sub.Tensor(mul_119, mul_120);  mul_119 = mul_120 = None
        convert_element_type_41 = torch.ops.prims.convert_element_type.default(sub_70, torch.bfloat16);  sub_70 = None
        mm_11 = torch.ops.aten.mm.default(convert_element_type_41, permute_8);  convert_element_type_41 = None
        scatter_5 = torch.ops.aten.scatter.value(full_default_18, 1, where_24, -1.0);  where_24 = None
        where_25 = torch.ops.aten.where.self(ne_53, div_1, full_default_2);  ne_53 = None
        mul_121 = torch.ops.aten.mul.Tensor(scatter_5, where_25);  scatter_5 = where_25 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        sub_32 = torch.ops.aten.sub.Tensor(convert_element_type_6, amax_1);  convert_element_type_6 = amax_1 = None
        sub_33 = torch.ops.aten.sub.Tensor(sub_32, log_1);  sub_32 = log_1 = None
        exp_12 = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(mul_121, [1], True)
        mul_122 = torch.ops.aten.mul.Tensor(exp_12, sum_27);  exp_12 = sum_27 = None
        sub_71 = torch.ops.aten.sub.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
        convert_element_type_44 = torch.ops.prims.convert_element_type.default(sub_71, torch.bfloat16);  sub_71 = None
        mm_12 = torch.ops.aten.mm.default(convert_element_type_44, permute_8);  convert_element_type_44 = None
        scatter_6 = torch.ops.aten.scatter.value(full_default_18, 1, where_26, -1.0);  full_default_18 = where_26 = None
        where_27 = torch.ops.aten.where.self(ne_55, div_1, full_default_2);  ne_55 = div_1 = full_default_2 = None
        mul_123 = torch.ops.aten.mul.Tensor(scatter_6, where_27);  scatter_6 = where_27 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        sub_26 = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        sub_27 = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
        exp_13 = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(mul_123, [1], True)
        mul_124 = torch.ops.aten.mul.Tensor(exp_13, sum_28);  exp_13 = sum_28 = None
        sub_72 = torch.ops.aten.sub.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        convert_element_type_47 = torch.ops.prims.convert_element_type.default(sub_72, torch.bfloat16);  sub_72 = None
        mm_13 = torch.ops.aten.mm.default(convert_element_type_47, permute_8);  convert_element_type_47 = permute_8 = None
        cat = torch.ops.aten.cat.default([mm_13, mm_12, mm_11, mm_10, mm_9, mm_8, mm_7]);  mm_13 = mm_12 = mm_11 = mm_10 = mm_9 = mm_8 = mm_7 = None
        view_9 = torch.ops.aten.view.default(cat, [1, primals_4, 2880]);  cat = primals_4 = None
        return (None, None, None, None, view_9, None)
        
def load_args(reader):
    reader.symint(1024)  # primals_4
    reader.symint(1024)  # primals_2
    reader.symint(142)  # sym_size_int_5
    buf0 = reader.storage(None, 402176*(((s2 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (((s2 + 6)//7), 201088), dtype=torch.bfloat16, is_leaf=True)  # mm
    buf1 = reader.storage(None, 4*(((s2 + 6)//7)), device=device(type='cuda', index=0))
    reader.tensor(buf1, (((s2 + 6)//7), 1), is_leaf=True)  # amax
    buf2 = reader.storage(None, 4*(((s2 + 6)//7)), device=device(type='cuda', index=0))
    reader.tensor(buf2, (((s2 + 6)//7), 1), is_leaf=True)  # log
    buf3 = reader.storage(None, 402176*(((s2 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (((s2 + 6)//7), 201088), dtype=torch.bfloat16, is_leaf=True)  # mm_1
    buf4 = reader.storage(None, 4*(((s2 + 6)//7)), device=device(type='cuda', index=0))
    reader.tensor(buf4, (((s2 + 6)//7), 1), is_leaf=True)  # amax_1
    buf5 = reader.storage(None, 4*(((s2 + 6)//7)), device=device(type='cuda', index=0))
    reader.tensor(buf5, (((s2 + 6)//7), 1), is_leaf=True)  # log_1
    buf6 = reader.storage(None, 402176*(((s2 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (((s2 + 6)//7), 201088), dtype=torch.bfloat16, is_leaf=True)  # mm_2
    buf7 = reader.storage(None, 4*(((s2 + 6)//7)), device=device(type='cuda', index=0))
    reader.tensor(buf7, (((s2 + 6)//7), 1), is_leaf=True)  # amax_2
    buf8 = reader.storage(None, 4*(((s2 + 6)//7)), device=device(type='cuda', index=0))
    reader.tensor(buf8, (((s2 + 6)//7), 1), is_leaf=True)  # log_2
    buf9 = reader.storage(None, 402176*(((s2 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (((s2 + 6)//7), 201088), dtype=torch.bfloat16, is_leaf=True)  # mm_3
    buf10 = reader.storage(None, 4*(((s2 + 6)//7)), device=device(type='cuda', index=0))
    reader.tensor(buf10, (((s2 + 6)//7), 1), is_leaf=True)  # amax_3
    buf11 = reader.storage(None, 4*(((s2 + 6)//7)), device=device(type='cuda', index=0))
    reader.tensor(buf11, (((s2 + 6)//7), 1), is_leaf=True)  # log_3
    buf12 = reader.storage(None, 402176*(((s2 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf12, (((s2 + 6)//7), 201088), dtype=torch.bfloat16, is_leaf=True)  # mm_4
    buf13 = reader.storage(None, 4*(((s2 + 6)//7)), device=device(type='cuda', index=0))
    reader.tensor(buf13, (((s2 + 6)//7), 1), is_leaf=True)  # amax_4
    buf14 = reader.storage(None, 4*(((s2 + 6)//7)), device=device(type='cuda', index=0))
    reader.tensor(buf14, (((s2 + 6)//7), 1), is_leaf=True)  # log_4
    buf15 = reader.storage(None, 402176*(((s2 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf15, (((s2 + 6)//7), 201088), dtype=torch.bfloat16, is_leaf=True)  # mm_5
    buf16 = reader.storage(None, 4*(((s2 + 6)//7)), device=device(type='cuda', index=0))
    reader.tensor(buf16, (((s2 + 6)//7), 1), is_leaf=True)  # amax_5
    buf17 = reader.storage(None, 4*(((s2 + 6)//7)), device=device(type='cuda', index=0))
    reader.tensor(buf17, (((s2 + 6)//7), 1), is_leaf=True)  # log_5
    buf18 = reader.storage(None, 402176*s2 - 2413056*(((s2 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf18, (s2 - 6*(((s2 + 6)//7)), 201088), dtype=torch.bfloat16, is_leaf=True)  # mm_6
    buf19 = reader.storage(None, 4*s2 - 24*(((s2 + 6)//7)), device=device(type='cuda', index=0))
    reader.tensor(buf19, (s2 - 6*(((s2 + 6)//7)), 1), is_leaf=True)  # amax_6
    buf20 = reader.storage(None, 4*s2 - 24*(((s2 + 6)//7)), device=device(type='cuda', index=0))
    reader.tensor(buf20, (s2 - 6*(((s2 + 6)//7)), 1), is_leaf=True)  # log_6
    buf21 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf21, (), is_leaf=True)  # convert_element_type_28
    buf22 = reader.storage(None, s0 - 6*(((s0 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf22, (s0 - 6*(((s0 + 6)//7)), 1), dtype=torch.bool, is_leaf=True)  # ne_43
    buf23 = reader.storage(None, 8*s0 - 48*(((s0 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf23, (s0 - 6*(((s0 + 6)//7)), 1), dtype=torch.int64, is_leaf=True)  # where_14
    buf24 = reader.storage(None, 1158266880, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf24, (201088, 2880), dtype=torch.bfloat16, is_leaf=True)  # permute_8
    buf25 = reader.storage(None, ((s0 + 6)//7), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf25, (((s0 + 6)//7), 1), dtype=torch.bool, is_leaf=True)  # ne_45
    buf26 = reader.storage(None, 8*(((s0 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf26, (((s0 + 6)//7), 1), dtype=torch.int64, is_leaf=True)  # where_16
    buf27 = reader.storage(None, ((s0 + 6)//7), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf27, (((s0 + 6)//7), 1), dtype=torch.bool, is_leaf=True)  # ne_47
    buf28 = reader.storage(None, 8*(((s0 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf28, (((s0 + 6)//7), 1), dtype=torch.int64, is_leaf=True)  # where_18
    buf29 = reader.storage(None, ((s0 + 6)//7), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf29, (((s0 + 6)//7), 1), dtype=torch.bool, is_leaf=True)  # ne_49
    buf30 = reader.storage(None, 8*(((s0 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf30, (((s0 + 6)//7), 1), dtype=torch.int64, is_leaf=True)  # where_20
    buf31 = reader.storage(None, ((s0 + 6)//7), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf31, (((s0 + 6)//7), 1), dtype=torch.bool, is_leaf=True)  # ne_51
    buf32 = reader.storage(None, 8*(((s0 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf32, (((s0 + 6)//7), 1), dtype=torch.int64, is_leaf=True)  # where_22
    buf33 = reader.storage(None, ((s0 + 6)//7), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf33, (((s0 + 6)//7), 1), dtype=torch.bool, is_leaf=True)  # ne_53
    buf34 = reader.storage(None, 8*(((s0 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf34, (((s0 + 6)//7), 1), dtype=torch.int64, is_leaf=True)  # where_24
    buf35 = reader.storage(None, ((s0 + 6)//7), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf35, (((s0 + 6)//7), 1), dtype=torch.bool, is_leaf=True)  # ne_55
    buf36 = reader.storage(None, 8*(((s0 + 6)//7)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf36, (((s0 + 6)//7), 1), dtype=torch.int64, is_leaf=True)  # where_26
    buf37 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf37, (), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)