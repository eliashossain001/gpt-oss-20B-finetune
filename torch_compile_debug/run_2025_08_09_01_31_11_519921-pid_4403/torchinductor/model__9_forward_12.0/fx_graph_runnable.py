
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6):
        empty = torch.ops.aten.empty.memory_format([1, primals_2], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute = torch.ops.aten.permute.default(empty, [0, 1]);  empty = None
        slice_1 = torch.ops.aten.slice.Tensor(primals_3, 1, 1, 9223372036854775807);  primals_3 = None
        slice_scatter = torch.ops.aten.slice_scatter.default(permute, slice_1, 1, 0, -1);  permute = slice_1 = None
        full_default = torch.ops.aten.full.default([], -100, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select_1 = torch.ops.aten.select.int(slice_scatter, 1, -1)
        copy_1 = torch.ops.aten.copy.default(select_1, full_default);  select_1 = full_default = None
        select_scatter = torch.ops.aten.select_scatter.default(slice_scatter, copy_1, 1, -1);  slice_scatter = copy_1 = None
        view_1 = torch.ops.aten.view.default(primals_5, [-1, 2880]);  primals_5 = None
        add_19 = primals_4 + 7
        sub_8 = add_19 - 1;  add_19 = None
        floordiv = sub_8 // 7;  sub_8 = None
        split = torch.ops.aten.split.Tensor(view_1, floordiv);  view_1 = floordiv = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        getitem_3 = split[3]
        getitem_4 = split[4]
        getitem_5 = split[5]
        getitem_6 = split[6];  split = None
        sym_size_int_5 = torch.ops.aten.sym_size.int(getitem_6, 0)
        add_41 = primals_2 + 7
        sub_16 = add_41 - 1;  add_41 = None
        floordiv_1 = sub_16 // 7;  sub_16 = None
        permute_1 = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mm = torch.ops.aten.mm.default(getitem, permute_1);  getitem = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(mm, torch.float32)
        amax = torch.ops.aten.amax.default(convert_element_type_2, [1], True)
        sub_26 = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = None
        exp = torch.ops.aten.exp.default(sub_26)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_27 = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = None
        view_2 = torch.ops.aten.view.default(select_scatter, [-1]);  select_scatter = None
        split_2 = torch.ops.aten.split.Tensor(view_2, floordiv_1);  view_2 = floordiv_1 = None
        getitem_14 = split_2[0]
        ne_4 = torch.ops.aten.ne.Scalar(getitem_14, -100)
        full_default_1 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(ne_4, getitem_14, full_default_1)
        unsqueeze = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather = torch.ops.aten.gather.default(sub_27, 1, unsqueeze);  sub_27 = unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_2 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne_4, neg, full_default_2);  ne_4 = neg = None
        sum_3 = torch.ops.aten.sum.default(where_1);  where_1 = None
        add_68 = torch.ops.aten.add.Tensor(sum_3, 0.0);  sum_3 = None
        mm_1 = torch.ops.aten.mm.default(getitem_1, permute_1);  getitem_1 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(mm_1, torch.float32)
        amax_1 = torch.ops.aten.amax.default(convert_element_type_6, [1], True)
        sub_32 = torch.ops.aten.sub.Tensor(convert_element_type_6, amax_1);  convert_element_type_6 = None
        exp_1 = torch.ops.aten.exp.default(sub_32)
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_1, [1], True);  exp_1 = None
        log_1 = torch.ops.aten.log.default(sum_4);  sum_4 = None
        sub_33 = torch.ops.aten.sub.Tensor(sub_32, log_1);  sub_32 = None
        getitem_22 = split_2[1]
        ne_10 = torch.ops.aten.ne.Scalar(getitem_22, -100)
        where_2 = torch.ops.aten.where.self(ne_10, getitem_22, full_default_1)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
        gather_1 = torch.ops.aten.gather.default(sub_33, 1, unsqueeze_1);  sub_33 = unsqueeze_1 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1 = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
        where_3 = torch.ops.aten.where.self(ne_10, neg_1, full_default_2);  ne_10 = neg_1 = None
        sum_6 = torch.ops.aten.sum.default(where_3);  where_3 = None
        add_81 = torch.ops.aten.add.Tensor(add_68, sum_6);  add_68 = sum_6 = None
        mm_2 = torch.ops.aten.mm.default(getitem_2, permute_1);  getitem_2 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(mm_2, torch.float32)
        amax_2 = torch.ops.aten.amax.default(convert_element_type_10, [1], True)
        sub_38 = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_2);  convert_element_type_10 = None
        exp_2 = torch.ops.aten.exp.default(sub_38)
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_2, [1], True);  exp_2 = None
        log_2 = torch.ops.aten.log.default(sum_7);  sum_7 = None
        sub_39 = torch.ops.aten.sub.Tensor(sub_38, log_2);  sub_38 = None
        getitem_30 = split_2[2]
        ne_16 = torch.ops.aten.ne.Scalar(getitem_30, -100)
        where_4 = torch.ops.aten.where.self(ne_16, getitem_30, full_default_1)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where_4, 1);  where_4 = None
        gather_2 = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_2);  sub_39 = unsqueeze_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(gather_2, 1);  gather_2 = None
        neg_2 = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        where_5 = torch.ops.aten.where.self(ne_16, neg_2, full_default_2);  ne_16 = neg_2 = None
        sum_9 = torch.ops.aten.sum.default(where_5);  where_5 = None
        add_94 = torch.ops.aten.add.Tensor(add_81, sum_9);  add_81 = sum_9 = None
        mm_3 = torch.ops.aten.mm.default(getitem_3, permute_1);  getitem_3 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(mm_3, torch.float32)
        amax_3 = torch.ops.aten.amax.default(convert_element_type_14, [1], True)
        sub_44 = torch.ops.aten.sub.Tensor(convert_element_type_14, amax_3);  convert_element_type_14 = None
        exp_3 = torch.ops.aten.exp.default(sub_44)
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_3, [1], True);  exp_3 = None
        log_3 = torch.ops.aten.log.default(sum_10);  sum_10 = None
        sub_45 = torch.ops.aten.sub.Tensor(sub_44, log_3);  sub_44 = None
        getitem_38 = split_2[3]
        ne_22 = torch.ops.aten.ne.Scalar(getitem_38, -100)
        where_6 = torch.ops.aten.where.self(ne_22, getitem_38, full_default_1)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather_3 = torch.ops.aten.gather.default(sub_45, 1, unsqueeze_3);  sub_45 = unsqueeze_3 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(gather_3, 1);  gather_3 = None
        neg_3 = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        where_7 = torch.ops.aten.where.self(ne_22, neg_3, full_default_2);  ne_22 = neg_3 = None
        sum_12 = torch.ops.aten.sum.default(where_7);  where_7 = None
        add_107 = torch.ops.aten.add.Tensor(add_94, sum_12);  add_94 = sum_12 = None
        mm_4 = torch.ops.aten.mm.default(getitem_4, permute_1);  getitem_4 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(mm_4, torch.float32)
        amax_4 = torch.ops.aten.amax.default(convert_element_type_18, [1], True)
        sub_50 = torch.ops.aten.sub.Tensor(convert_element_type_18, amax_4);  convert_element_type_18 = None
        exp_4 = torch.ops.aten.exp.default(sub_50)
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_4, [1], True);  exp_4 = None
        log_4 = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_51 = torch.ops.aten.sub.Tensor(sub_50, log_4);  sub_50 = None
        getitem_46 = split_2[4]
        ne_28 = torch.ops.aten.ne.Scalar(getitem_46, -100)
        where_8 = torch.ops.aten.where.self(ne_28, getitem_46, full_default_1)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(where_8, 1);  where_8 = None
        gather_4 = torch.ops.aten.gather.default(sub_51, 1, unsqueeze_4);  sub_51 = unsqueeze_4 = None
        squeeze_4 = torch.ops.aten.squeeze.dim(gather_4, 1);  gather_4 = None
        neg_4 = torch.ops.aten.neg.default(squeeze_4);  squeeze_4 = None
        where_9 = torch.ops.aten.where.self(ne_28, neg_4, full_default_2);  ne_28 = neg_4 = None
        sum_15 = torch.ops.aten.sum.default(where_9);  where_9 = None
        add_120 = torch.ops.aten.add.Tensor(add_107, sum_15);  add_107 = sum_15 = None
        mm_5 = torch.ops.aten.mm.default(getitem_5, permute_1);  getitem_5 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(mm_5, torch.float32)
        amax_5 = torch.ops.aten.amax.default(convert_element_type_22, [1], True)
        sub_56 = torch.ops.aten.sub.Tensor(convert_element_type_22, amax_5);  convert_element_type_22 = None
        exp_5 = torch.ops.aten.exp.default(sub_56)
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_5, [1], True);  exp_5 = None
        log_5 = torch.ops.aten.log.default(sum_16);  sum_16 = None
        sub_57 = torch.ops.aten.sub.Tensor(sub_56, log_5);  sub_56 = None
        getitem_54 = split_2[5]
        ne_34 = torch.ops.aten.ne.Scalar(getitem_54, -100)
        where_10 = torch.ops.aten.where.self(ne_34, getitem_54, full_default_1)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(where_10, 1);  where_10 = None
        gather_5 = torch.ops.aten.gather.default(sub_57, 1, unsqueeze_5);  sub_57 = unsqueeze_5 = None
        squeeze_5 = torch.ops.aten.squeeze.dim(gather_5, 1);  gather_5 = None
        neg_5 = torch.ops.aten.neg.default(squeeze_5);  squeeze_5 = None
        where_11 = torch.ops.aten.where.self(ne_34, neg_5, full_default_2);  ne_34 = neg_5 = None
        sum_18 = torch.ops.aten.sum.default(where_11);  where_11 = None
        add_133 = torch.ops.aten.add.Tensor(add_120, sum_18);  add_120 = sum_18 = None
        mm_6 = torch.ops.aten.mm.default(getitem_6, permute_1);  getitem_6 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(mm_6, torch.float32)
        amax_6 = torch.ops.aten.amax.default(convert_element_type_26, [1], True)
        sub_62 = torch.ops.aten.sub.Tensor(convert_element_type_26, amax_6);  convert_element_type_26 = None
        exp_6 = torch.ops.aten.exp.default(sub_62)
        sum_19 = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
        log_6 = torch.ops.aten.log.default(sum_19);  sum_19 = None
        sub_63 = torch.ops.aten.sub.Tensor(sub_62, log_6);  sub_62 = None
        getitem_62 = split_2[6];  split_2 = None
        ne_40 = torch.ops.aten.ne.Scalar(getitem_62, -100)
        where_12 = torch.ops.aten.where.self(ne_40, getitem_62, full_default_1)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather_6 = torch.ops.aten.gather.default(sub_63, 1, unsqueeze_6);  sub_63 = unsqueeze_6 = None
        squeeze_6 = torch.ops.aten.squeeze.dim(gather_6, 1);  gather_6 = None
        neg_6 = torch.ops.aten.neg.default(squeeze_6);  squeeze_6 = None
        where_13 = torch.ops.aten.where.self(ne_40, neg_6, full_default_2);  ne_40 = neg_6 = full_default_2 = None
        sum_21 = torch.ops.aten.sum.default(where_13);  where_13 = None
        add_146 = torch.ops.aten.add.Tensor(add_133, sum_21);  add_133 = sum_21 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(primals_6, torch.float32);  primals_6 = None
        div = torch.ops.aten.div.Tensor(add_146, convert_element_type_28);  add_146 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(getitem_62, 1);  getitem_62 = None
        ne_43 = torch.ops.aten.ne.Scalar(unsqueeze_7, -100)
        where_14 = torch.ops.aten.where.self(ne_43, unsqueeze_7, full_default_1);  unsqueeze_7 = None
        permute_8 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(getitem_54, 1);  getitem_54 = None
        ne_45 = torch.ops.aten.ne.Scalar(unsqueeze_8, -100)
        where_16 = torch.ops.aten.where.self(ne_45, unsqueeze_8, full_default_1);  unsqueeze_8 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(getitem_46, 1);  getitem_46 = None
        ne_47 = torch.ops.aten.ne.Scalar(unsqueeze_9, -100)
        where_18 = torch.ops.aten.where.self(ne_47, unsqueeze_9, full_default_1);  unsqueeze_9 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(getitem_38, 1);  getitem_38 = None
        ne_49 = torch.ops.aten.ne.Scalar(unsqueeze_10, -100)
        where_20 = torch.ops.aten.where.self(ne_49, unsqueeze_10, full_default_1);  unsqueeze_10 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(getitem_30, 1);  getitem_30 = None
        ne_51 = torch.ops.aten.ne.Scalar(unsqueeze_11, -100)
        where_22 = torch.ops.aten.where.self(ne_51, unsqueeze_11, full_default_1);  unsqueeze_11 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(getitem_22, 1);  getitem_22 = None
        ne_53 = torch.ops.aten.ne.Scalar(unsqueeze_12, -100)
        where_24 = torch.ops.aten.where.self(ne_53, unsqueeze_12, full_default_1);  unsqueeze_12 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(getitem_14, 1);  getitem_14 = None
        ne_55 = torch.ops.aten.ne.Scalar(unsqueeze_13, -100)
        where_26 = torch.ops.aten.where.self(ne_55, unsqueeze_13, full_default_1);  unsqueeze_13 = full_default_1 = None
        return (div, mm, amax, log, mm_1, amax_1, log_1, mm_2, amax_2, log_2, mm_3, amax_3, log_3, mm_4, amax_4, log_4, mm_5, amax_5, log_5, mm_6, amax_6, log_6, convert_element_type_28, ne_43, where_14, permute_8, ne_45, where_16, ne_47, where_18, ne_49, where_20, ne_51, where_22, ne_53, where_24, ne_55, where_26, primals_4, primals_2, sym_size_int_5)
        
def load_args(reader):
    buf0 = reader.storage(None, 1158266880, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (201088, 2880), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    reader.symint(1024)  # primals_2
    buf1 = reader.storage(None, 8*s0, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (1, s0), dtype=torch.int64, is_leaf=True)  # primals_3
    reader.symint(1024)  # primals_4
    buf2 = reader.storage(None, 5760*s1, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (1, s2, 2880), dtype=torch.bfloat16, is_leaf=True)  # primals_5
    buf3 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf3, (), dtype=torch.int64, is_leaf=True)  # primals_6
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)