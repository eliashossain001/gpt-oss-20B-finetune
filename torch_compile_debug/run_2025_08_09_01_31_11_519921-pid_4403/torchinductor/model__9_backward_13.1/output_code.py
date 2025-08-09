# AOT ID: ['9_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_research/5x/c5x6q4vhuoh7dnsdvkx5r2vxgknl4abt4d6uu356jeqnj2arqyj4.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, float_7, cross_entropy_loss_6], Original ATen: [aten.div, aten.nll_loss_backward, aten.nll_loss_forward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2
#   cross_entropy_loss_6 => sub_62, sub_63
#   float_7 => convert_element_type_26
# Graph fragment:
#   %div_1 : [num_users=7] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_28), kwargs = {})
#   %scatter_upon_const_tensor : [num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [%sym_size_int_5, 201088], background_val: 0, dtype: torch.float32, dim: 1, selector: %where_14, val: -1.0})
#   %full_default_2 : [num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_15 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_43, %div_1, %full_default_2), kwargs = {})
#   %mul_111 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_upon_const_tensor, %where_15), kwargs = {})
#   %convert_element_type_26 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_6, torch.float32), kwargs = {})
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_26, %amax_6), kwargs = {})
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_62, %log_6), kwargs = {})
#   %exp_7 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_63,), kwargs = {})
#   %sum_22 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_111, [1], True), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_7, %sum_22), kwargs = {})
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_111, %mul_112), kwargs = {})
#   %convert_element_type_29 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_66, torch.bfloat16), kwargs = {})
triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 201088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = r1
        tmp2 = tmp0 == tmp1
        tmp3 = -1.0
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp11 = tmp8 / tmp10
        tmp12 = tl.where(tmp6, tmp11, tmp4)
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp22 = tl.load(in_ptr2 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr3 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp31 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + 201088*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = r1
        tmp18 = tmp0 == tmp17
        tmp19 = -1.0
        tmp20 = 0.0
        tmp21 = tl.where(tmp18, tmp19, tmp20)
        tmp26 = tmp23 / tmp25
        tmp27 = tl.where(tmp6, tmp26, tmp20)
        tmp28 = tmp21 * tmp27
        tmp30 = tmp29.to(tl.float32)
        tmp32 = tmp30 - tmp31
        tmp34 = tmp32 - tmp33
        tmp35 = tl_math.exp(tmp34)
        tmp36 = tmp35 * tmp15
        tmp37 = tmp28 - tmp36
        tmp38 = tmp37.to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + 201088*x0), tmp38, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/kb/ckbddi47g5ln6qo66enwxzzhmw7u6fii5puloerfquwjkpe2c7iq.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_18 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([%floordiv, 201088], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%full_default_18, 1, %where_16, -1.0), kwargs = {})
triton_poi_fused_nll_loss_backward_1 = async_compile.triton('triton_poi_fused_nll_loss_backward_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_nll_loss_backward_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/is/cis7dcvh2wgrhfqpjbfdkrzftqyk7nd6bej2obm3t6wldbkxz244.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_18 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([%floordiv, 201088], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%full_default_18, 1, %where_16, -1.0), kwargs = {})
triton_poi_fused_nll_loss_backward_2 = async_compile.triton('triton_poi_fused_nll_loss_backward_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_2', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_nll_loss_backward_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < 201088)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 201088")
    tmp2 = -1.0
    tl.store(out_ptr0 + (tmp0 + 201088*x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/kg/ckggv7d6gxig7yasbjilr2pbykwduzn7jpmjb27jkb2vgxcgs6xl.py
# Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2
# Graph fragment:
#   %div_1 : [num_users=7] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_28), kwargs = {})
#   %full_default_2 : [num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_17 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_45, %div_1, %full_default_2), kwargs = {})
#   %mul_113 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_1, %where_17), kwargs = {})
#   %sum_23 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_113, [1], True), kwargs = {})
triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3 = async_compile.triton('triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 67030
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 3)
    x1 = xindex // 3
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp7 = tl.load(in_ptr3 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + 67030*x0
        tmp1 = tl.full([1, 1], 201088, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r2 + 67030*x0 + 201088*x1), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp9 = tmp6 / tmp8
        tmp10 = 0.0
        tmp11 = tl.where(tmp4, tmp9, tmp10)
        tmp12 = tmp3 * tmp11
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/sf/csfl6gxmpl7ewylytvhzegfvjd4h3ivzzlvsjqugedr2wypptuew.py
# Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2
# Graph fragment:
#   %div_1 : [num_users=7] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_28), kwargs = {})
#   %full_default_2 : [num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_17 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_45, %div_1, %full_default_2), kwargs = {})
#   %mul_113 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_1, %where_17), kwargs = {})
#   %sum_23 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_113, [1], True), kwargs = {})
triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4 = async_compile.triton('triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 3*x0), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/oy/coyafq35cddtgslrxlsnubnvpu2wht4jww2igynlvvigliiocnoz.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, float_6, cross_entropy_loss_5], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2
#   cross_entropy_loss_5 => sub_56, sub_57
#   float_6 => convert_element_type_22
# Graph fragment:
#   %div_1 : [num_users=7] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_28), kwargs = {})
#   %full_default_2 : [num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_17 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_45, %div_1, %full_default_2), kwargs = {})
#   %mul_113 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_1, %where_17), kwargs = {})
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_5, torch.float32), kwargs = {})
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_22, %amax_5), kwargs = {})
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_56, %log_5), kwargs = {})
#   %exp_8 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_57,), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_8, %sum_23), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_113, %mul_114), kwargs = {})
#   %convert_element_type_32 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_67, torch.bfloat16), kwargs = {})
triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5 = async_compile.triton('triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 201088
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.load(in_ptr3 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp10 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tmp3 / tmp5
    tmp7 = 0.0
    tmp8 = tl.where(tmp1, tmp6, tmp7)
    tmp9 = tmp0 * tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 - tmp14
    tmp16 = tl_math.exp(tmp15)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_2, sym_size_int_5, mm, amax, log, mm_1, amax_1, log_1, mm_2, amax_2, log_2, mm_3, amax_3, log_3, mm_4, amax_4, log_4, mm_5, amax_5, log_5, mm_6, amax_6, log_6, convert_element_type_28, ne_43, where_14, permute_8, ne_45, where_16, ne_47, where_18, ne_49, where_20, ne_51, where_22, ne_53, where_24, ne_55, where_26, tangents_1 = args
    args.clear()
    s2 = primals_4
    s0 = primals_2
    assert_size_stride(mm, ((6 + s2) // 7, 201088), (201088, 1))
    assert_size_stride(amax, ((6 + s2) // 7, 1), (1, 1))
    assert_size_stride(log, ((6 + s2) // 7, 1), (1, 1))
    assert_size_stride(mm_1, ((6 + s2) // 7, 201088), (201088, 1))
    assert_size_stride(amax_1, ((6 + s2) // 7, 1), (1, 1))
    assert_size_stride(log_1, ((6 + s2) // 7, 1), (1, 1))
    assert_size_stride(mm_2, ((6 + s2) // 7, 201088), (201088, 1))
    assert_size_stride(amax_2, ((6 + s2) // 7, 1), (1, 1))
    assert_size_stride(log_2, ((6 + s2) // 7, 1), (1, 1))
    assert_size_stride(mm_3, ((6 + s2) // 7, 201088), (201088, 1))
    assert_size_stride(amax_3, ((6 + s2) // 7, 1), (1, 1))
    assert_size_stride(log_3, ((6 + s2) // 7, 1), (1, 1))
    assert_size_stride(mm_4, ((6 + s2) // 7, 201088), (201088, 1))
    assert_size_stride(amax_4, ((6 + s2) // 7, 1), (1, 1))
    assert_size_stride(log_4, ((6 + s2) // 7, 1), (1, 1))
    assert_size_stride(mm_5, ((6 + s2) // 7, 201088), (201088, 1))
    assert_size_stride(amax_5, ((6 + s2) // 7, 1), (1, 1))
    assert_size_stride(log_5, ((6 + s2) // 7, 1), (1, 1))
    assert_size_stride(mm_6, (s2 + ((-6)*((6 + s2) // 7)), 201088), (201088, 1))
    assert_size_stride(amax_6, (s2 + ((-6)*((6 + s2) // 7)), 1), (1, 1))
    assert_size_stride(log_6, (s2 + ((-6)*((6 + s2) // 7)), 1), (1, 1))
    assert_size_stride(convert_element_type_28, (), ())
    assert_size_stride(ne_43, (s0 + ((-6)*((6 + s0) // 7)), 1), (1, 1))
    assert_size_stride(where_14, (s0 + ((-6)*((6 + s0) // 7)), 1), (1, 1))
    assert_size_stride(permute_8, (201088, 2880), (2880, 1))
    assert_size_stride(ne_45, ((6 + s0) // 7, 1), (1, 1))
    assert_size_stride(where_16, ((6 + s0) // 7, 1), (1, 1))
    assert_size_stride(ne_47, ((6 + s0) // 7, 1), (1, 1))
    assert_size_stride(where_18, ((6 + s0) // 7, 1), (1, 1))
    assert_size_stride(ne_49, ((6 + s0) // 7, 1), (1, 1))
    assert_size_stride(where_20, ((6 + s0) // 7, 1), (1, 1))
    assert_size_stride(ne_51, ((6 + s0) // 7, 1), (1, 1))
    assert_size_stride(where_22, ((6 + s0) // 7, 1), (1, 1))
    assert_size_stride(ne_53, ((6 + s0) // 7, 1), (1, 1))
    assert_size_stride(where_24, ((6 + s0) // 7, 1), (1, 1))
    assert_size_stride(ne_55, ((6 + s0) // 7, 1), (1, 1))
    assert_size_stride(where_26, ((6 + s0) // 7, 1), (1, 1))
    assert_size_stride(tangents_1, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = mm_6; del mm_6  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_7, cross_entropy_loss_6], Original ATen: [aten.div, aten.nll_loss_backward, aten.nll_loss_forward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0_xnumel = s2 + ((-6)*((6 + s2) // 7))
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0.run(buf1, where_14, ne_43, tangents_1, convert_element_type_28, amax_6, log_6, triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0_xnumel, 201088, grid=grid(triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0_xnumel), stream=stream0)
        del amax_6
        del log_6
        del ne_43
        del where_14
        buf39 = empty_strided_cuda((s2, 2880), (2880, 1), torch.bfloat16)
        buf2 = reinterpret_tensor(buf39, (s2 + ((-6)*((6 + s2) // 7)), 2880), (2880, 1), 17280*((6 + s2) // 7))  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_7, cross_entropy_loss_6], Original ATen: [aten.div, aten.nll_loss_backward, aten.nll_loss_forward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf1, permute_8, out=buf2)
        del buf1
        buf3 = empty_strided_cuda(((6 + s2) // 7, 201088), (201088, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1_xnumel = 201088*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_1.run(buf3, triton_poi_fused_nll_loss_backward_1_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_1_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = (6 + s0) // 7
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(where_16, buf3, triton_poi_fused_nll_loss_backward_2_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_2_xnumel), stream=stream0)
        del where_16
        buf5 = empty_strided_cuda(((6 + s2) // 7, 1, 3), (3, 3*((6 + s2) // 7), 1), torch.float32)
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf3, ne_45, tangents_1, convert_element_type_28, buf5, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel, 67030, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel), stream=stream0)
        buf6 = empty_strided_cuda(((6 + s2) // 7, 1), (1, (6 + s2) // 7), torch.float32)
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4.run(buf5, buf6, triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel, 3, grid=grid(triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel), stream=stream0)
        buf7 = mm_5; del mm_5  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_6, cross_entropy_loss_5], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel = 201088*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5.run(buf7, buf3, ne_45, tangents_1, convert_element_type_28, amax_5, log_5, buf6, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel), stream=stream0)
        del amax_5
        del log_5
        del ne_45
        buf8 = reinterpret_tensor(buf39, ((6 + s2) // 7, 2880), (2880, 1), 14400*((6 + s2) // 7))  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_6, cross_entropy_loss_5], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf7, permute_8, out=buf8)
        del buf7
        buf9 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1_xnumel = 201088*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_1.run(buf9, triton_poi_fused_nll_loss_backward_1_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_1_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = (6 + s0) // 7
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(where_18, buf9, triton_poi_fused_nll_loss_backward_2_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_2_xnumel), stream=stream0)
        del where_18
        buf11 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf9, ne_47, tangents_1, convert_element_type_28, buf11, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel, 67030, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel), stream=stream0)
        buf12 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4.run(buf11, buf12, triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel, 3, grid=grid(triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel), stream=stream0)
        buf13 = mm_4; del mm_4  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_5, cross_entropy_loss_4], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel = 201088*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5.run(buf13, buf9, ne_47, tangents_1, convert_element_type_28, amax_4, log_4, buf12, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel), stream=stream0)
        del amax_4
        del log_4
        del ne_47
        buf14 = reinterpret_tensor(buf39, ((6 + s2) // 7, 2880), (2880, 1), 11520*((6 + s2) // 7))  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_5, cross_entropy_loss_4], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf13, permute_8, out=buf14)
        del buf13
        buf15 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1_xnumel = 201088*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_1.run(buf15, triton_poi_fused_nll_loss_backward_1_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_1_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = (6 + s0) // 7
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(where_20, buf15, triton_poi_fused_nll_loss_backward_2_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_2_xnumel), stream=stream0)
        del where_20
        buf17 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf15, ne_49, tangents_1, convert_element_type_28, buf17, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel, 67030, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel), stream=stream0)
        buf18 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4.run(buf17, buf18, triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel, 3, grid=grid(triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel), stream=stream0)
        buf19 = mm_3; del mm_3  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_4, cross_entropy_loss_3], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel = 201088*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5.run(buf19, buf15, ne_49, tangents_1, convert_element_type_28, amax_3, log_3, buf18, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel), stream=stream0)
        del amax_3
        del log_3
        del ne_49
        buf20 = reinterpret_tensor(buf39, ((6 + s2) // 7, 2880), (2880, 1), 8640*((6 + s2) // 7))  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_4, cross_entropy_loss_3], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf19, permute_8, out=buf20)
        del buf19
        buf21 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1_xnumel = 201088*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_1.run(buf21, triton_poi_fused_nll_loss_backward_1_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_1_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = (6 + s0) // 7
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(where_22, buf21, triton_poi_fused_nll_loss_backward_2_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_2_xnumel), stream=stream0)
        del where_22
        buf23 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf21, ne_51, tangents_1, convert_element_type_28, buf23, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel, 67030, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel), stream=stream0)
        buf24 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4.run(buf23, buf24, triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel, 3, grid=grid(triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel), stream=stream0)
        buf25 = mm_2; del mm_2  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_3, cross_entropy_loss_2], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel = 201088*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5.run(buf25, buf21, ne_51, tangents_1, convert_element_type_28, amax_2, log_2, buf24, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel), stream=stream0)
        del amax_2
        del log_2
        del ne_51
        buf26 = reinterpret_tensor(buf39, ((6 + s2) // 7, 2880), (2880, 1), 5760*((6 + s2) // 7))  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_3, cross_entropy_loss_2], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf25, permute_8, out=buf26)
        del buf25
        buf27 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1_xnumel = 201088*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_1.run(buf27, triton_poi_fused_nll_loss_backward_1_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_1_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = (6 + s0) // 7
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(where_24, buf27, triton_poi_fused_nll_loss_backward_2_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_2_xnumel), stream=stream0)
        del where_24
        buf29 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf27, ne_53, tangents_1, convert_element_type_28, buf29, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel, 67030, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel), stream=stream0)
        buf30 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4.run(buf29, buf30, triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel, 3, grid=grid(triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel), stream=stream0)
        buf31 = mm_1; del mm_1  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_2, cross_entropy_loss_1], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel = 201088*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5.run(buf31, buf27, ne_53, tangents_1, convert_element_type_28, amax_1, log_1, buf30, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel), stream=stream0)
        del amax_1
        del log_1
        del ne_53
        buf32 = reinterpret_tensor(buf39, ((6 + s2) // 7, 2880), (2880, 1), 2880*((6 + s2) // 7))  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_2, cross_entropy_loss_1], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf31, permute_8, out=buf32)
        del buf31
        buf33 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1_xnumel = 201088*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_1.run(buf33, triton_poi_fused_nll_loss_backward_1_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_1_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = (6 + s0) // 7
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(where_26, buf33, triton_poi_fused_nll_loss_backward_2_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_2_xnumel), stream=stream0)
        del where_26
        buf35 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf33, ne_55, tangents_1, convert_element_type_28, buf35, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel, 67030, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel), stream=stream0)
        buf36 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4.run(buf35, buf36, triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel, 3, grid=grid(triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel), stream=stream0)
        del buf35
        buf37 = mm; del mm  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_1], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel = 201088*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5.run(buf37, buf33, ne_55, tangents_1, convert_element_type_28, amax, log, buf36, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_5_xnumel), stream=stream0)
        del amax
        del buf33
        del buf36
        del convert_element_type_28
        del log
        del ne_55
        del tangents_1
        buf38 = reinterpret_tensor(buf39, ((6 + s2) // 7, 2880), (2880, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_1], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf37, permute_8, out=buf38)
        del buf37
        del permute_8
    return (None, None, None, None, reinterpret_tensor(buf39, (1, s2, 2880), (2880*s2, 2880, 1), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = 1024
    primals_2 = 1024
    sym_size_int_5 = 142
    mm = rand_strided((147, 201088), (201088, 1), device='cuda:0', dtype=torch.bfloat16)
    amax = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_1 = rand_strided((147, 201088), (201088, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_1 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_1 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_2 = rand_strided((147, 201088), (201088, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_2 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_2 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_3 = rand_strided((147, 201088), (201088, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_3 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_3 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_4 = rand_strided((147, 201088), (201088, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_4 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_4 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_5 = rand_strided((147, 201088), (201088, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_5 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_5 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_6 = rand_strided((142, 201088), (201088, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_6 = rand_strided((142, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_6 = rand_strided((142, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_28 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    ne_43 = rand_strided((142, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_14 = rand_strided((142, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    permute_8 = rand_strided((201088, 2880), (2880, 1), device='cuda:0', dtype=torch.bfloat16)
    ne_45 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_16 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_47 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_18 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_49 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_20 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_51 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_22 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_53 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_24 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_55 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_26 = rand_strided((147, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_4, primals_2, sym_size_int_5, mm, amax, log, mm_1, amax_1, log_1, mm_2, amax_2, log_2, mm_3, amax_3, log_3, mm_4, amax_4, log_4, mm_5, amax_5, log_5, mm_6, amax_6, log_6, convert_element_type_28, ne_43, where_14, permute_8, ne_45, where_16, ne_47, where_18, ne_49, where_20, ne_51, where_22, ne_53, where_24, ne_55, where_26, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
