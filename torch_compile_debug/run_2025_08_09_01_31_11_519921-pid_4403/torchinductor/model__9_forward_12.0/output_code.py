# AOT ID: ['9_forward']
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


# kernel path: /tmp/torchinductor_research/tp/ctpnlunqgle2rhuwo5zjyfellkww6mzn7mtklnwno3ex4nfnbydg.py
# Topologically Sorted Source Nodes: [float_7, cross_entropy_loss_6], Original ATen: [aten._to_copy, aten._log_softmax]
# Source node to ATen node mapping:
#   cross_entropy_loss_6 => amax_6
#   float_7 => convert_element_type_26
# Graph fragment:
#   %convert_element_type_26 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_6, torch.float32), kwargs = {})
#   %amax_6 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_26, [1], True), kwargs = {})
triton_red_fused__log_softmax__to_copy_0 = async_compile.triton('triton_red_fused__log_softmax__to_copy_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 50272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 50272*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/li/clivkbuctl7hr3isu22fvp2joqfr3an2drmuhfc7sun5kq3a4rbp.py
# Topologically Sorted Source Nodes: [float_7, cross_entropy_loss_6], Original ATen: [aten._to_copy, aten._log_softmax]
# Source node to ATen node mapping:
#   cross_entropy_loss_6 => amax_6
#   float_7 => convert_element_type_26
# Graph fragment:
#   %convert_element_type_26 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_6, torch.float32), kwargs = {})
#   %amax_6 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_26, [1], True), kwargs = {})
triton_per_fused__log_softmax__to_copy_1 = async_compile.triton('triton_per_fused__log_softmax__to_copy_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax__to_copy_1(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 4*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/wx/cwx27bjsl5hyb52kkk2tna74fo2d6umg6vtbjeg2qa34odx37pwc.py
# Topologically Sorted Source Nodes: [float_7, cross_entropy_loss_6], Original ATen: [aten._to_copy, aten._log_softmax]
# Source node to ATen node mapping:
#   cross_entropy_loss_6 => exp_6, sub_62, sum_19
#   float_7 => convert_element_type_26
# Graph fragment:
#   %convert_element_type_26 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_6, torch.float32), kwargs = {})
#   %sub_62 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_26, %amax_6), kwargs = {})
#   %exp_6 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_62,), kwargs = {})
#   %sum_19 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_6, [1], True), kwargs = {})
triton_red_fused__log_softmax__to_copy_2 = async_compile.triton('triton_red_fused__log_softmax__to_copy_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy_2(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 50272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 4
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + 50272*x3), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 - tmp2
        tmp4 = tl_math.exp(tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/qt/cqtuphs2srtkguxj5vdjr4c3wdsunj3ox3tmciiysdvpydmdoh3s.py
# Topologically Sorted Source Nodes: [float_7, cross_entropy_loss_6], Original ATen: [aten._to_copy, aten._log_softmax]
# Source node to ATen node mapping:
#   cross_entropy_loss_6 => exp_6, log_6, sub_62, sum_19
#   float_7 => convert_element_type_26
# Graph fragment:
#   %convert_element_type_26 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_6, torch.float32), kwargs = {})
#   %sub_62 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_26, %amax_6), kwargs = {})
#   %exp_6 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_62,), kwargs = {})
#   %sum_19 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_6, [1], True), kwargs = {})
#   %log_6 : [num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%sum_19,), kwargs = {})
triton_per_fused__log_softmax__to_copy_3 = async_compile.triton('triton_per_fused__log_softmax__to_copy_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__to_copy_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax__to_copy_3(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 4*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tl_math.log(tmp4)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/yd/cydr7fchfci7tyf6h3izwcugl2xh7rs6xvt5lqlt762dh3d33nmz.py
# Topologically Sorted Source Nodes: [float_1, cross_entropy_loss], Original ATen: [aten._to_copy, aten._log_softmax]
# Source node to ATen node mapping:
#   cross_entropy_loss => amax
#   float_1 => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_2, [1], True), kwargs = {})
triton_red_fused__log_softmax__to_copy_4 = async_compile.triton('triton_red_fused__log_softmax__to_copy_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy_4(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 67030
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 3)
    x1 = xindex // 3
    _tmp8 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + 67030*x0
        tmp1 = tl.full([1, 1], 201088, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r2 + 67030*x0 + 201088*x1), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.full(tmp4.shape, float("-inf"), tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/5f/c5fecgxttchc5orletmbvwxjmrbok77sklcv3aduwvxjkgxaxq3m.py
# Topologically Sorted Source Nodes: [float_1, cross_entropy_loss], Original ATen: [aten._to_copy, aten._log_softmax]
# Source node to ATen node mapping:
#   cross_entropy_loss => amax
#   float_1 => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_2, [1], True), kwargs = {})
triton_per_fused__log_softmax__to_copy_5 = async_compile.triton('triton_per_fused__log_softmax__to_copy_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__to_copy_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax__to_copy_5(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/gq/cgqirtvw5nemm2wlvm3k2qwe6ra7lb3tm4gtfgwkjf5ue7qvwwrg.py
# Topologically Sorted Source Nodes: [float_1, cross_entropy_loss], Original ATen: [aten._to_copy, aten._log_softmax]
# Source node to ATen node mapping:
#   cross_entropy_loss => exp, sub_26, sum_1
#   float_1 => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %sub_26 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_26,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_red_fused__log_softmax__to_copy_6 = async_compile.triton('triton_red_fused__log_softmax__to_copy_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy_6(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 67030
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 3)
    x1 = xindex // 3
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + 67030*x0
        tmp1 = tl.full([1, 1], 201088, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r2 + 67030*x0 + 201088*x1), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tl_math.exp(tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/ds/cds6fm7jpryqbx4yhfcsotnsqijmotdk5u7r4ewghmd5zpwbk4kf.py
# Topologically Sorted Source Nodes: [float_1, cross_entropy_loss], Original ATen: [aten._to_copy, aten._log_softmax]
# Source node to ATen node mapping:
#   cross_entropy_loss => exp, log, sub_26, sum_1
#   float_1 => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %sub_26 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_26,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : [num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
triton_per_fused__log_softmax__to_copy_7 = async_compile.triton('triton_per_fused__log_softmax__to_copy_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__to_copy_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax__to_copy_7(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl_math.log(tmp4)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/ln/cln3zj42jsgu24yzfolpfp6he4ealm2uavec6t3ckqpijypmafwq.py
# Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_1, full_default_2, ne_4, neg, sum_3, where_1
# Graph fragment:
#   %ne_4 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_14, -100), kwargs = {})
#   %full_default_1 : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_2 : [num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %neg, %full_default_2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_55 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_13, -100), kwargs = {})
#   %where_26 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_55, %unsqueeze_13, %full_default_1), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_8 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': (9,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(1 + r0, [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 201088, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 201088)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 201088")
        tmp17 = tl.load(in_ptr2 + (tmp15 + 201088*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/jq/cjqvkog6udxregqsixdw5jg63ca4vfk6xrjgmrbr35docnv57pn4.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_1], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_1, full_default_2
#   cross_entropy_loss_1 => ne_10, neg_1, sum_6, where_3
# Graph fragment:
#   %full_default_1 : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_10 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_22, -100), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_1,), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_10, %neg_1, %full_default_2), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_3,), kwargs = {})
#   %ne_53 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_12, -100), kwargs = {})
#   %where_24 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_53, %unsqueeze_12, %full_default_1), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_9 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': (9,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr1 + (r0 + ((6 + ks0) // 7)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0 + ((6 + ks0) // 7)
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(1 + r0 + ((6 + ks0) // 7), [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 201088, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 201088)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 201088")
        tmp17 = tl.load(in_ptr2 + (tmp15 + 201088*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/pa/cpafqzws3bbr6bm2rsn26oitxqi2eimdzwezp6cnd4ay3ifqijif.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_2], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_1, full_default_2
#   cross_entropy_loss_2 => ne_16, neg_2, sum_9, where_5
# Graph fragment:
#   %full_default_1 : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_16 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_30, -100), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_2,), kwargs = {})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_16, %neg_2, %full_default_2), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_5,), kwargs = {})
#   %ne_51 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_11, -100), kwargs = {})
#   %where_22 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_51, %unsqueeze_11, %full_default_1), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_10 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': (9,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr1 + (r0 + 2*((6 + ks0) // 7)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0 + 2*((6 + ks0) // 7)
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(1 + r0 + 2*((6 + ks0) // 7), [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 201088, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 201088)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 201088")
        tmp17 = tl.load(in_ptr2 + (tmp15 + 201088*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/ne/cne27bbh2rwuqvcba26kn4rztjop2fth6htnxy4cpq54if4bvzyu.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_3], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_1, full_default_2
#   cross_entropy_loss_3 => ne_22, neg_3, sum_12, where_7
# Graph fragment:
#   %full_default_1 : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_22 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_38, -100), kwargs = {})
#   %neg_3 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_3,), kwargs = {})
#   %where_7 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_22, %neg_3, %full_default_2), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_7,), kwargs = {})
#   %ne_49 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_10, -100), kwargs = {})
#   %where_20 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_49, %unsqueeze_10, %full_default_1), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_11 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': (9,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr1 + (r0 + 3*((6 + ks0) // 7)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0 + 3*((6 + ks0) // 7)
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(1 + r0 + 3*((6 + ks0) // 7), [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 201088, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 201088)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 201088")
        tmp17 = tl.load(in_ptr2 + (tmp15 + 201088*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/6o/c6oyneawmlrjqfllehcsa6zquv2cvyg6mxlmek7tpiit4gvsmi7t.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_4], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_1, full_default_2
#   cross_entropy_loss_4 => ne_28, neg_4, sum_15, where_9
# Graph fragment:
#   %full_default_1 : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_28 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_46, -100), kwargs = {})
#   %neg_4 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_4,), kwargs = {})
#   %where_9 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_28, %neg_4, %full_default_2), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_9,), kwargs = {})
#   %ne_47 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_9, -100), kwargs = {})
#   %where_18 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_47, %unsqueeze_9, %full_default_1), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_12 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': (9,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr1 + (r0 + 4*((6 + ks0) // 7)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0 + 4*((6 + ks0) // 7)
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(1 + r0 + 4*((6 + ks0) // 7), [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 201088, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 201088)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 201088")
        tmp17 = tl.load(in_ptr2 + (tmp15 + 201088*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/mu/cmugz455cmljsycce257bvbohlairw5oztsgny3sjfxqzq5hn3lz.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_5], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_1, full_default_2
#   cross_entropy_loss_5 => ne_34, neg_5, sum_18, where_11
# Graph fragment:
#   %full_default_1 : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_34 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_54, -100), kwargs = {})
#   %neg_5 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_5,), kwargs = {})
#   %where_11 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_34, %neg_5, %full_default_2), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_11,), kwargs = {})
#   %ne_45 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_8, -100), kwargs = {})
#   %where_16 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_45, %unsqueeze_8, %full_default_1), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_13 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': (9,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr1 + (r0 + 5*((6 + ks0) // 7)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0 + 5*((6 + ks0) // 7)
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(1 + r0 + 5*((6 + ks0) // 7), [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 201088, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 201088)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 201088")
        tmp17 = tl.load(in_ptr2 + (tmp15 + 201088*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_research/uv/cuvr6fzquwcipcz3xdzebjw3lsqbzlyuix4brx2hffcpaa73uxtx.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, loss, loss_1, loss_2, loss_3, loss_4, loss_5, cross_entropy_loss_6, loss_6, tensor, loss_7], Original ATen: [aten.nll_loss_forward, aten.add, aten._to_copy, aten.div, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_1, full_default_2
#   cross_entropy_loss_6 => ne_40, neg_6, sum_21, where_13
#   loss => add_68
#   loss_1 => add_81
#   loss_2 => add_94
#   loss_3 => add_107
#   loss_4 => add_120
#   loss_5 => add_133
#   loss_6 => add_146
#   loss_7 => div
#   tensor => convert_element_type_28
# Graph fragment:
#   %full_default_1 : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_3, 0.0), kwargs = {})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_68, %sum_6), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_81, %sum_9), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_94, %sum_12), kwargs = {})
#   %add_120 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_107, %sum_15), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_120, %sum_18), kwargs = {})
#   %ne_40 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_62, -100), kwargs = {})
#   %neg_6 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_6,), kwargs = {})
#   %where_13 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_40, %neg_6, %full_default_2), kwargs = {})
#   %sum_21 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_13,), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_133, %sum_21), kwargs = {})
#   %convert_element_type_28 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_6, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_146, %convert_element_type_28), kwargs = {})
#   %ne_43 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_7, -100), kwargs = {})
#   %where_14 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_43, %unsqueeze_7, %full_default_1), kwargs = {})
triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_14 = async_compile.triton('triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'out_ptr3': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': (16,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 1, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, out_ptr2, out_ptr3, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr1 + (r0 + 6*((6 + ks0) // 7)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0 + 6*((6 + ks0) // 7)
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(1 + r0 + 6*((6 + ks0) // 7), [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 201088, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 201088)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 201088")
        tmp17 = tl.load(in_ptr2 + (tmp15 + 201088*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tmp29 = tl.load(in_ptr5 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, 1])
    tmp32 = tl.load(in_out_ptr0 + (0))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, 1])
    tmp36 = tl.load(in_ptr6 + (0))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, 1])
    tmp39 = tl.load(in_ptr7 + (0))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, 1])
    tmp42 = tl.load(in_ptr8 + (0))
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, 1])
    tmp45 = tl.load(in_ptr9 + (0))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, 1])
    tmp48 = tl.load(in_ptr10 + (0))
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, 1])
    tmp31 = tmp30.to(tl.float32)
    tmp34 = 0.0
    tmp35 = tmp33 + tmp34
    tmp38 = tmp35 + tmp37
    tmp41 = tmp38 + tmp40
    tmp44 = tmp41 + tmp43
    tmp47 = tmp44 + tmp46
    tmp50 = tmp47 + tmp49
    tmp51 = tmp50 + tmp27
    tmp52 = tmp51 / tmp31
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp31, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp52, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    s0 = primals_2
    s2 = primals_4
    assert_size_stride(primals_1, (201088, 2880), (2880, 1))
    assert_size_stride(primals_3, (1, s0), (s0, 1))
    assert_size_stride(primals_5, (1, s2, 2880), (2880*s2, 2880, 1))
    assert_size_stride(primals_6, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, s0), (s0, 1), torch.int64)
        buf43 = empty_strided_cuda((s2 + ((-6)*((6 + s2) // 7)), 201088), (201088, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_5, (s2 + ((-6)*((6 + s2) // 7)), 2880), (2880, 1), 17280*((6 + s2) // 7)), reinterpret_tensor(primals_1, (2880, 201088), (1, 2880), 0), out=buf43)
        buf44 = empty_strided_cuda((s2 + ((-6)*((6 + s2) // 7)), 1, 4), (4, ((-24)*((6 + s2) // 7)) + 4*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_7, cross_entropy_loss_6], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_0_xnumel = ((-24)*((6 + s2) // 7)) + 4*s2
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_0.run(buf43, buf44, triton_red_fused__log_softmax__to_copy_0_xnumel, 50272, grid=grid(triton_red_fused__log_softmax__to_copy_0_xnumel), stream=stream0)
        buf45 = empty_strided_cuda((s2 + ((-6)*((6 + s2) // 7)), 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_7, cross_entropy_loss_6], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_per_fused__log_softmax__to_copy_1_xnumel = s2 + ((-6)*((6 + s2) // 7))
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_1.run(buf44, buf45, triton_per_fused__log_softmax__to_copy_1_xnumel, 4, grid=grid(triton_per_fused__log_softmax__to_copy_1_xnumel), stream=stream0)
        buf46 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [float_7, cross_entropy_loss_6], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_2_xnumel = ((-24)*((6 + s2) // 7)) + 4*s2
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_2.run(buf43, buf45, buf46, triton_red_fused__log_softmax__to_copy_2_xnumel, 50272, grid=grid(triton_red_fused__log_softmax__to_copy_2_xnumel), stream=stream0)
        buf47 = empty_strided_cuda((s2 + ((-6)*((6 + s2) // 7)), 1), (1, s2 + ((-6)*((6 + s2) // 7))), torch.float32)
        buf48 = reinterpret_tensor(buf47, (s2 + ((-6)*((6 + s2) // 7)), 1), (1, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [float_7, cross_entropy_loss_6], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_per_fused__log_softmax__to_copy_3_xnumel = s2 + ((-6)*((6 + s2) // 7))
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_3.run(buf48, buf46, triton_per_fused__log_softmax__to_copy_3_xnumel, 4, grid=grid(triton_per_fused__log_softmax__to_copy_3_xnumel), stream=stream0)
        del buf46
        buf1 = empty_strided_cuda(((6 + s2) // 7, 201088), (201088, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_5, ((6 + s2) // 7, 2880), (2880, 1), 0), reinterpret_tensor(primals_1, (2880, 201088), (1, 2880), 0), out=buf1)
        buf2 = empty_strided_cuda(((6 + s2) // 7, 1, 3), (3, 3*((6 + s2) // 7), 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_1, cross_entropy_loss], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_4_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_4.run(buf1, buf2, triton_red_fused__log_softmax__to_copy_4_xnumel, 67030, grid=grid(triton_red_fused__log_softmax__to_copy_4_xnumel), stream=stream0)
        buf3 = empty_strided_cuda(((6 + s2) // 7, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_1, cross_entropy_loss], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_per_fused__log_softmax__to_copy_5_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_5.run(buf2, buf3, triton_per_fused__log_softmax__to_copy_5_xnumel, 3, grid=grid(triton_per_fused__log_softmax__to_copy_5_xnumel), stream=stream0)
        buf4 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [float_1, cross_entropy_loss], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_6_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_6.run(buf1, buf3, buf4, triton_red_fused__log_softmax__to_copy_6_xnumel, 67030, grid=grid(triton_red_fused__log_softmax__to_copy_6_xnumel), stream=stream0)
        buf5 = empty_strided_cuda(((6 + s2) // 7, 1), (1, (6 + s2) // 7), torch.float32)
        buf6 = reinterpret_tensor(buf5, ((6 + s2) // 7, 1), (1, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [float_1, cross_entropy_loss], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_per_fused__log_softmax__to_copy_7_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_7.run(buf6, buf4, triton_per_fused__log_softmax__to_copy_7_xnumel, 3, grid=grid(triton_per_fused__log_softmax__to_copy_7_xnumel), stream=stream0)
        buf7 = empty_strided_cuda((), (), torch.float32)
        buf63 = empty_strided_cuda(((6 + s0) // 7, 1), (1, 1), torch.bool)
        buf64 = empty_strided_cuda(((6 + s0) // 7, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_8_rnumel = (6 + s0) // 7
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_8.run(primals_3, buf0, buf1, buf3, buf6, buf7, buf63, buf64, s0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_8_rnumel, grid=grid(1), stream=stream0)
        buf8 = empty_strided_cuda(((6 + s2) // 7, 201088), (201088, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_5, ((6 + s2) // 7, 2880), (2880, 1), 2880*((6 + s2) // 7)), reinterpret_tensor(primals_1, (2880, 201088), (1, 2880), 0), out=buf8)
        buf9 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [float_2, cross_entropy_loss_1], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_4_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_4.run(buf8, buf9, triton_red_fused__log_softmax__to_copy_4_xnumel, 67030, grid=grid(triton_red_fused__log_softmax__to_copy_4_xnumel), stream=stream0)
        buf10 = empty_strided_cuda(((6 + s2) // 7, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_2, cross_entropy_loss_1], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_per_fused__log_softmax__to_copy_5_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_5.run(buf9, buf10, triton_per_fused__log_softmax__to_copy_5_xnumel, 3, grid=grid(triton_per_fused__log_softmax__to_copy_5_xnumel), stream=stream0)
        buf11 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [float_2, cross_entropy_loss_1], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_6_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_6.run(buf8, buf10, buf11, triton_red_fused__log_softmax__to_copy_6_xnumel, 67030, grid=grid(triton_red_fused__log_softmax__to_copy_6_xnumel), stream=stream0)
        buf12 = empty_strided_cuda(((6 + s2) // 7, 1), (1, (6 + s2) // 7), torch.float32)
        buf13 = reinterpret_tensor(buf12, ((6 + s2) // 7, 1), (1, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [float_2, cross_entropy_loss_1], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_per_fused__log_softmax__to_copy_7_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_7.run(buf13, buf11, triton_per_fused__log_softmax__to_copy_7_xnumel, 3, grid=grid(triton_per_fused__log_softmax__to_copy_7_xnumel), stream=stream0)
        buf14 = empty_strided_cuda((), (), torch.float32)
        buf61 = empty_strided_cuda(((6 + s0) // 7, 1), (1, 1), torch.bool)
        buf62 = empty_strided_cuda(((6 + s0) // 7, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_1], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_9_rnumel = (6 + s0) // 7
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_9.run(primals_3, buf0, buf8, buf10, buf13, buf14, buf61, buf62, s0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_9_rnumel, grid=grid(1), stream=stream0)
        buf15 = empty_strided_cuda(((6 + s2) // 7, 201088), (201088, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_5, ((6 + s2) // 7, 2880), (2880, 1), 5760*((6 + s2) // 7)), reinterpret_tensor(primals_1, (2880, 201088), (1, 2880), 0), out=buf15)
        buf16 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [float_3, cross_entropy_loss_2], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_4_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_4.run(buf15, buf16, triton_red_fused__log_softmax__to_copy_4_xnumel, 67030, grid=grid(triton_red_fused__log_softmax__to_copy_4_xnumel), stream=stream0)
        buf17 = empty_strided_cuda(((6 + s2) // 7, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_3, cross_entropy_loss_2], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_per_fused__log_softmax__to_copy_5_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_5.run(buf16, buf17, triton_per_fused__log_softmax__to_copy_5_xnumel, 3, grid=grid(triton_per_fused__log_softmax__to_copy_5_xnumel), stream=stream0)
        buf18 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [float_3, cross_entropy_loss_2], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_6_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_6.run(buf15, buf17, buf18, triton_red_fused__log_softmax__to_copy_6_xnumel, 67030, grid=grid(triton_red_fused__log_softmax__to_copy_6_xnumel), stream=stream0)
        buf19 = empty_strided_cuda(((6 + s2) // 7, 1), (1, (6 + s2) // 7), torch.float32)
        buf20 = reinterpret_tensor(buf19, ((6 + s2) // 7, 1), (1, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [float_3, cross_entropy_loss_2], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_per_fused__log_softmax__to_copy_7_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_7.run(buf20, buf18, triton_per_fused__log_softmax__to_copy_7_xnumel, 3, grid=grid(triton_per_fused__log_softmax__to_copy_7_xnumel), stream=stream0)
        buf21 = empty_strided_cuda((), (), torch.float32)
        buf59 = empty_strided_cuda(((6 + s0) // 7, 1), (1, 1), torch.bool)
        buf60 = empty_strided_cuda(((6 + s0) // 7, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_2], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_10_rnumel = (6 + s0) // 7
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_10.run(primals_3, buf0, buf15, buf17, buf20, buf21, buf59, buf60, s0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_10_rnumel, grid=grid(1), stream=stream0)
        buf22 = empty_strided_cuda(((6 + s2) // 7, 201088), (201088, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_5, ((6 + s2) // 7, 2880), (2880, 1), 8640*((6 + s2) // 7)), reinterpret_tensor(primals_1, (2880, 201088), (1, 2880), 0), out=buf22)
        buf23 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [float_4, cross_entropy_loss_3], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_4_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_4.run(buf22, buf23, triton_red_fused__log_softmax__to_copy_4_xnumel, 67030, grid=grid(triton_red_fused__log_softmax__to_copy_4_xnumel), stream=stream0)
        buf24 = empty_strided_cuda(((6 + s2) // 7, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_4, cross_entropy_loss_3], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_per_fused__log_softmax__to_copy_5_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_5.run(buf23, buf24, triton_per_fused__log_softmax__to_copy_5_xnumel, 3, grid=grid(triton_per_fused__log_softmax__to_copy_5_xnumel), stream=stream0)
        buf25 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [float_4, cross_entropy_loss_3], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_6_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_6.run(buf22, buf24, buf25, triton_red_fused__log_softmax__to_copy_6_xnumel, 67030, grid=grid(triton_red_fused__log_softmax__to_copy_6_xnumel), stream=stream0)
        buf26 = empty_strided_cuda(((6 + s2) // 7, 1), (1, (6 + s2) // 7), torch.float32)
        buf27 = reinterpret_tensor(buf26, ((6 + s2) // 7, 1), (1, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [float_4, cross_entropy_loss_3], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_per_fused__log_softmax__to_copy_7_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_7.run(buf27, buf25, triton_per_fused__log_softmax__to_copy_7_xnumel, 3, grid=grid(triton_per_fused__log_softmax__to_copy_7_xnumel), stream=stream0)
        buf28 = empty_strided_cuda((), (), torch.float32)
        buf57 = empty_strided_cuda(((6 + s0) // 7, 1), (1, 1), torch.bool)
        buf58 = empty_strided_cuda(((6 + s0) // 7, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_3], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_11_rnumel = (6 + s0) // 7
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_11.run(primals_3, buf0, buf22, buf24, buf27, buf28, buf57, buf58, s0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_11_rnumel, grid=grid(1), stream=stream0)
        buf29 = empty_strided_cuda(((6 + s2) // 7, 201088), (201088, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_5, ((6 + s2) // 7, 2880), (2880, 1), 11520*((6 + s2) // 7)), reinterpret_tensor(primals_1, (2880, 201088), (1, 2880), 0), out=buf29)
        buf30 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [float_5, cross_entropy_loss_4], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_4_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_4.run(buf29, buf30, triton_red_fused__log_softmax__to_copy_4_xnumel, 67030, grid=grid(triton_red_fused__log_softmax__to_copy_4_xnumel), stream=stream0)
        buf31 = empty_strided_cuda(((6 + s2) // 7, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_5, cross_entropy_loss_4], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_per_fused__log_softmax__to_copy_5_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_5.run(buf30, buf31, triton_per_fused__log_softmax__to_copy_5_xnumel, 3, grid=grid(triton_per_fused__log_softmax__to_copy_5_xnumel), stream=stream0)
        buf32 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [float_5, cross_entropy_loss_4], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_6_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_6.run(buf29, buf31, buf32, triton_red_fused__log_softmax__to_copy_6_xnumel, 67030, grid=grid(triton_red_fused__log_softmax__to_copy_6_xnumel), stream=stream0)
        buf33 = empty_strided_cuda(((6 + s2) // 7, 1), (1, (6 + s2) // 7), torch.float32)
        buf34 = reinterpret_tensor(buf33, ((6 + s2) // 7, 1), (1, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [float_5, cross_entropy_loss_4], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_per_fused__log_softmax__to_copy_7_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_7.run(buf34, buf32, triton_per_fused__log_softmax__to_copy_7_xnumel, 3, grid=grid(triton_per_fused__log_softmax__to_copy_7_xnumel), stream=stream0)
        buf35 = empty_strided_cuda((), (), torch.float32)
        buf55 = empty_strided_cuda(((6 + s0) // 7, 1), (1, 1), torch.bool)
        buf56 = empty_strided_cuda(((6 + s0) // 7, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_4], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_12_rnumel = (6 + s0) // 7
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_12.run(primals_3, buf0, buf29, buf31, buf34, buf35, buf55, buf56, s0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_12_rnumel, grid=grid(1), stream=stream0)
        buf36 = empty_strided_cuda(((6 + s2) // 7, 201088), (201088, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_5, ((6 + s2) // 7, 2880), (2880, 1), 14400*((6 + s2) // 7)), reinterpret_tensor(primals_1, (2880, 201088), (1, 2880), 0), out=buf36)
        del primals_5
        buf37 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [float_6, cross_entropy_loss_5], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_4_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_4.run(buf36, buf37, triton_red_fused__log_softmax__to_copy_4_xnumel, 67030, grid=grid(triton_red_fused__log_softmax__to_copy_4_xnumel), stream=stream0)
        buf38 = empty_strided_cuda(((6 + s2) // 7, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_6, cross_entropy_loss_5], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_per_fused__log_softmax__to_copy_5_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_5.run(buf37, buf38, triton_per_fused__log_softmax__to_copy_5_xnumel, 3, grid=grid(triton_per_fused__log_softmax__to_copy_5_xnumel), stream=stream0)
        buf39 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [float_6, cross_entropy_loss_5], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_6_xnumel = 3*((6 + s2) // 7)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_6.run(buf36, buf38, buf39, triton_red_fused__log_softmax__to_copy_6_xnumel, 67030, grid=grid(triton_red_fused__log_softmax__to_copy_6_xnumel), stream=stream0)
        buf40 = empty_strided_cuda(((6 + s2) // 7, 1), (1, (6 + s2) // 7), torch.float32)
        buf41 = reinterpret_tensor(buf40, ((6 + s2) // 7, 1), (1, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [float_6, cross_entropy_loss_5], Original ATen: [aten._to_copy, aten._log_softmax]
        triton_per_fused__log_softmax__to_copy_7_xnumel = (6 + s2) // 7
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__to_copy_7.run(buf41, buf39, triton_per_fused__log_softmax__to_copy_7_xnumel, 3, grid=grid(triton_per_fused__log_softmax__to_copy_7_xnumel), stream=stream0)
        del buf39
        buf42 = empty_strided_cuda((), (), torch.float32)
        buf53 = empty_strided_cuda(((6 + s0) // 7, 1), (1, 1), torch.bool)
        buf54 = empty_strided_cuda(((6 + s0) // 7, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_5], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_13_rnumel = (6 + s0) // 7
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_13.run(primals_3, buf0, buf36, buf38, buf41, buf42, buf53, buf54, s0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_13_rnumel, grid=grid(1), stream=stream0)
        buf51 = empty_strided_cuda((s0 + ((-6)*((6 + s0) // 7)), 1), (1, 1), torch.bool)
        buf52 = empty_strided_cuda((s0 + ((-6)*((6 + s0) // 7)), 1), (1, 1), torch.int64)
        buf50 = empty_strided_cuda((), (), torch.float32)
        buf65 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, loss, loss_1, loss_2, loss_3, loss_4, loss_5, cross_entropy_loss_6, loss_6, tensor, loss_7], Original ATen: [aten.nll_loss_forward, aten.add, aten._to_copy, aten.div, aten.nll_loss_backward]
        triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_14_rnumel = s0 + ((-6)*((6 + s0) // 7))
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_14.run(buf65, primals_3, buf0, buf43, buf45, buf48, primals_6, buf14, buf21, buf28, buf35, buf42, buf51, buf52, buf50, s0, 1, triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_14_rnumel, grid=grid(1), stream=stream0)
        del buf0
        del buf14
        del buf21
        del buf28
        del buf35
        del buf42
        del primals_3
        del primals_6
    return (buf65, buf1, buf3, buf6, buf8, buf10, buf13, buf15, buf17, buf20, buf22, buf24, buf27, buf29, buf31, buf34, buf36, buf38, buf41, buf43, buf45, buf48, buf50, buf51, buf52, primals_1, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, s2, s0, s2 + ((-6)*((6 + s2) // 7)), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((201088, 2880), (2880, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = 1024
    primals_3 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_4 = 1024
    primals_5 = rand_strided((1, 1024, 2880), (2949120, 2880, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
