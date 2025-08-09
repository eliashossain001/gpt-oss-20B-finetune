class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "Sym(s2)", primals_2: "Sym(s0)", sym_size_int_5: "Sym(s2 - 6*(((s2 + 6)//7)))", mm: "bf16[((s2 + 6)//7), 201088]", amax: "f32[((s2 + 6)//7), 1]", log: "f32[((s2 + 6)//7), 1]", mm_1: "bf16[((s2 + 6)//7), 201088]", amax_1: "f32[((s2 + 6)//7), 1]", log_1: "f32[((s2 + 6)//7), 1]", mm_2: "bf16[((s2 + 6)//7), 201088]", amax_2: "f32[((s2 + 6)//7), 1]", log_2: "f32[((s2 + 6)//7), 1]", mm_3: "bf16[((s2 + 6)//7), 201088]", amax_3: "f32[((s2 + 6)//7), 1]", log_3: "f32[((s2 + 6)//7), 1]", mm_4: "bf16[((s2 + 6)//7), 201088]", amax_4: "f32[((s2 + 6)//7), 1]", log_4: "f32[((s2 + 6)//7), 1]", mm_5: "bf16[((s2 + 6)//7), 201088]", amax_5: "f32[((s2 + 6)//7), 1]", log_5: "f32[((s2 + 6)//7), 1]", mm_6: "bf16[s2 - 6*(((s2 + 6)//7)), 201088]", amax_6: "f32[s2 - 6*(((s2 + 6)//7)), 1]", log_6: "f32[s2 - 6*(((s2 + 6)//7)), 1]", convert_element_type_28: "f32[]", ne_43: "b8[s0 - 6*(((s0 + 6)//7)), 1]", where_14: "i64[s0 - 6*(((s0 + 6)//7)), 1]", permute_8: "bf16[201088, 2880]", ne_45: "b8[((s0 + 6)//7), 1]", where_16: "i64[((s0 + 6)//7), 1]", ne_47: "b8[((s0 + 6)//7), 1]", where_18: "i64[((s0 + 6)//7), 1]", ne_49: "b8[((s0 + 6)//7), 1]", where_20: "i64[((s0 + 6)//7), 1]", ne_51: "b8[((s0 + 6)//7), 1]", where_22: "i64[((s0 + 6)//7), 1]", ne_53: "b8[((s0 + 6)//7), 1]", where_24: "i64[((s0 + 6)//7), 1]", ne_55: "b8[((s0 + 6)//7), 1]", where_26: "i64[((s0 + 6)//7), 1]", tangents_1: "f32[]"):
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:560 in _unsloth_compiled_fused_ce_loss_function, code: loss = loss / torch.tensor(divisor, dtype = torch.float32)
        div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_28);  tangents_1 = convert_element_type_28 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_upon_const_tensor: "f32[s2 - 6*(((s2 + 6)//7)), 201088]" = torch__inductor_fx_passes_post_grad_scatter_upon_const_tensor(shape = [sym_size_int_5, 201088], background_val = 0, dtype = torch.float32, dim = 1, selector = where_14, val = -1.0);  sym_size_int_5 = where_14 = None
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "f32[s0 - 6*(((s0 + 6)//7)), 1]" = torch.ops.aten.where.self(ne_43, div_1, full_default_2);  ne_43 = None
        mul_111: "f32[s2 - 6*(((s2 + 6)//7)), 201088]" = torch.ops.aten.mul.Tensor(scatter_upon_const_tensor, where_15);  scatter_upon_const_tensor = where_15 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_26: "f32[s2 - 6*(((s2 + 6)//7)), 201088]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_62: "f32[s2 - 6*(((s2 + 6)//7)), 201088]" = torch.ops.aten.sub.Tensor(convert_element_type_26, amax_6);  convert_element_type_26 = amax_6 = None
        sub_63: "f32[s2 - 6*(((s2 + 6)//7)), 201088]" = torch.ops.aten.sub.Tensor(sub_62, log_6);  sub_62 = log_6 = None
        exp_7: "f32[s2 - 6*(((s2 + 6)//7)), 201088]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
        sum_22: "f32[s2 - 6*(((s2 + 6)//7)), 1]" = torch.ops.aten.sum.dim_IntList(mul_111, [1], True)
        mul_112: "f32[s2 - 6*(((s2 + 6)//7)), 201088]" = torch.ops.aten.mul.Tensor(exp_7, sum_22);  exp_7 = sum_22 = None
        sub_66: "f32[s2 - 6*(((s2 + 6)//7)), 201088]" = torch.ops.aten.sub.Tensor(mul_111, mul_112);  mul_111 = mul_112 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_29: "bf16[s2 - 6*(((s2 + 6)//7)), 201088]" = torch.ops.prims.convert_element_type.default(sub_66, torch.bfloat16);  sub_66 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_7: "bf16[s2 - 6*(((s2 + 6)//7)), 2880]" = torch.ops.aten.mm.default(convert_element_type_29, permute_8);  convert_element_type_29 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:530 in _unsloth_compiled_fused_ce_loss_function, code: __shift_states = torch.chunk(hidden_states, n_chunks, dim = 0)
        add_19: "Sym(s2 + 7)" = primals_4 + 7
        sub_8: "Sym(s2 + 6)" = add_19 - 1;  add_19 = None
        floordiv: "Sym(((s2 + 6)//7))" = sub_8 // 7;  sub_8 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        full_default_18: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.full.default([floordiv, 201088], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  floordiv = None
        scatter_1: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.scatter.value(full_default_18, 1, where_16, -1.0);  where_16 = None
        where_17: "f32[((s0 + 6)//7), 1]" = torch.ops.aten.where.self(ne_45, div_1, full_default_2);  ne_45 = None
        mul_113: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.mul.Tensor(scatter_1, where_17);  scatter_1 = where_17 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_22: "f32[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_56: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(convert_element_type_22, amax_5);  convert_element_type_22 = amax_5 = None
        sub_57: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(sub_56, log_5);  sub_56 = log_5 = None
        exp_8: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
        sum_23: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.sum.dim_IntList(mul_113, [1], True)
        mul_114: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.mul.Tensor(exp_8, sum_23);  exp_8 = sum_23 = None
        sub_67: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_32: "bf16[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(sub_67, torch.bfloat16);  sub_67 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_8: "bf16[((s2 + 6)//7), 2880]" = torch.ops.aten.mm.default(convert_element_type_32, permute_8);  convert_element_type_32 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_2: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.scatter.value(full_default_18, 1, where_18, -1.0);  where_18 = None
        where_19: "f32[((s0 + 6)//7), 1]" = torch.ops.aten.where.self(ne_47, div_1, full_default_2);  ne_47 = None
        mul_115: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.mul.Tensor(scatter_2, where_19);  scatter_2 = where_19 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_18: "f32[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_50: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(convert_element_type_18, amax_4);  convert_element_type_18 = amax_4 = None
        sub_51: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(sub_50, log_4);  sub_50 = log_4 = None
        exp_9: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
        sum_24: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.sum.dim_IntList(mul_115, [1], True)
        mul_116: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.mul.Tensor(exp_9, sum_24);  exp_9 = sum_24 = None
        sub_68: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_35: "bf16[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(sub_68, torch.bfloat16);  sub_68 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_9: "bf16[((s2 + 6)//7), 2880]" = torch.ops.aten.mm.default(convert_element_type_35, permute_8);  convert_element_type_35 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_3: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.scatter.value(full_default_18, 1, where_20, -1.0);  where_20 = None
        where_21: "f32[((s0 + 6)//7), 1]" = torch.ops.aten.where.self(ne_49, div_1, full_default_2);  ne_49 = None
        mul_117: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.mul.Tensor(scatter_3, where_21);  scatter_3 = where_21 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_14: "f32[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_44: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(convert_element_type_14, amax_3);  convert_element_type_14 = amax_3 = None
        sub_45: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(sub_44, log_3);  sub_44 = log_3 = None
        exp_10: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
        sum_25: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [1], True)
        mul_118: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.mul.Tensor(exp_10, sum_25);  exp_10 = sum_25 = None
        sub_69: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_38: "bf16[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(sub_69, torch.bfloat16);  sub_69 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_10: "bf16[((s2 + 6)//7), 2880]" = torch.ops.aten.mm.default(convert_element_type_38, permute_8);  convert_element_type_38 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_4: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.scatter.value(full_default_18, 1, where_22, -1.0);  where_22 = None
        where_23: "f32[((s0 + 6)//7), 1]" = torch.ops.aten.where.self(ne_51, div_1, full_default_2);  ne_51 = None
        mul_119: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.mul.Tensor(scatter_4, where_23);  scatter_4 = where_23 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_10: "f32[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_38: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_2);  convert_element_type_10 = amax_2 = None
        sub_39: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(sub_38, log_2);  sub_38 = log_2 = None
        exp_11: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_26: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [1], True)
        mul_120: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.mul.Tensor(exp_11, sum_26);  exp_11 = sum_26 = None
        sub_70: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(mul_119, mul_120);  mul_119 = mul_120 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_41: "bf16[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(sub_70, torch.bfloat16);  sub_70 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_11: "bf16[((s2 + 6)//7), 2880]" = torch.ops.aten.mm.default(convert_element_type_41, permute_8);  convert_element_type_41 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_5: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.scatter.value(full_default_18, 1, where_24, -1.0);  where_24 = None
        where_25: "f32[((s0 + 6)//7), 1]" = torch.ops.aten.where.self(ne_53, div_1, full_default_2);  ne_53 = None
        mul_121: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.mul.Tensor(scatter_5, where_25);  scatter_5 = where_25 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_6: "f32[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_32: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(convert_element_type_6, amax_1);  convert_element_type_6 = amax_1 = None
        sub_33: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(sub_32, log_1);  sub_32 = log_1 = None
        exp_12: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_27: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [1], True)
        mul_122: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.mul.Tensor(exp_12, sum_27);  exp_12 = sum_27 = None
        sub_71: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_44: "bf16[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(sub_71, torch.bfloat16);  sub_71 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_12: "bf16[((s2 + 6)//7), 2880]" = torch.ops.aten.mm.default(convert_element_type_44, permute_8);  convert_element_type_44 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_6: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.scatter.value(full_default_18, 1, where_26, -1.0);  full_default_18 = where_26 = None
        where_27: "f32[((s0 + 6)//7), 1]" = torch.ops.aten.where.self(ne_55, div_1, full_default_2);  ne_55 = div_1 = full_default_2 = None
        mul_123: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.mul.Tensor(scatter_6, where_27);  scatter_6 = where_27 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_2: "f32[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_26: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        sub_27: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
        exp_13: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_28: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [1], True)
        mul_124: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.mul.Tensor(exp_13, sum_28);  exp_13 = sum_28 = None
        sub_72: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_47: "bf16[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(sub_72, torch.bfloat16);  sub_72 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_13: "bf16[((s2 + 6)//7), 2880]" = torch.ops.aten.mm.default(convert_element_type_47, permute_8);  convert_element_type_47 = permute_8 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:530 in _unsloth_compiled_fused_ce_loss_function, code: __shift_states = torch.chunk(hidden_states, n_chunks, dim = 0)
        cat: "bf16[s2, 2880]" = torch.ops.aten.cat.default([mm_13, mm_12, mm_11, mm_10, mm_9, mm_8, mm_7]);  mm_13 = mm_12 = mm_11 = mm_10 = mm_9 = mm_8 = mm_7 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:528 in _unsloth_compiled_fused_ce_loss_function, code: hidden_states = hidden_states.view(-1, hd)
        view_9: "bf16[1, s2, 2880]" = torch.ops.aten.reshape.default(cat, [1, primals_4, 2880]);  cat = primals_4 = None
        return (None, None, None, None, view_9, None)
        