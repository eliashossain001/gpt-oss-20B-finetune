class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[201088, 2880]", primals_2: "Sym(s0)", primals_3: "i64[1, s0]", primals_4: "Sym(s2)", primals_5: "bf16[1, s2, 2880]", primals_6: "i64[]"):
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:517 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels = torch.empty_like(output_labels, device = device)
        empty: "i64[1, s0]" = torch.ops.aten.empty.memory_format([1, primals_2], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute: "i64[1, s0]" = torch.ops.aten.permute.default(empty, [0, 1]);  empty = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:518 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels[..., :-1] = output_labels[..., 1:]
        slice_1: "i64[1, s0 - 1]" = torch.ops.aten.slice.Tensor(primals_3, 1, 1, 9223372036854775807);  primals_3 = None
        
        # No stacktrace found for following nodes
        slice_scatter_default: "i64[1, s0]" = torch.ops.aten.slice_scatter.default(permute, slice_1, 1, 0, -1);  permute = slice_1 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:523 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels[..., -1] = -100
        full_default: "i64[]" = torch.ops.aten.full.default([], -100, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_1: "i64[1]" = torch.ops.aten.select.int(slice_scatter_default, 1, -1)
        copy_1: "i64[1]" = torch.ops.aten.copy.default(select_1, full_default);  select_1 = full_default = None
        
        # No stacktrace found for following nodes
        select_scatter_default: "i64[1, s0]" = torch.ops.aten.select_scatter.default(slice_scatter_default, copy_1, 1, -1);  slice_scatter_default = copy_1 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:528 in _unsloth_compiled_fused_ce_loss_function, code: hidden_states = hidden_states.view(-1, hd)
        view_1: "bf16[s2, 2880]" = torch.ops.aten.reshape.default(primals_5, [-1, 2880]);  primals_5 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:530 in _unsloth_compiled_fused_ce_loss_function, code: __shift_states = torch.chunk(hidden_states, n_chunks, dim = 0)
        add_19: "Sym(s2 + 7)" = primals_4 + 7
        sub_8: "Sym(s2 + 6)" = add_19 - 1;  add_19 = None
        floordiv: "Sym(((s2 + 6)//7))" = sub_8 // 7;  sub_8 = None
        split = torch.ops.aten.split.Tensor(view_1, floordiv);  view_1 = floordiv = None
        getitem: "bf16[((s2 + 6)//7), 2880]" = split[0]
        getitem_1: "bf16[((s2 + 6)//7), 2880]" = split[1]
        getitem_2: "bf16[((s2 + 6)//7), 2880]" = split[2]
        getitem_3: "bf16[((s2 + 6)//7), 2880]" = split[3]
        getitem_4: "bf16[((s2 + 6)//7), 2880]" = split[4]
        getitem_5: "bf16[((s2 + 6)//7), 2880]" = split[5]
        getitem_6: "bf16[s2 - 6*(((s2 + 6)//7)), 2880]" = split[6];  split = None
        sym_size_int_5: "Sym(s2 - 6*(((s2 + 6)//7)))" = torch.ops.aten.sym_size.int(getitem_6, 0)
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:531 in _unsloth_compiled_fused_ce_loss_function, code: __shift_labels = torch.chunk(shift_labels,  n_chunks, dim = 0)
        add_41: "Sym(s0 + 7)" = primals_2 + 7
        sub_16: "Sym(s0 + 6)" = add_41 - 1;  add_41 = None
        floordiv_1: "Sym(((s0 + 6)//7))" = sub_16 // 7;  sub_16 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        permute_1: "bf16[2880, 201088]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mm: "bf16[((s2 + 6)//7), 201088]" = torch.ops.aten.mm.default(getitem, permute_1);  getitem = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_2: "f32[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(mm, torch.float32)
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.amax.default(convert_element_type_2, [1], True)
        sub_26: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = None
        exp: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.exp.default(sub_26)
        sum_1: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_27: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = None
        view_2: "i64[s0]" = torch.ops.aten.reshape.default(select_scatter_default, [-1]);  select_scatter_default = None
        split_2 = torch.ops.aten.split.Tensor(view_2, floordiv_1);  view_2 = floordiv_1 = None
        getitem_14: "i64[((s0 + 6)//7)]" = split_2[0]
        ne_4: "b8[((s0 + 6)//7)]" = torch.ops.aten.ne.Scalar(getitem_14, -100)
        full_default_1: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[((s0 + 6)//7)]" = torch.ops.aten.where.self(ne_4, getitem_14, full_default_1)
        unsqueeze: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[((s0 + 6)//7), 1]" = torch.ops.aten.gather.default(sub_27, 1, unsqueeze);  sub_27 = unsqueeze = None
        squeeze: "f32[((s0 + 6)//7)]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[((s0 + 6)//7)]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[((s0 + 6)//7)]" = torch.ops.aten.where.self(ne_4, neg, full_default_2);  ne_4 = neg = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_68: "f32[]" = torch.ops.aten.add.Tensor(sum_3, 0.0);  sum_3 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_1: "bf16[((s2 + 6)//7), 201088]" = torch.ops.aten.mm.default(getitem_1, permute_1);  getitem_1 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_6: "f32[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32)
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax_1: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.amax.default(convert_element_type_6, [1], True)
        sub_32: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(convert_element_type_6, amax_1);  convert_element_type_6 = None
        exp_1: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.exp.default(sub_32)
        sum_4: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [1], True);  exp_1 = None
        log_1: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.log.default(sum_4);  sum_4 = None
        sub_33: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(sub_32, log_1);  sub_32 = None
        getitem_22: "i64[((s0 + 6)//7)]" = split_2[1]
        ne_10: "b8[((s0 + 6)//7)]" = torch.ops.aten.ne.Scalar(getitem_22, -100)
        where_2: "i64[((s0 + 6)//7)]" = torch.ops.aten.where.self(ne_10, getitem_22, full_default_1)
        unsqueeze_1: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
        gather_1: "f32[((s0 + 6)//7), 1]" = torch.ops.aten.gather.default(sub_33, 1, unsqueeze_1);  sub_33 = unsqueeze_1 = None
        squeeze_1: "f32[((s0 + 6)//7)]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1: "f32[((s0 + 6)//7)]" = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
        where_3: "f32[((s0 + 6)//7)]" = torch.ops.aten.where.self(ne_10, neg_1, full_default_2);  ne_10 = neg_1 = None
        sum_6: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_81: "f32[]" = torch.ops.aten.add.Tensor(add_68, sum_6);  add_68 = sum_6 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_2: "bf16[((s2 + 6)//7), 201088]" = torch.ops.aten.mm.default(getitem_2, permute_1);  getitem_2 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_10: "f32[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32)
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax_2: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.amax.default(convert_element_type_10, [1], True)
        sub_38: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_2);  convert_element_type_10 = None
        exp_2: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.exp.default(sub_38)
        sum_7: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [1], True);  exp_2 = None
        log_2: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.log.default(sum_7);  sum_7 = None
        sub_39: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(sub_38, log_2);  sub_38 = None
        getitem_30: "i64[((s0 + 6)//7)]" = split_2[2]
        ne_16: "b8[((s0 + 6)//7)]" = torch.ops.aten.ne.Scalar(getitem_30, -100)
        where_4: "i64[((s0 + 6)//7)]" = torch.ops.aten.where.self(ne_16, getitem_30, full_default_1)
        unsqueeze_2: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.unsqueeze.default(where_4, 1);  where_4 = None
        gather_2: "f32[((s0 + 6)//7), 1]" = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_2);  sub_39 = unsqueeze_2 = None
        squeeze_2: "f32[((s0 + 6)//7)]" = torch.ops.aten.squeeze.dim(gather_2, 1);  gather_2 = None
        neg_2: "f32[((s0 + 6)//7)]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        where_5: "f32[((s0 + 6)//7)]" = torch.ops.aten.where.self(ne_16, neg_2, full_default_2);  ne_16 = neg_2 = None
        sum_9: "f32[]" = torch.ops.aten.sum.default(where_5);  where_5 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_94: "f32[]" = torch.ops.aten.add.Tensor(add_81, sum_9);  add_81 = sum_9 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_3: "bf16[((s2 + 6)//7), 201088]" = torch.ops.aten.mm.default(getitem_3, permute_1);  getitem_3 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_14: "f32[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32)
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax_3: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.amax.default(convert_element_type_14, [1], True)
        sub_44: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(convert_element_type_14, amax_3);  convert_element_type_14 = None
        exp_3: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.exp.default(sub_44)
        sum_10: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [1], True);  exp_3 = None
        log_3: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.log.default(sum_10);  sum_10 = None
        sub_45: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(sub_44, log_3);  sub_44 = None
        getitem_38: "i64[((s0 + 6)//7)]" = split_2[3]
        ne_22: "b8[((s0 + 6)//7)]" = torch.ops.aten.ne.Scalar(getitem_38, -100)
        where_6: "i64[((s0 + 6)//7)]" = torch.ops.aten.where.self(ne_22, getitem_38, full_default_1)
        unsqueeze_3: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather_3: "f32[((s0 + 6)//7), 1]" = torch.ops.aten.gather.default(sub_45, 1, unsqueeze_3);  sub_45 = unsqueeze_3 = None
        squeeze_3: "f32[((s0 + 6)//7)]" = torch.ops.aten.squeeze.dim(gather_3, 1);  gather_3 = None
        neg_3: "f32[((s0 + 6)//7)]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        where_7: "f32[((s0 + 6)//7)]" = torch.ops.aten.where.self(ne_22, neg_3, full_default_2);  ne_22 = neg_3 = None
        sum_12: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_107: "f32[]" = torch.ops.aten.add.Tensor(add_94, sum_12);  add_94 = sum_12 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_4: "bf16[((s2 + 6)//7), 201088]" = torch.ops.aten.mm.default(getitem_4, permute_1);  getitem_4 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_18: "f32[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32)
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax_4: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.amax.default(convert_element_type_18, [1], True)
        sub_50: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(convert_element_type_18, amax_4);  convert_element_type_18 = None
        exp_4: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.exp.default(sub_50)
        sum_13: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [1], True);  exp_4 = None
        log_4: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_51: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(sub_50, log_4);  sub_50 = None
        getitem_46: "i64[((s0 + 6)//7)]" = split_2[4]
        ne_28: "b8[((s0 + 6)//7)]" = torch.ops.aten.ne.Scalar(getitem_46, -100)
        where_8: "i64[((s0 + 6)//7)]" = torch.ops.aten.where.self(ne_28, getitem_46, full_default_1)
        unsqueeze_4: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.unsqueeze.default(where_8, 1);  where_8 = None
        gather_4: "f32[((s0 + 6)//7), 1]" = torch.ops.aten.gather.default(sub_51, 1, unsqueeze_4);  sub_51 = unsqueeze_4 = None
        squeeze_4: "f32[((s0 + 6)//7)]" = torch.ops.aten.squeeze.dim(gather_4, 1);  gather_4 = None
        neg_4: "f32[((s0 + 6)//7)]" = torch.ops.aten.neg.default(squeeze_4);  squeeze_4 = None
        where_9: "f32[((s0 + 6)//7)]" = torch.ops.aten.where.self(ne_28, neg_4, full_default_2);  ne_28 = neg_4 = None
        sum_15: "f32[]" = torch.ops.aten.sum.default(where_9);  where_9 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_120: "f32[]" = torch.ops.aten.add.Tensor(add_107, sum_15);  add_107 = sum_15 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_5: "bf16[((s2 + 6)//7), 201088]" = torch.ops.aten.mm.default(getitem_5, permute_1);  getitem_5 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_22: "f32[((s2 + 6)//7), 201088]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32)
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax_5: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.amax.default(convert_element_type_22, [1], True)
        sub_56: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(convert_element_type_22, amax_5);  convert_element_type_22 = None
        exp_5: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.exp.default(sub_56)
        sum_16: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [1], True);  exp_5 = None
        log_5: "f32[((s2 + 6)//7), 1]" = torch.ops.aten.log.default(sum_16);  sum_16 = None
        sub_57: "f32[((s2 + 6)//7), 201088]" = torch.ops.aten.sub.Tensor(sub_56, log_5);  sub_56 = None
        getitem_54: "i64[((s0 + 6)//7)]" = split_2[5]
        ne_34: "b8[((s0 + 6)//7)]" = torch.ops.aten.ne.Scalar(getitem_54, -100)
        where_10: "i64[((s0 + 6)//7)]" = torch.ops.aten.where.self(ne_34, getitem_54, full_default_1)
        unsqueeze_5: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.unsqueeze.default(where_10, 1);  where_10 = None
        gather_5: "f32[((s0 + 6)//7), 1]" = torch.ops.aten.gather.default(sub_57, 1, unsqueeze_5);  sub_57 = unsqueeze_5 = None
        squeeze_5: "f32[((s0 + 6)//7)]" = torch.ops.aten.squeeze.dim(gather_5, 1);  gather_5 = None
        neg_5: "f32[((s0 + 6)//7)]" = torch.ops.aten.neg.default(squeeze_5);  squeeze_5 = None
        where_11: "f32[((s0 + 6)//7)]" = torch.ops.aten.where.self(ne_34, neg_5, full_default_2);  ne_34 = neg_5 = None
        sum_18: "f32[]" = torch.ops.aten.sum.default(where_11);  where_11 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_133: "f32[]" = torch.ops.aten.add.Tensor(add_120, sum_18);  add_120 = sum_18 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_6: "bf16[s2 - 6*(((s2 + 6)//7)), 201088]" = torch.ops.aten.mm.default(getitem_6, permute_1);  getitem_6 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_26: "f32[s2 - 6*(((s2 + 6)//7)), 201088]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32)
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax_6: "f32[s2 - 6*(((s2 + 6)//7)), 1]" = torch.ops.aten.amax.default(convert_element_type_26, [1], True)
        sub_62: "f32[s2 - 6*(((s2 + 6)//7)), 201088]" = torch.ops.aten.sub.Tensor(convert_element_type_26, amax_6);  convert_element_type_26 = None
        exp_6: "f32[s2 - 6*(((s2 + 6)//7)), 201088]" = torch.ops.aten.exp.default(sub_62)
        sum_19: "f32[s2 - 6*(((s2 + 6)//7)), 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
        log_6: "f32[s2 - 6*(((s2 + 6)//7)), 1]" = torch.ops.aten.log.default(sum_19);  sum_19 = None
        sub_63: "f32[s2 - 6*(((s2 + 6)//7)), 201088]" = torch.ops.aten.sub.Tensor(sub_62, log_6);  sub_62 = None
        getitem_62: "i64[s0 - 6*(((s0 + 6)//7))]" = split_2[6];  split_2 = None
        ne_40: "b8[s0 - 6*(((s0 + 6)//7))]" = torch.ops.aten.ne.Scalar(getitem_62, -100)
        where_12: "i64[s0 - 6*(((s0 + 6)//7))]" = torch.ops.aten.where.self(ne_40, getitem_62, full_default_1)
        unsqueeze_6: "i64[s0 - 6*(((s0 + 6)//7)), 1]" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather_6: "f32[s0 - 6*(((s0 + 6)//7)), 1]" = torch.ops.aten.gather.default(sub_63, 1, unsqueeze_6);  sub_63 = unsqueeze_6 = None
        squeeze_6: "f32[s0 - 6*(((s0 + 6)//7))]" = torch.ops.aten.squeeze.dim(gather_6, 1);  gather_6 = None
        neg_6: "f32[s0 - 6*(((s0 + 6)//7))]" = torch.ops.aten.neg.default(squeeze_6);  squeeze_6 = None
        where_13: "f32[s0 - 6*(((s0 + 6)//7))]" = torch.ops.aten.where.self(ne_40, neg_6, full_default_2);  ne_40 = neg_6 = full_default_2 = None
        sum_21: "f32[]" = torch.ops.aten.sum.default(where_13);  where_13 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_146: "f32[]" = torch.ops.aten.add.Tensor(add_133, sum_21);  add_133 = sum_21 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:560 in _unsloth_compiled_fused_ce_loss_function, code: loss = loss / torch.tensor(divisor, dtype = torch.float32)
        convert_element_type_28: "f32[]" = torch.ops.prims.convert_element_type.default(primals_6, torch.float32);  primals_6 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(add_146, convert_element_type_28);  add_146 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        unsqueeze_7: "i64[s0 - 6*(((s0 + 6)//7)), 1]" = torch.ops.aten.unsqueeze.default(getitem_62, 1);  getitem_62 = None
        ne_43: "b8[s0 - 6*(((s0 + 6)//7)), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_7, -100)
        where_14: "i64[s0 - 6*(((s0 + 6)//7)), 1]" = torch.ops.aten.where.self(ne_43, unsqueeze_7, full_default_1);  unsqueeze_7 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        permute_8: "bf16[201088, 2880]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        
         # File: /home/research/cipher-aegis/elias/Securin-FT-Scripts/eliasenv/lib/python3.12/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        unsqueeze_8: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.unsqueeze.default(getitem_54, 1);  getitem_54 = None
        ne_45: "b8[((s0 + 6)//7), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_8, -100)
        where_16: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.where.self(ne_45, unsqueeze_8, full_default_1);  unsqueeze_8 = None
        unsqueeze_9: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.unsqueeze.default(getitem_46, 1);  getitem_46 = None
        ne_47: "b8[((s0 + 6)//7), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_9, -100)
        where_18: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.where.self(ne_47, unsqueeze_9, full_default_1);  unsqueeze_9 = None
        unsqueeze_10: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.unsqueeze.default(getitem_38, 1);  getitem_38 = None
        ne_49: "b8[((s0 + 6)//7), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_10, -100)
        where_20: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.where.self(ne_49, unsqueeze_10, full_default_1);  unsqueeze_10 = None
        unsqueeze_11: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.unsqueeze.default(getitem_30, 1);  getitem_30 = None
        ne_51: "b8[((s0 + 6)//7), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_11, -100)
        where_22: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.where.self(ne_51, unsqueeze_11, full_default_1);  unsqueeze_11 = None
        unsqueeze_12: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.unsqueeze.default(getitem_22, 1);  getitem_22 = None
        ne_53: "b8[((s0 + 6)//7), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_12, -100)
        where_24: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.where.self(ne_53, unsqueeze_12, full_default_1);  unsqueeze_12 = None
        unsqueeze_13: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.unsqueeze.default(getitem_14, 1);  getitem_14 = None
        ne_55: "b8[((s0 + 6)//7), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_13, -100)
        where_26: "i64[((s0 + 6)//7), 1]" = torch.ops.aten.where.self(ne_55, unsqueeze_13, full_default_1);  unsqueeze_13 = full_default_1 = None
        return (div, mm, amax, log, mm_1, amax_1, log_1, mm_2, amax_2, log_2, mm_3, amax_3, log_3, mm_4, amax_4, log_4, mm_5, amax_5, log_5, mm_6, amax_6, log_6, convert_element_type_28, ne_43, where_14, permute_8, ne_45, where_16, ne_47, where_18, ne_49, where_20, ne_51, where_22, ne_53, where_24, ne_55, where_26, primals_4, primals_2, sym_size_int_5)
        