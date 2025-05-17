import copy
import random

import torch
from torch.nn import functional as F
from .utils import parent_module, brackets_to_periods, EarlyStopMeter, EditingMeanAct
import transformers
import numpy as np
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from .merge import slerp, GTA, linear
import torch.nn as nn
import gc
from collections import defaultdict

import os.path
import sys
import json
import argparse

sys.path.append('..')
# from easyeditor import (
#     FTHyperParams,
#     GraceHyperParams,
#     MEMITHyperParams,
#     ROMEHyperParams,
#     MENDHyperParams,
#     WISEHyperParams,
#     BaseEditor,
#     summary_metrics,
# )

merge_dict = {
    'slerp': slerp(),
    'ties': GTA('magnitude', 'sum', normalize=True),
    'magnitude_norm': GTA('magnitude', None, normalize=True),
    'magnitude': GTA('magnitude', None, normalize=False),
    'sign': GTA(None, 'sum', normalize=True),
    'dare_ties': GTA('rescaled_random', 'sum'),
    'dare_linear': GTA('random', None),
    'linear': linear()
}

edit_history = []
merge_group_edit_history = []


def euc(query, key, config, act_mask=None, infer=False):
    # Euclidean distance

    act_fn = ACT2FN[config.hidden_act]
    l2_norm = torch.norm(act_fn(key) - act_fn(query), dim=-1)
    if infer and l2_norm.size(1) > 100:
        topk = torch.topk(l2_norm, k=1, largest=True)
        return topk.values.mean()

    if act_mask is not None:
        return torch.sum(l2_norm * act_mask, dim=1) / torch.sum(act_mask, dim=1)
    else:
        return torch.mean(l2_norm, dim=-1)


class WISE(torch.nn.Module):
    def __init__(self, config, model, device):
        super(WISE, self).__init__()
        self.config = config
        self.model = model
        self.config = config
        if hasattr(self.model.config, 'hidden_act'):
            self.config.hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            self.config.hidden_act = self.model.config.activation_function
        # self.tokenizer = model.tokenizer
        layer = config.inner_params[0]
        self.device = device
        self.adapter_layer = None
        self.original_layer = None

        # --- ensure proper formatting (WISE edits weights matrices) ---
        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer

        for n, p in self.model.named_parameters():
            p.requires_grad = False

        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        # --- Add WISE to chosen layers ---
        self.edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        self.layer_name = self.layer.rsplit(".", 1)[-1]
        adapter_layer = getattr(self.edit_module, self.layer_name)

        if type(adapter_layer) is not WISEAdapter:
            setattr(self.edit_module, self.layer_name, WISEAdapter(config, adapter_layer, transpose=transpose))
            self.original_layer = copy.deepcopy(adapter_layer)
            print(f"New weights successfully inserted into {layer}")

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    # Forward
    def __call__(self, **kwargs):
        if not self.config.retrieve:
            if hasattr(self.get_adapter_layer(), 'editing') and not self.get_adapter_layer().editing:
                # final merge
                if not self.get_adapter_layer().original_layer.weight.equal(
                        self.get_adapter_layer().new_weight) and self.get_adapter_layer().editing_total_cnt >= self.config.save_freq:
                    self.get_adapter_layer().memory_weight.append(self.get_adapter_layer().new_weight)
                if len(self.get_adapter_layer().memory_weight) > 0 and self.get_adapter_layer().editing_total_cnt >= self.config.save_freq:
                    print('length of memory is ', len(self.get_adapter_layer().memory_weight), '!!!!!!')
                    self.get_adapter_layer().merge_weight()
        return self.model(**kwargs)

    def reset_layer(self):
        layer = getattr(self.edit_module, self.layer_name)
        del layer
        setattr(self.edit_module, self.layer_name, self.get_adapter_layer().original_layer)

    def get_adapter_layer(self):
        adapter_layer = getattr(self.edit_module, self.layer_name)
        assert type(adapter_layer) is WISEAdapter, print('Adapter Layer is not added correctly....')
        return adapter_layer.to(self.model.device)

    # TODO: generation
    def generate(self, *args, **kwargs):
        setattr(eval(f"self.model.{self.layer}"), "key_id", -1)
        return self.model.generate(*args, **kwargs)

    def edit(self, config, tokens, act_mask=None, deact_mask=None):
        # for retrieve ##
        global edit_history
        global merge_group_edit_history
        edit_history.append([{f"{k1}": v1.to('cpu') for k1, v1 in tokens.items()}, False])
        # for retrieve ##
        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1

        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()
        if getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") % self.config.save_freq == 0:
            self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)

        # --- train Wise value ---
        loss_meter = EarlyStopMeter()
        for i in range(config.n_iter):

            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.SGD([self.get_adapter_layer().new_weight], config.edit_lr, weight_decay=1e-5)

            ft_loss = self._cal_ft_loss(tokens, last_prompt_token_loc)

            act_loss = self._cal_activation_loss(self.get_adapter_layer().original_layer_output,
                                                 self.get_adapter_layer().new_weight_layer_output,
                                                 config=config, act_mask=act_mask, deact_mask=deact_mask)
            loss = ft_loss + act_loss.to(ft_loss.device)

            if loss_meter.stop():
                self.get_adapter_layer().save_editing_activation()  # add last gradient
                break
            if i == config.n_iter - 1:
                self.get_adapter_layer().save_editing_activation()  # add last gradient

            if self.config.retrieve and self.get_adapter_layer().merge_cnt > 0 and self.config.replay:
                memory_loss = []
                for _ in merge_group_edit_history:
                    idx = 0
                    while True:
                        memo_input, is_used = _[idx]
                        if not is_used:
                            _[idx][1] = True
                            break
                        idx += 1
                        if idx == len(_):  ## re Assign
                            for m in range(len(_)):
                                _[m][1] = False
                            idx = 0

                    memo_input = {f"{k1}": v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    memory_act_loss = self._cal_memory_neg_activation_loss(
                        self.get_adapter_layer().original_layer_output,
                        self.get_adapter_layer().new_weight_layer_output, config=config,
                        act_mask=act_mask, deact_mask=deact_mask)
                    memory_loss.append(memory_act_loss.to(ft_loss.device))
                    del memo_input
                neg_memo_loss = torch.stack(memory_loss).mean()
                loss += neg_memo_loss
                if len(edit_history) > 0:
                    memo_input = random.choice(edit_history)[0]
                    memo_input = {f"{k1}": v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    pos_memo_loss = self._cal_memory_pos_activation_loss(self.get_adapter_layer().original_layer_output,
                                                                         self.get_adapter_layer().new_weight_layer_output,
                                                                         config=config,
                                                                         act_mask=act_mask, deact_mask=deact_mask)
                    del memo_input
                    loss += pos_memo_loss.to(ft_loss.device)
            # for replay Appendix B.3

            optimizer.zero_grad()

            loss.backward()
            self.get_adapter_layer().mask_new_weight_gradient()

            if self.config.retrieve and self.get_adapter_layer().merge_cnt > 0 and self.config.replay:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)} + {np.round(neg_memo_loss.item(), 3)} + {np.round(pos_memo_loss.item(), 3)}"
                )
            else:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)}"
                )

            optimizer.step()
            loss_meter.update(loss.item())

            if type(self.config.norm_constraint) is float:
                self._norm_constraint(self.config.norm_constraint)

        # --- pull out info we want to log from the Wise layer ---
        setattr(eval(f"self.model.{self.layer}"), "editing", False)
        setattr(eval(f"self.model.{self.layer}"), "training", False)

        editing_total_cnt = getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") + 1
        setattr(eval(f"self.model.{self.layer}"), "editing_total_cnt", editing_total_cnt)
        #
        if self.config.save_freq is not None and editing_total_cnt % self.config.save_freq == 0:
            self.get_adapter_layer().save_weight()
            print(f'Add New Weight to Memory...')
        if editing_total_cnt % self.config.merge_freq == 0:
            # for retrieve ##
            merge_group_edit_history.append(edit_history)
            edit_history = []
            # for retrieve ##

            self.get_adapter_layer().merge_weight()
            print(f'Merge Weight of (New, Original) Matrix... with {self.config.merge_alg}')

    def _norm_constraint(self, norm_constraint):
        new_weight = self.get_adapter_layer().new_weight
        original_weight = self.get_adapter_layer().weight
        with torch.no_grad():
            new_weight[...] = torch.clamp(
                new_weight, min=original_weight - norm_constraint, max=original_weight + norm_constraint
            )

    def _cal_ft_loss(self, tokens, last_prompt_token_loc):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        bs = tokens["input_ids"].shape[0] - k
        logits = self.model(**tokens).logits
        shift_logits = logits[:-k, :-1, :].contiguous()
        shift_labels = tokens['labels'][:-k, 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(bs, -1)

        label_mask = torch.zeros_like(loss, dtype=torch.bool)

        for i, col_index in enumerate(last_prompt_token_loc[:-k]):
            label_mask[i, col_index - 1:] = True

        ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()
        return ft_loss

    def _cal_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                             deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        total_loss = []
        len_temp = original_layer_output.shape[0] / k - 1
        for i, act_mk in enumerate(act_mask):
            if act_mk is not None:
                in_scope_dist = euc(original_layer_output[int(i * len_temp):int((i + 1) * len_temp), ...],
                                    new_weight_layer_output[int(i * len_temp):int((i + 1) * len_temp), ...], config,
                                    act_mask=act_mk)
                out_scope_dist = euc(original_layer_output[int(i * len_temp):int((i + 1) * len_temp), ...],
                                     new_weight_layer_output[int(i * len_temp):int((i + 1) * len_temp), ...], config,
                                     act_mask=deact_mask[i])
            else:
                in_scope_dist = euc(original_layer_output[int(i * len_temp):int((i + 1) * len_temp), ...],
                                    new_weight_layer_output[int(i * len_temp):int((i + 1) * len_temp), ...], config)
                if (i == k - 1):
                    out_scope_dist = euc(original_layer_output[int(i - k):, ...],
                                         new_weight_layer_output[int(i - k):, ...], config)
                else:
                    out_scope_dist = euc(original_layer_output[int(i - k):int(i + 1 - k), ...],
                                         new_weight_layer_output[int(i - k):int(i + 1 - k), ...], config)

            loss = out_scope_dist.view(-1, 1) - in_scope_dist + config.gamma
            loss2 = out_scope_dist - config.alpha
            loss3 = config.beta - in_scope_dist
            loss3 = torch.mean(loss3[loss3 > 0]) if min(loss3[loss3 > 0].size()) > 0 else torch.tensor(0.).to(
                original_layer_output.device)
            loss2 = torch.mean(loss2[loss2 > 0]) if min(loss2[loss2 > 0].size()) > 0 else torch.tensor(0.).to(
                original_layer_output.device)
            loss = torch.mean(loss[loss > 0]) if min(loss[loss > 0].size()) > 0 else torch.tensor(0.).to(
                original_layer_output.device)
            total_loss.append(loss + loss2 + loss3)
        return sum(total_loss) / len(total_loss)

    def _cal_memory_pos_activation_loss(self, original_layer_output, new_weight_layer_output, config=None,
                                        act_mask=None,
                                        deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = 20 - in_scope_dist

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def _cal_memory_neg_activation_loss(self, original_layer_output, new_weight_layer_output, config=None,
                                        act_mask=None,
                                        deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = in_scope_dist - 5

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def save(self, save_path):
        import os
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist

        # Save additional information, such as memory_weight, memory_mean_act, etc.
        additional_info = {
            'memory_weight': self.get_adapter_layer().memory_weight,
            'memory_mean_act': self.get_adapter_layer().memory_mean_act,
            'merge_cnt': self.get_adapter_layer().merge_cnt,
            'editing_mean_act': self.get_adapter_layer().editing_mean_act,
            'editing_total_cnt': self.get_adapter_layer().editing_total_cnt,
            'weight_mask': self.get_adapter_layer().weight_mask,
            # Add other variables that need to be saved
        }
        if hasattr(self.get_adapter_layer(), 'key_id') and self.get_adapter_layer().key_id is not None:
            additional_info['key_id'] = self.get_adapter_layer().key_id
        # Save all information to the file
        torch.save({
            'adapter_state_dict': self.get_adapter_layer().state_dict(),
            'config': self.config,
            'additional_info': additional_info,
            'edit_history': edit_history,
            'merge_group_edit_history': merge_group_edit_history
        }, save_path)

    def load(self, load_path):
        import os
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

        # Load all previously saved information
        saved_data = torch.load(load_path)
        if hasattr(self.model.config, 'hidden_act'):
            saved_data['config'].hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            saved_data['config'].hidden_act = self.model.config.activation_function
        if saved_data['config'] != self.config:
            print("Warning: The loaded WISE config is different from the original config")

        # Restore the state dictionary of the WISE Adapter instance
        self.get_adapter_layer().load_state_dict(saved_data['adapter_state_dict'])
        # Restore additional information
        adapter_layer = self.get_adapter_layer()
        for key, value in saved_data['additional_info'].items():
            setattr(adapter_layer, key, value)

        # Restore editing history
        global edit_history, merge_group_edit_history
        edit_history = saved_data['edit_history']
        merge_group_edit_history = saved_data['merge_group_edit_history']
        print(f"Model configuration and WISE state loaded from {load_path}")


class WISEAdapter(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super(WISEAdapter, self).__init__()

        self.layer = layer
        self.weight = self.layer.weight
        self.device = layer.weight.device
        self.config = config
        self.new_weight = copy.deepcopy(self.weight)
        self.original_layer = copy.deepcopy(self.layer)
        self.memory_weight = []
        self.memory_mean_act = []
        if 'gpt2' in self.config.model_name:
            self.bias = self.layer.bias  # For Conv1D
        else:
            self.bias = None
        self.merge_cnt = 0  # only for retrieve
        assert not self.weight.requires_grad, print('Original Layer can not be tunable....')

        self.used_mask = None

        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        self.training = False
        self.editing = False

        self.editing_mean_act = EditingMeanAct()
        self.editing_total_cnt = 0
        self.selected_memory_index = -1  ###new code

    def set_parameter_tunable(self):
        self.new_weight.requires_grad = True

    def save_weight(self):
        self.memory_weight.append(copy.deepcopy(self.new_weight))
        self.new_weight = copy.deepcopy(self.original_layer.weight)
        if self.config.retrieve:
            self.memory_mean_act.append(copy.deepcopy(self.editing_mean_act))
            self.editing_mean_act = EditingMeanAct()

    def merge_weight(self):
        if self.config.save_freq is not None:  # for ties dare dare_ties
            if not self.config.retrieve:
                merge_alg = merge_dict[self.config.merge_alg]
                if self.original_layer.weight.equal(self.layer.weight):
                    cur_new_weight = merge_alg.execute(
                        [self.config.weights / len(self.memory_weight) for _ in range(len(self.memory_weight))],
                        self.original_layer.weight, self.memory_weight, densities=self.config.densities)
                else:
                    cur_new_weight = merge_alg.execute(
                        [0.4 / len(self.memory_weight) for _ in range(len(self.memory_weight))] + [0.6],
                        self.original_layer.weight, self.memory_weight + [self.layer.weight],
                        densities=self.config.densities)
                self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                del self.memory_weight
                self.memory_weight = []
            else:
                merge_alg = merge_dict[self.config.merge_alg]
                merge_num = self.config.merge_freq // self.config.save_freq
                assert len(self.memory_weight) >= merge_num
                new_merge_weight = merge_alg.execute([self.config.weights / merge_num for _ in range(merge_num)],
                                                     self.original_layer.weight, self.memory_weight[-merge_num:],
                                                     densities=self.config.densities)
                min_a = 1e9
                for _ in range(merge_num):
                    self.memory_weight.pop()
                    edit_act = self.memory_mean_act.pop()
                    min_a = min(min_a, edit_act.min_act())
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                self.memory_weight.append(new_merge_weight)
                self.memory_mean_act.append(EditingMeanAct(min_a=min_a))
                print(len(self.memory_weight))
                assert len(self.memory_mean_act) == len(self.memory_weight)
                self.merge_cnt += 1
        else:
            merge_alg = merge_dict[self.config.merge_alg]
            cur_new_weight = merge_alg.execute(0.5, self.layer.weight, [self.new_weight],
                                               densities=self.config.densities)
            self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
            self.new_weight = copy.deepcopy(self.original_layer.weight)

    def save_editing_activation(self):
        in_scope_dist = euc(self.original_layer_output[:-1, ...], self.new_weight_layer_output[:-1, ...], self.config)
        self.editing_mean_act.update(in_scope_dist.mean().item())

    def generate_activation_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        p_mask = np.random.choice([1, 0], size=p_grad.size()[0], p=[mask_ratio, 1 - mask_ratio])
        p_mask = torch.from_numpy(p_mask).to(p_grad.device)
        self.weight_mask = p_mask

    def generate_non_overlapping_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        mask_size = int(mask_ratio * p_grad.size()[0])
        if self.used_mask is None:
            self.used_mask = np.zeros(p_grad.size()[0], dtype=bool)
        available_indices = np.where(~self.used_mask)[0]  # 获取未被遮罩的元素索引
        if len(available_indices) < mask_size:
            raise ValueError("Not enough unused elements to generate a new mask.")
        chosen_indices = np.random.choice(available_indices, size=mask_size, replace=False)
        mask_array = np.zeros(p_grad.size()[0], dtype=int)
        mask_array[chosen_indices] = 1
        self.used_mask[chosen_indices] = True  # 更新遮罩状态
        self.weight_mask = torch.from_numpy(mask_array).to(p_grad.device)

    def new_weight_forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.new_weight) if self.bias is None else torch.addmm(self.bias,
                                                                                      input.view(-1, input.size(-1)),
                                                                                      self.new_weight).view(
            input.size()[:-1] + (self.layer.nf,))

    def mask_new_weight_gradient(self):
        assert self.new_weight.grad is not None, print('Gradient Collection for New Weight error, gradient not found')
        # Add gradient mask after the loss updates
        p_size = self.new_weight.grad.size()
        p_grad = self.new_weight.grad.reshape(-1)

        # mask = torch.from_numpy(np.random.choice([0, 1], size=p_grad.size()[0], p=[.1, .9])).cuda()
        p_grad = p_grad * self.weight_mask
        self.new_weight.grad = p_grad.view(p_size).to(self.new_weight.grad.dtype)

    def forward(self, *args):
        selected_memory = -1  # 初始化为使用new_weight或原始权重  new code
        if self.editing:
            layer_out = self.new_weight_forward(*args)
            self.new_weight_layer_output = layer_out
            self.original_layer_output = self.original_layer(*args)
        else:
            if not self.config.retrieve:
                original_layer_output = self.original_layer(*args)
                layer_output = self.layer(*args)
                new_weight_layer_output = self.new_weight_forward(*args)
                dist2 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True)
                dist1 = euc(original_layer_output, layer_output, self.config, infer=True)
                threshold = self.editing_mean_act.min_act() * self.config.act_ratio

                if dist1.item() < threshold and dist2.item() < threshold:
                    layer_out = original_layer_output
                elif dist1.item() > dist2.item():
                    layer_out = layer_output
                else:
                    layer_out = new_weight_layer_output
            else:
                original_layer_output = self.original_layer(*args)
                new_weight_layer_output = self.new_weight_forward(*args)
                dist1 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True)
                threshold = self.editing_mean_act.min_act() * self.config.act_ratio
                min_dist = dist1
                if min_dist.dim() > 0:
                    min_dist = min_dist.mean()
                if min_dist.item() < threshold:
                    layer_out = original_layer_output
                    selected_memory = -1  ##new code
                else:
                    layer_out = new_weight_layer_output
                    selected_memory = -1  ##new code
                for i in range(len(self.memory_weight)):
                    memory_retrieve_weight = self.memory_weight[i]
                    memory_weight_layer_output = F.linear(*args, memory_retrieve_weight)
                    dist = euc(original_layer_output, memory_weight_layer_output, self.config, infer=True)
                    if dist > min_dist and dist > self.memory_mean_act[i].min_act() * self.config.act_ratio:
                        layer_out = memory_weight_layer_output
                        min_dist = dist
                        selected_memory = i  ##new code
                self.selected_memory_index = selected_memory  # new code
        return layer_out


class WISEMultimodal(WISE):
    def edit(self, config, multimodal_inputs, text_tokens, ans_token_len, act_mask=None, deact_mask=None):
        global edit_history
        global merge_group_edit_history
        edit_history.append([{f"{k1}": v1.to('cpu') for k1, v1 in text_tokens.items()}, False])
        last_prompt_token_loc = (text_tokens["labels"] == -100).sum(dim=-1) - 1

        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()
        if getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") % self.config.save_freq == 0:
            self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)

            # --- train Wise value ---
        loss_meter = EarlyStopMeter()
        for i in range(config.n_iter):
            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.SGD([super().get_adapter_layer().new_weight], config.edit_lr, weight_decay=1e-5)

            ft_loss = self._cal_ft_loss(multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len)

            act_loss = super()._cal_activation_loss(super().get_adapter_layer().original_layer_output,
                                                    super().get_adapter_layer().new_weight_layer_output,
                                                    config=config, act_mask=act_mask, deact_mask=deact_mask)
            loss = ft_loss + act_loss.to(ft_loss.device)

            if loss_meter.stop():
                super().get_adapter_layer().save_editing_activation()  # add last gradient
                break
            if i == config.n_iter - 1:
                super().get_adapter_layer().save_editing_activation()  # add last gradient

            if self.config.retrieve and super().get_adapter_layer().merge_cnt > 0 and self.config.replay:
                memory_loss = []
                for _ in merge_group_edit_history:
                    idx = 0
                    while True:
                        memo_input, is_used = _[idx]
                        if not is_used:
                            _[idx][1] = True
                            break
                        idx += 1
                        if idx == len(_):  ## re Assign
                            for m in range(len(_)):
                                _[m][1] = False
                            idx = 0

                    memo_input = {f"{k1}": v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    memory_act_loss = super()._cal_memory_neg_activation_loss(
                        super().get_adapter_layer().original_layer_output,
                        super().get_adapter_layer().new_weight_layer_output, config=config,
                        act_mask=act_mask, deact_mask=deact_mask)
                    memory_loss.append(memory_act_loss.to(ft_loss.device))
                    del memo_input
                neg_memo_loss = torch.stack(memory_loss).mean()
                loss += neg_memo_loss
                if len(edit_history) > 0:
                    memo_input = random.choice(edit_history)[0]
                    memo_input = {f"{k1}": v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    pos_memo_loss = super()._cal_memory_pos_activation_loss(
                        super().get_adapter_layer().original_layer_output,
                        super().get_adapter_layer().new_weight_layer_output, config=config,
                        act_mask=act_mask, deact_mask=deact_mask)
                    del memo_input
                    loss += pos_memo_loss.to(ft_loss.device)
            # for replay Appendix B.3

            optimizer.zero_grad()

            loss.backward()
            super().get_adapter_layer().mask_new_weight_gradient()

            if self.config.retrieve and super().get_adapter_layer().merge_cnt > 0 and self.config.replay:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)} + {np.round(neg_memo_loss.item(), 3)} + {np.round(pos_memo_loss.item(), 3)}"
                )
            else:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)}"
                )

            optimizer.step()
            loss_meter.update(loss.item())

            if type(self.config.norm_constraint) is float:
                super()._norm_constraint(self.config.norm_constraint)

        # --- pull out info we want to log from the Wise layer ---
        setattr(eval(f"self.model.{self.layer}"), "editing", False)
        setattr(eval(f"self.model.{self.layer}"), "training", False)

        editing_total_cnt = getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") + 1
        setattr(eval(f"self.model.{self.layer}"), "editing_total_cnt", editing_total_cnt)
        if self.config.save_freq is not None and editing_total_cnt % self.config.save_freq == 0:
            super().get_adapter_layer().save_weight()
            print(f'Add New Weight to Memory...')
        if editing_total_cnt % self.config.merge_freq == 0:
            # for retrieve ##
            merge_group_edit_history.append(edit_history)
            edit_history = []
            # for retrieve ##

            super().get_adapter_layer().merge_weight()
            print(f'Merge Weight of (New, Original) Matrix... with {self.config.merge_alg}')

    def _cal_ft_loss(self, multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1

        if k != 1:
            raise AssertionError("Not support Batch Edit")

        bs = text_tokens["input_ids"].shape[0] - k
        logits = self.model(**multimodal_inputs).logits
        shift_logits = logits[:-k, :-1, :].contiguous()
        shift_labels = multimodal_inputs['input_ids'][:-k, 1:].contiguous()
        # only cal loss of target text tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        a = shift_logits.view(-1, shift_logits.size(-1))
        b = shift_labels.view(-1)[-ans_token_len:]
        a = a[-b.size(0):, :]
        loss = loss_fct(a, b)
        loss = loss.view(bs, -1)
        label_mask = torch.ones_like(loss, dtype=torch.bool)
        ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()
        return ft_loss


#####new code
class WISE(WISE):

    def batch_edit(self, dataset, group_size=100, save_freq=10, merge_freq=20):
        """
        按组批量执行编辑操作，每组处理group_size个样本
        参数:
            dataset: 可迭代的数据集，每个元素为单个样本的tokens
            group_size: 每组包含的样本数 (默认100)
            save_freq: 权重保存频率 (默认10个样本保存一次)
            merge_freq: 权重合并频率 (默认20个样本合并一次)
        """
        # 保存原始配置
        original_save = self.config.save_freq
        original_merge = self.config.merge_freq

        # 应用新配置
        self.config.save_freq = save_freq
        self.config.merge_freq = merge_freq

        # 全局变量备份
        global edit_history, merge_group_edit_history
        original_history = edit_history.copy()
        original_merge_group = merge_group_edit_history.copy()

        all_group_results = []

        # 按组处理数据
        for group_idx in range(0, len(dataset), group_size):
            group = dataset[group_idx:group_idx + group_size]

            # 每组初始化状态
            adapter = self.get_adapter_layer()
            adapter.memory_weight = []
            adapter.memory_mean_act = []
            adapter.merge_cnt = 0
            adapter.editing_total_cnt = 0
            edit_history = []
            merge_group_edit_history = []

            # 处理单个组
            for case in group:
                self.edit(config=self.config, tokens=case)

            # 验证内存合并结果
            print(f"组 {group_idx // group_size} 完成，生成 {len(adapter.memory_weight)} 个记忆权重")
            print("=" * 50 + "\n")

            # 恢复原始状态
            edit_history = original_history
            merge_group_edit_history = original_merge_group
            self.config.save_freq = original_save
            self.config.merge_freq = original_merge
            # 处理完group后，进行推理
            group_results = self.evaluate_group(group)
            all_group_results.append(group_results)
        # ...恢复状态...
        return all_group_results

    def get_current_memory_index(self):
        return self.get_adapter_layer().selected_memory_index

    def evaluate_group(self, group_data):
        error_records = defaultdict(int)
        case_errors = []

        for case_idx, case in enumerate(group_data):
            # 假设每个case是一个字典，包含输入和标签
            inputs = self.prepare_inputs(case)  # 需要根据实际数据结构调整
            label = case["label"]

            # 推理
            with torch.no_grad():
                output = self.model(**inputs)

            # 获取预测结果和memory索引
            pred = output.logits.argmax(dim=-1)
            correct = pred.eq(label).all().item()
            memory_index = self.get_adapter_layer().selected_memory_index

            if not correct:
                case_errors.append((case_idx, memory_index))
                error_records[memory_index] += 1

        # 统计错误最多的memory_index
        max_error = max(error_records.values(), default=0)
        worst_memories = [k for k, v in error_records.items() if v == max_error]

        return {
            "case_errors": case_errors,
            "error_summary": dict(error_records),
            "worst_memories": worst_memories
        }

    def prepare_inputs(self, case):
        # 根据实际情况将case转换为模型输入格式
        # 例如，转换为tokenized输入
        tokens = {
            "input_ids": case["input_ids"].to(self.device),
            "attention_mask": case["attention_mask"].to(self.device),
            "labels": case["labels"].to(self.device)
        }
        return tokens
# 使用示例
# wise = WISE(config, model, device)
# wise.batch_edit(dataset, group_size=100, save_freq=10, merge_freq=20)
def slice_list(matrix,start_indices,left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[start_indices[0]-1:-1]
        else:
            return matrix[start_indices[0]:]

def test_prediction_acc(model, tok, hparams, prompts, targets, device=0, locality=False, vanilla_generation=False):
    """
    参数说明
参数	类型	说明
model	模型对象	待评估的模型（如GPT-2、T5等）。
tok	AutoTokenizer	分词器，用于文本编码。
hparams	HyperParams	超参数配置（如最大生成长度）。
prompts	str/list	输入的提示文本（单条或多条）。
targets	str/list	期望模型生成的目标文本。
device	str	计算设备（如CPU/GPU）。
locality	bool	是否返回完整生成结果（默认False，返回准确率）。
vanilla_generation	bool	是否使用原始生成模式（默认False，使用编辑后模式）
    """
    if vanilla_generation:
        if isinstance(prompts, str):
            prompts, targets = [prompts, ], [targets, ]
        results = []
        for prompt, target_new in zip(prompts, targets):
            target_new_tokens = tok.encode(target_new, add_special_tokens=False)
            prompt_tok = tok(
                prompt,
                return_tensors="pt",
            ).to(f"cuda:{device}")
            gen_token = model.generate(
                input_ids=prompt_tok['input_ids'],
                attention_mask=prompt_tok['attention_mask'],
                max_new_tokens=len(target_new_tokens),
                pad_token_id=tok.eos_token_id,
                do_sample=False,
                use_cache=False,
            )
            if locality:
                results.append(gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):])
            else:
                results.append(np.mean(np.equal(target_new_tokens, gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):])))
        return results

    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    if not locality and hasattr(hparams, 'use_chat_template') and hparams.use_chat_template:
        prompts = [[{"role":"user", "content":m}] for m in prompts]
        prompts=tok.apply_chat_template(prompts,
                                        add_generation_prompt=True,
                                        tokenize=False)
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    before_padding_side = tok.padding_side
    tok.padding_side = 'left'
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=max(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    ).to(f"cuda:{device}")
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    )
    tok.padding_side = before_padding_side
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(answers,prompt_len,left=True)
        labels = slice_list(labels,prompt_len,left=False)
        if locality:
            return answers if type(answers[0]) is list else [answers,]
        if isinstance(answers[0], list):
            res = []
            for ans,label in zip(answers,labels):
                temp_acc = np.mean(np.equal(ans, label))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res
        else:
            return [np.mean(np.equal(answers, labels))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=False, type=str,default='WISE')
    parser.add_argument('--hparams_dir', required=False, type=str,default='../hparams/WISE/llama-3-8b.yaml')
    parser.add_argument('--data_dir', required=False, type=str,default='../data/wise')
    parser.add_argument('--data_type', required=False, type=str,
                        choices=['ZsRE', 'temporal', 'hallucination'],default='ZsRE')
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--ds_size', default=100, type=int)
    parser.add_argument('--sequential_edit', action="store_true",default=True)

    args = parser.parse_args()

    editing_hparams = WISEHyperParams
    K = 100


    edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))[:K]
    loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
    loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

    prompts = [edit_data_['src'] for edit_data_ in edit_data]
    subject = [edit_data_['subject'] for edit_data_ in edit_data]
    rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
    target_new = [edit_data_['alt'] for edit_data_ in edit_data]
    locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
    locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
    locality_inputs = {
        'neighborhood':{
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }


    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}.json'
        )

    print("See results at: ", output_file)

    eval_metric = {
        'ZsRE': 'token em',
        'hallucination': 'ppl',
        'temporal': 'ood_ppl'
    }

    #%wise

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        loc_prompts=loc_prompts,
        subject=subject,
        locality_inputs=locality_inputs,
        sequential_edit=args.sequential_edit,
        eval_metric=eval_metric[args.data_type]
    )

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    if len(metrics) > 0:
        summary_metrics(metrics)
#                self.model = AutoModelForCausalLM.from_pretrained('D:/EasyEdit-main/Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct', **model_kwargs)
#                 self.tok = AutoTokenizer.from_pretrained('D:/EasyEdit-main/Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct')