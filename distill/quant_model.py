from transformers import PreTrainedModel
from torch import nn
# from torch.ao.quantization import QConfig
from typing import Optional
from torch.ao.nn.qat import Linear as QLinear
from torch.ao.quantization import (QConfig, enable_fake_quant, enable_observer,
                                   disable_fake_quant, disable_observer)
from .fake_quants import LearnableFakeQuantize
from .observers import LSQPerChannelObserver
import torch


class HfLlamaWrapper(nn.Module):

    def __init__(self, reference, qconfig: Optional[QConfig] = None):
        super().__init__()
        self.reference = reference

        if qconfig:
            self.qconfig = QConfig(weight=LearnableFakeQuantize.with_args(
                observer=LSQPerChannelObserver.with_args(),
                quant_min=-7,
                quant_max=8,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric),
                                   activation=None)

            self._inplace_qlinear(self.reference, self.qconfig)

    def _inplace_qlinear(self, module, qconfig, skip_module_names=[]):

        def travase(m, qconfig, skip_module_names=[], prefix=''):

            for name, child in m.named_children():

                full_child_name = f'{prefix}.{name}' if len(prefix) else name
                if isinstance(child,
                              nn.Linear) and name not in skip_module_names:
                    child.qconfig = qconfig
                    qlinear = QLinear.from_float(child)
                    setattr(m, name, qlinear)
                    print(f'Convert {full_child_name} to QLinear')
                else:
                    travase(child, qconfig, skip_module_names, full_child_name)

        travase(module, qconfig, skip_module_names)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return self.reference(input_ids, attention_mask, position_ids,
                              past_key_values, inputs_embeds, labels,
                              use_cache, output_attentions,
                              output_hidden_states, return_dict)

    def train(self, training):
        super().train(training)

        if training:
            enable_fake_quant(self)
            enable_observer(self)
        else:
            enable_fake_quant(self)
            disable_observer(self)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.reference, name)