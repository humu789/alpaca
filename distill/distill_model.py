import torch
import torch.nn as nn


class SelfDistillAlgorithm(nn.Module):

    def __init__(self, model: nn.Module, teacher: nn.Module = None) -> None:
        super().__init__()
        self.model = model
        self.teacher = teacher

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
        stu_out = self.model(input_ids, attention_mask, position_ids,
                             past_key_values, inputs_embeds, labels, use_cache,
                             output_attentions, output_hidden_states,
                             return_dict)
        with torch.no_grad():
            teacher_out = self.teacher(input_ids, attention_mask, position_ids,
                                       past_key_values, inputs_embeds, labels,
                                       use_cache, output_attentions,
                                       output_hidden_states, return_dict)
        stu_logits = stu_out['logits']
        tea_logits = teacher_out['logits']
        mse_loss = (stu_logits - tea_logits).pow(2).mean()
        stu_out['loss'] = mse_loss
        return stu_out

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)