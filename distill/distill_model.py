import torch
import torch.nn as nn


class SelfDistillAlgorithm(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

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
        return self.model(input_ids, attention_mask, position_ids,
                          past_key_values, inputs_embeds, labels, use_cache,
                          output_attentions, output_hidden_states, return_dict)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)