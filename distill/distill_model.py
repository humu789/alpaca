import torch
import torch.nn as nn
from mmrazor.implementations.pruning.sparse_gpt.distill_ops import DistillSparseGptMutator


class SelfDistillAlgorithm(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

        self.mutator = DistillSparseGptMutator()
        self.mutator.prepare_from_supernet(self.model)

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
        input_ids = torch.cat([input_ids, input_ids], dim=0)
        if labels is not None:
            labels = torch.cat([labels, labels], dim=0)
        stu_out = self.model(input_ids, attention_mask, position_ids,
                             past_key_values, inputs_embeds, labels, use_cache,
                             output_attentions, output_hidden_states,
                             return_dict)
        if self.training:
            loss = self.mutator.gather_distill_loss()
            if isinstance(loss, torch.Tensor):
                stu_out['loss'] = loss
        return stu_out

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)