import torch.nn as nn
from transformers import BertModel


class AiModel(nn.Module):
    def __init__(
        self,
        base_model: str = "google-bert/bert-base-chinese",
        num_labels: int = 10,
    ) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(base_model)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return self.linear(output.pooler_output)
