import torch
import torch.nn as nn
from transformers import AutoModel


class BERT_Regressor(nn.Module):

    def __init__(
        self,
        base_model: AutoModel,
        hidden_size: int = 768,
        num_labels:int = 1
    ) -> None:
        
        super(BERT_Regressor, self).__init__()
        self.base_model = base_model
        self.regressor = nn.Linear(hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.regressor(pooled_output)

        return self.sigmoid(logits)