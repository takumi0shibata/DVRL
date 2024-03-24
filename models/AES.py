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
    

class ConcatenateLayer(nn.Module):
    def __init__(self, dim):
        super(ConcatenateLayer, self).__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, self.dim)


class FeatureModel(nn.Module):
    def __init__(self, readability_size, linguistic_size, num_labels=1):
        super(FeatureModel, self).__init__()
        self.linear = nn.Linear(readability_size+linguistic_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, inputs):
        output = self.linear(inputs)
        output = self.sigmoid(output)
        return output