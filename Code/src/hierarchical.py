import math
import numpy as np
import torch
from torch import nn, Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, dropout_p: float = 0.1, max_len: int=1024):
        """Initializes the positional embedding layer to enrich data fed into transformers
           with positional information.

        Args:
            dim_model (int): model dimension
            dropout_p (float, optional): _description_. Defaults to 0.1.
            max_len (int, optional): determines how far the position can influence other tokens. Defaults to 128.

        Note:
            This code is a modified version of: `<https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`_.
        """
        super().__init__()

        # Dropout
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pos_encoding',pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        """Generates positional embeddings.

        Args:
            token_embedding (torch.tensor): original embeddings

        Returns:
            torch.tensor: transformed embeddings
        """
        # Residual connection + positional encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
 
 class HierarchicalModel(nn.Module):
    def __init__(self, base_model: nn.Module, num_labels: int=2, max_parags: int=64, max_parag_length: int=128,
                 hier_layers: int=2, freeze_base: bool=False):
        """A hierarchical model using a transformer-based base model as a base.

        Args:
            base_model (nn.Module): tranformer-based base model.
            num_labels (int): number of output classes. Defaults to 2.
            max_parags (int, optional): maximum number of paragraphs. Defaults to 64.
            max_segment_length (int, optional): maximum length of paragraph. Defaults to 128.
            hier_layers (int, optional): number of hierarchical transformer layers. Defaults to 2.
            freeze_base (int, optional): whether base model parameters should be freezed. Defaults to False.
        """
        super(HierarchicalModel, self).__init__()
        # Sentence encoder model forming the base of ther hierarchical model
        self.base_model = base_model
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.num_labels = num_labels
        self.hidden_size = base_model.config.hidden_size
        self.max_parags = max_parags
        self.max_parag_length = max_parag_length

        # Init sinusoidal positional embeddings
        self.pos_encoder = PositionalEncoding(base_model.config.hidden_size, max_len=max_parag_length)

        # Init segment-wise transformer-based encoder
        self.hier_model = nn.Transformer(d_model=base_model.config.hidden_size,
                                          nhead=base_model.config.num_attention_heads,
                                          batch_first=True,
                                          dim_feedforward=base_model.config.intermediate_size,
                                          activation=base_model.config.hidden_act,
                                          dropout=base_model.config.hidden_dropout_prob,
                                          layer_norm_eps=base_model.config.layer_norm_eps,
                                          num_encoder_layers=hier_layers,
                                          num_decoder_layers=0).encoder

        # Classification prediction head
        self.cls_head = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, paragraph_attention_mask=None, **kwargs):

        # Hypothetical Example
        # Batch of 10 paragraphs: (batch_size, n_paragraphs, max_parag_length) --> (10, 64, 128)
        # If i.e using BERT as encoder: 768 hidden units

        # Combine first two dimensions to feed through base model (batch_size * n_paragraphs, max_parag_length) --> (640, 128)
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))
        token_type_ids_reshape = None
        if token_type_ids is not None:
            token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1))

        # Encode sentences with BERT --> (640, 768)
        paragraph_embeddings = self.encoder(input_ids=input_ids_reshape,
                                            attention_mask=attention_mask_reshape,
                                            token_type_ids=token_type_ids_reshape).pooler_output

        # Reshape back to (batch_size, n_paragraphs, output_size) --> (10, 64, 768)
        paragraph_embeddings = paragraph_embeddings.contiguous().view(input_ids.size(0), self.max_parags, self.hidden_size)

        # ADD MASKING FOR PARAGRAPHS
        
        # Adding positional encoding for paragraphs
        paragraph_embeddings = self.pos_encoder(paragraph_embeddings)

        # Compute case embeddings --> (10, 768)
        case_embeddings = self.hier_model(paragraph_embeddings)

        # Linear transform --> (10, num_labels)
        logits = self.cls_head(case_embeddings)
        return logits