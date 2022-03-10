import math
import numpy as np
import torch
from torch import nn, Tensor
from positional_encoder import PositionalEncoder
 
class HierarchicalModel(nn.Module):
    def __init__(self, base_model: nn.Module, num_labels: int=1, max_parags: int=64, max_parag_length: int=256,
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
        print(f'Initializing hierarchical model with input shape [-1, {max_parags}, {max_parag_length}].')

        # Encoder model forming the base of the hierarchical model
        self.base_model = base_model
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.num_labels = num_labels
        self.hidden_size = base_model.config.hidden_size
        self.max_parags = max_parags
        self.max_parag_length = max_parag_length
        self.hier_layers = hier_layers

        # Init sinusoidal positional embeddings
        self.pos_encoder = PositionalEncoder(self.hidden_size, max_len=max_parag_length)

        # Init segment-wise transformer-based encoder
        self.hier_model = nn.Transformer(d_model=self.hidden_size,
                                          nhead=base_model.config.num_attention_heads,
                                          batch_first=True,
                                          dim_feedforward=base_model.config.intermediate_size,
                                          activation=base_model.config.hidden_act,
                                          dropout=base_model.config.hidden_dropout_prob,
                                          layer_norm_eps=base_model.config.layer_norm_eps,
                                          num_encoder_layers=hier_layers,
                                          num_decoder_layers=0).encoder

        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(self.hidden_size, 
                                               base_model.config.num_attention_heads,
                                               dropout=0.1,
                                               batch_first=True)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        # Classification prediction head
        self.cls_head = nn.Linear(self.hidden_size, num_labels)

    def forward(self, paragraph_attention_mask: Tensor, input_ids: Tensor, attention_mask: Tensor, 
                labels: Tensor, token_type_ids: Tensor=None) -> Tensor:
        """Computes a forward pass through the model.

        Args:
            paragraph_attention_mask (Tensor): attention mask indicating relevant paragph indices.
            input_ids (Tensor): token ids for base model.
            attention_mask (Tensor): attention mask for base model.
            labels (Tensor): labels for the model.
            token_type_ids (Tensor, optional): mask indicating special tokens for base model.

        Returns:
            Tensor: logits of hierarchical model.
        """
        # Hypothetical Example
        # Batch of 10 cases with 64 paragraphs each: (batch_size, n_paragraphs, max_parag_length) --> (10, 64, 128)

        # Combine first two dimensions to feed through base model (batch_size * n_paragraphs, max_parag_length) --> (640, 128)
        device = input_ids.get_device()
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1)).type(torch.LongTensor).to(device)
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1)).type(torch.LongTensor).to(device)
        token_type_ids_reshape = None
        if token_type_ids is not None:
            token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1)).type(torch.LongTensor).to(device)

        # If i.e using BERT as encoder: 768 hidden units
        # Encode sentences with BERT --> (640, 768)
        paragraph_embeddings = self.base_model(input_ids=input_ids_reshape,
                                               attention_mask=attention_mask_reshape,
                                               token_type_ids=token_type_ids_reshape).pooler_output

        # Reshape back to (batch_size, n_paragraphs, output_size) --> (10, 64, 768)
        paragraph_embeddings = paragraph_embeddings.contiguous().view(input_ids.size(0), self.max_parags, self.hidden_size)
        
        # Adding positional encoding for paragraphs
        paragraph_embeddings = self.pos_encoder(paragraph_embeddings)

        # Compute case embeddings --> (10, 64, 768)
        padding_mask = (paragraph_attention_mask == 0)

        # --- Using transformer layers
        if self.hier_layers > 0:
            case_embeddings = self.hier_model(paragraph_embeddings, src_key_padding_mask=padding_mask)

        # --- Using multi-head self attention
        else:
            case_embeddings = self.self_attn(paragraph_embeddings,
                                             paragraph_embeddings,
                                             paragraph_embeddings,
                                             key_padding_mask=padding_mask,
                                             need_weights=False)[0]
        
        # Pool case embeddings (choose first embedding -> CLS token) --> (10, 768) NOTE: Experimented with different pooling strategies
        case_embeddings = case_embeddings[:, 0]

        # Dropout
        case_embeddings = self.dropout(case_embeddings)

        # Linear transform --> (10, num_labels)
        logits = self.cls_head(case_embeddings)
        
        # Compute binary cross-entropy loss
        if self.num_labels == 1:
            labels = torch.unsqueeze(labels, 1)

        loss_fnc = nn.BCEWithLogitsLoss()
        loss = loss_fnc(logits, labels)

        return loss, logits