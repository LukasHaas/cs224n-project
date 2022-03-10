import torch
from torch import nn, Tensor
from positional_encoder import PositionalEncoder
from transformer_encoder import CustomTransformerEncoder, CustomTransformerEncoderLayer

class aLEXa(nn.Module):
    def __init__(self, base_model: nn.Module, num_labels: int=1, max_parags: int=64, max_parag_length: int=128,
                 hier_layers: int=2, learn_loss_weights: bool=True, freeze_base: bool=False):
        """A hierarchical model using a transformer-based base model as a base.

        Args:
            base_model (nn.Module): tranformer-based base model.
            num_labels (int): number of output classes. Defaults to 2.
            max_parags (int, optional): maximum number of paragraphs. Defaults to 64.
            max_segment_length (int, optional): maximum length of paragraph. Defaults to 128.
            hier_layers (int, optional): number of hierarchical transformer layers. Defaults to 2.
            learn_loss_weights (bool, optional): whether the loss weights should be learned. Defaults to True.
            freeze_base (int, optional): whether base model parameters should be freezed. Defaults to False.
        """
        super(aLEXa, self).__init__()
        # Sentence encoder model forming the base of ther hierarchical model
        self.base_model = base_model
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.num_labels = num_labels
        self.hidden_size = base_model.config.hidden_size
        self.max_parags = max_parags
        self.max_parag_length = max_parag_length
        self.hier_layers = hier_layers
        self.learn_loss_weights = learn_loss_weights

        # Init sinusoidal positional embeddings
        self.pos_encoder = PositionalEncoder(base_model.config.hidden_size, max_len=max_parag_length)

        # Init segment-wise transformer-based encoder
        transformer_encoder_layer = CustomTransformerEncoderLayer(d_model=base_model.config.hidden_size,
                                                                  nhead=base_model.config.num_attention_heads,
                                                                  dim_feedforward=base_model.config.intermediate_size,
                                                                  dropout=base_model.config.hidden_dropout_prob,
                                                                  activation=base_model.config.hidden_act,
                                                                  layer_norm_eps=base_model.config.layer_norm_eps,
                                                                  batch_first=True)
        self.hier_model = CustomTransformerEncoder(encoder_layer=transformer_encoder_layer,
                                                   num_layers=hier_layers)

        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(base_model.config.hidden_size, 
                                               base_model.config.num_attention_heads,
                                               dropout=0.1,
                                               batch_first=True)

        # Attention forcing head
        self.attn_head = nn.Linear(1, 1)
        
        # Learn multi-task loss weighting
        if learn_loss_weights:
            self.class_weight = nn.Parameter(torch.rand(1))
            self.class_weight.requires_grad = True
            self.attn_forcing_weight = nn.Parameter(torch.rand(1))
            self.attn_forcing_weight.requires_grad = True

        # Dropout
        self.dropout = nn.Dropout(0.1)

        # Classification prediction head
        self.cls_head = nn.Linear(base_model.config.hidden_size, num_labels)

    def compute_classification_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Computes classification binary cross-entropy loss.

        Args:
            logits (Tensor): pre-sigmoid model output.
            labels (Tensor): labels.

        Returns:
            Tensor: loss.
        """
        if self.num_labels == 1:
            labels = torch.unsqueeze(labels, 1)

        loss_fnc = nn.BCEWithLogitsLoss()
        loss = loss_fnc(logits, labels)
        return loss

    def compute_attention_forcing_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Computes attention forcing binary cross-entropy loss.

        Args:
            logits (Tensor): pre-sigmoid model output.
            labels (Tensor): labels.

        Returns:
            Tensor: loss.
        """
        loss_fnc = nn.BCEWithLogitsLoss()
        loss = loss_fnc(logits, labels)
        return loss

    def compute_total_loss(self, class_loss: Tensor, attn_loss: Tensor) -> Tensor:
        """Computes attention forcing binary cross-entropy loss.

        Args:
            logits (Tensor): pre-sigmoid model output.
            class_loss (Tensor): classification loss.
            attn_loss (Tensor): attention forcing loss.

        Returns:
            Tensor: loss.
        """
        if self.learn_loss_weights:
            weighted_class_loss = (1 / (2 * (self.class_weight ** 2))) * class_loss
            weighted_attn_loss = (1 / (2 * (self.attn_forcing_weight ** 2))) * attn_loss
            regularization = torch.log(self.class_weight * self.attn_forcing_weight)
            loss = weighted_class_loss + weighted_attn_loss + regularization
            return loss

        raise NotImplementedError()

    def forward(self, paragraph_attention_mask: Tensor, input_ids: Tensor, attention_mask: Tensor, 
                token_type_ids: Tensor, class_labels: Tensor, attn_labels: Tensor, force_attn: Tensor) -> Tensor:
        """Computes a forward pass through the model.

        Args:
            paragraph_attention_mask (Tensor): attention mask indicating relevant paragph indices.
            input_ids (Tensor): token ids for base model.
            attention_mask (Tensor): attention mask for base model.
            token_type_ids (Tensor, optional): mask indicating special tokens for base model.
            class_labels (Tensor): classification labels.
            attn_labels (Tensor): attention forcing labels.
            force_attn (Tensor): integer boolean if attention should be forced or not.

        Returns:
            Tensor: logits of hierarchical model.
        """
        # Hypothetical Example
        # Batch of 10 cases with 64 paragraphs each: (batch_size, n_paragraphs, max_parag_length) --> (10, 64, 128)

        # Combine first two dimensions to feed through base model (batch_size * n_paragraphs, max_parag_length) --> (640, 128)
        device = input_ids.get_device()
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1)).type(torch.LongTensor)#.to(device)
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1)).type(torch.LongTensor)#.to(device)
        token_type_ids_reshape = None
        if token_type_ids is not None:
            token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1)).type(torch.LongTensor)#.to(device)

        # If i.e using BERT as encoder: 768 hidden units
        # Encode sentences with BERT --> (640, 768)
        paragraph_embeddings = self.base_model(input_ids=input_ids_reshape,
                                               attention_mask=attention_mask_reshape,
                                               token_type_ids=token_type_ids_reshape).pooler_output

        # Reshape back to (batch_size, n_paragraphs, output_size) --> (10, 64, 768)
        paragraph_embeddings = paragraph_embeddings.contiguous().view(input_ids.size(0), self.max_parags, self.hidden_size)
        
        # Adding positional encoding for paragraphs
        paragraph_embeddings = self.pos_encoder(paragraph_embeddings)

        # Compute case embeddings --> (10, 64, 768) and attention values --> (10, 64, 64)
        padding_mask = (paragraph_attention_mask == 0)
        if self.hier_layers > 0:
            case_embeddings, attn_output_weights = self.hier_model(paragraph_embeddings, src_key_padding_mask=padding_mask)
        else:
            case_embeddings, attn_output_weights = self.self_attn(paragraph_embeddings,
                                                                  paragraph_embeddings,
                                                                  paragraph_embeddings,
                                                                  key_padding_mask=padding_mask,
                                                                  need_weights=False)
        
        # Pool case embeddings (choose first embedding -> CLS token) --> (10, 768) NOTE: Experiment with different pooling strategies
        case_embeddings = case_embeddings[:, 0]

        # Pool paragraph attention values --> (10, 64)
        attn_output_weights = attn_output_weights[:, 0]

        # Reshape attention tensor to perform same transformation on all values -> (640, 1)
        attn_output_weights = attn_output_weights.contiguous().view(-1, 1).type(torch.LongTensor)#.to(device)

        # Linear transform attention values --> (640, 1)
        attn_logits = self.attn_head(attn_output_weights)

        # Reshape attention tensor back to original form
        attn_logits = attn_logits.contiguous().view(-1, self.max_parags).type(torch.LongTensor)#.to(device)

        # Dropout
        case_embeddings = self.dropout(case_embeddings)

        # Linear transform case embeddings --> (10, num_labels)
        class_logits = self.cls_head(case_embeddings)
        
        # Compute losses
        attn_loss = self.compute_attention_forcing_loss(attn_logits, attn_labels) * force_attn
        class_loss = self.compute_classification_loss(class_logits, class_labels)
        loss = self.compute_total_loss(class_loss, attn_loss)

        return loss, class_logits, attn_logits