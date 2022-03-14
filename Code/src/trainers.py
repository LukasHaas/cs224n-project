import torch
from transformers import Trainer

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

class aLEXaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, class_logits, attn_logits = model(**inputs)
        return (loss, class_logits) if return_outputs else loss 