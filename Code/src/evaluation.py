import logging
import numpy as np
from typing import Any
from model import predict
from datasets import DatasetDict
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger('evaluation')

def compute_binary_metrics(eval_pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.round(sigmoid(logits))
    eval_dict = {
        'precision': precision_score(labels, predictions, average='macro'),
        'recall': recall_score(labels, predictions, average='macro'),
        'f1': f1_score(labels, predictions, average='macro')
    }
    return eval_dict

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_multilabel_metrics(eval_pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    print(logits)
    predictions = np.round(sigmoid(logits))
    eval_dict = {
        'precision_micro': precision_score(labels, predictions, average='micro'),
        'recall_micro': recall_score(labels, predictions, average='micro'),
        'f1_micro': f1_score(labels, predictions, average='micro'),
        'precision_macro': precision_score(labels, predictions, average='macro'),
        'recall_macro': recall_score(labels, predictions, average='macro'),
        'f1_macro': f1_score(labels, predictions, average='macro')
    }
    return eval_dict

def evaluate(model: Any, dataset: Any, hierarchical: bool):
    """Evaluates model on test dataset.

    Args:
        model (Any): trained pytorch model.
        dataset (DatasetDict): dataset.
        hierarchical (bool) if using hierarchical model.
    """
    logger.warning('Evaluating on test set.')
    predictions = predict(model, dataset['val'], hierarchical)
    train_labels = dataset['train'][0]['labels']
    eval_fnc = compute_binary_metrics if train_labels.dim() == 0 else compute_multilabel_metrics
    evaluation = eval_fnc(predictions)
    print(evaluation)