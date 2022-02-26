import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_binary_metrics(eval_pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(predictions)
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
    predictions = np.round(sigmoid(logits))
    print(predictions)
    eval_dict = {
        'precision_micro': precision_score(labels, predictions, average='micro'),
        'recall_micro': recall_score(labels, predictions, average='micro'),
        'f1_micro': f1_score(labels, predictions, average='micro'),
        'precision_macro': precision_score(labels, predictions, average='macro'),
        'recall_macro': recall_score(labels, predictions, average='macro'),
        'f1_macro': f1_score(labels, predictions, average='macro')
    }
    return eval_dict