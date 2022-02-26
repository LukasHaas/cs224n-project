from transformers import AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from trainers import MultilabelTrainer 
from typing import Tuple, Any

def load_model(path: str) -> AutoModelForSequenceClassification:
    """Load a Huggingface moddel from path.

    Args:
        path (str): path to model.

    Returns:
        AutoModelForSequenceClassification: model.
    """
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return model

def predict(model: str, dataset: Dataset) -> Tuple:
    """Makes predictions given a Huggingface model.

    Args:
        model (str): trained model.
        dataset (Dataset): dataset.

    Returns:
        Tuple: prediction tuple.
    """
    train_labels = dataset['train']['labels']
    n_train_labels = 2 if train_labels.dim() == 1 else train_labels.size()[1]
    train_class = MultilabelTrainer if n_train_labels > 2 else Trainer
    trainer = train_class(model=model)
    predictions = trainer.predict(dataset)
    return predictions