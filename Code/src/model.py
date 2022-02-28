from transformers import AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from dataset import HierarchicalDataset
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

def predict(model: str, dataset: Dataset, hierarchical: bool) -> Tuple:
    """Makes predictions given a Huggingface model.

    Args:
        model (str): trained model.
        dataset (Dataset): dataset.
        hierarchical (bool) if using hierarchical model.

    Returns:
        Tuple: prediction tuple.
    """
    labels = dataset[0]['labels']
    n_labels = 2 if labels.dim() == 0 else len(labels)
    train_class = MultilabelTrainer if labels > 2 else Trainer
    dataset = HierarchicalDataset(dataset) if hierarchical else dataset
    trainer = train_class(model=model)
    predictions = trainer.predict(dataset)
    return predictions