from transformers import AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from trainers import MultilabelTrainer 
from typing import Tuple

def load_model(path) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return model

def predict(model, dataset: Dataset) -> Tuple:
    train_labels =dataset['labels']
    n_train_labels = len(set([x for xs in train_labels for x in xs]))
    train_class = MultilabelTrainer if n_train_labels > 2 else Trainer
    trainer = train_class(model=model)
    predictions = trainer.predict(dataset)
    return predictions