from transformers import AutoModelForSequenceClassification, AutoModel, Trainer
from datasets import Dataset
from dataset import HierarchicalDataset
from trainers import MultilabelTrainer 
from typing import Tuple, Any
import torch
from hierarchical import HierarchicalModel
from tqdm import tqdm

def load_model(path: str, hierarchical: bool, base_model: str=None, num_labels: int=21) -> Any:
    """Load a Huggingface moddel from path.

    Args:
        path (str): path to model.
        hierarchical (bool) if using hierarchical model.
        base_model (str): name of base model if model is hierarchical. Defeaults to None.
        num_labels (int): number of labels in the model. Defaults to 21.

    Returns:
        Any: model.
    """
    if hierarchical:
        checkpoint = torch.load(path)
        base_model = AutoModel.from_pretrained(base_model)
        model = HierarchicalModel(base_model, num_labels, 64, 128, 1, False)
        model.load_state_dict(checkpoint)
    else:
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
    if hierarchical == False:
        train_class = MultilabelTrainer if n_labels > 2 else Trainer
        trainer = train_class(model=model)
        return trainer.predict(dataset)

    trainloader = torch.utils.data.DataLoader(HierarchicalDataset(dataset), shuffle=False, batch_size=32)
    logits = []

    with torch.no_grad():
        model.eval()
        for data in tqdm(trainloader):
            _, batch_logits = model(**data)
            logits += batch_logits

    logits = torch.stack(logits)
    return logits