import logging
from typing import Any
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import DatasetDict, Dataset
from dataset import HierarchicalDataset
from trainers import MultilabelTrainer, aLEXaTrainer
from evaluation import compute_binary_metrics, compute_multilabel_metrics
from callbacks import LoggingCallback
from hierarchical import HierarchicalModel
from alexa import aLEXa
from torch import Tensor

# Initialize Logger
logging.basicConfig()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')

DEFAULT_TRAIN_ARGS = TrainingArguments(
    output_dir='model_outputs',
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    evaluation_strategy='steps',
    eval_steps=25,
    gradient_accumulation_steps=64,
    save_strategy='epoch',
    learning_rate=2e-5, #5e-6 3e-6 # 1e-5 2e-5 1e-3
    logging_steps=1,
    load_best_model_at_end=True,
    seed=1111
)

def compute_class_weights(dataset: Dataset, pos_weight=1.0) -> Tensor:
    """Computes class weights for the multilabel task.

    Args:
        dataset (Dataset): train dataset.

    Returns:
        Tensor: weights.
    """
    labels = dataset['labels']
    label_sums = labels.sum(dim=0)
    n_labels = label_sums.size()[0]
    label_weights = (label_sums / labels.sum()) * n_labels
    pos_weights = Tensor([pos_weight]).repeat(n_labels)
    return label_weights, pos_weights

def finetune_model(model: Any, dataset: DatasetDict, hierarchical: bool, alexa: bool, output: str,
                   max_paragraphs: int=64, max_paragraph_len: int=512, log: bool=True, early_stopping: int=3,
                   train_args: TrainingArguments=DEFAULT_TRAIN_ARGS) -> AutoModel:
    """Finetunes a given model.

    Args:
        model (Any): name of Huggingface model or trainable object.
        dataset (DatasetDict): dataset.
        hierarchical (bool): if hierarchical model should be chosen.
        alexa (bool): whether the attention forcing model is used.
        output (str): output directory of trained model.
        max_paragraphs (int, optional): maximum number of paragraphs considered. Defaults to 64.
        max_paragraph_len (int, optional): maximum length of paragraphs. Defaults to 128.
        log (bool, optional): if results should be logged. Defaults to False.
        early_stopping (int, optional): early stopping patience. Defaults to 2.
        train_args (TrainingArguments, optional): training arguments. Defaults to DEFAULT_TRAIN_ARGS.

    Returns:
        AutoModelForSequenceClassification: finetuned model.
    """
    train_labels = Tensor(dataset['train']['labels'])
    n_train_labels = 2 if train_labels.dim() == 1 else train_labels.size()[1]
    logger.warning(f'Number of classes detected: {n_train_labels}.')

    train_args.__setattr__('output_dir', output)
    logger.warning(f'Downloading model: {model}.')

    # Loaded pre-loaded
    if type(model) != str:
        loaded_model = model

    # Hierarchical model types
    elif hierarchical or alexa:
        base_model = AutoModel.from_pretrained(model)
        n_train_labels = 1 if train_labels.dim() == 1 else train_labels.size()[1]
        lw, pw = compute_class_weights(dataset['train'], pos_weight=1.2)

        if hierarchical:
            loaded_model = HierarchicalModel(base_model, n_train_labels, max_paragraphs, max_paragraph_len,
                                             hier_layers=2, freeze_base=False, label_weights=lw,
                                             pos_weights=pw)
        else:
            loaded_model = aLEXa(base_model, n_train_labels, max_paragraphs, max_paragraph_len,
                                    hier_layers=1, learn_loss_weights=True, freeze_base=False,
                                    label_weights=lw, pos_weights=pw)

    # Base models 
    else:
        loaded_model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=n_train_labels)
 
    trainer = generate_trainer(dataset, loaded_model, hierarchical, alexa, train_args)

    if log:
        log_name = model.split('/')[-1] if type(model) == str else 'loaded_model'
        trainer.add_callback(LoggingCallback(f'finetune_trainer/log_{log_name}.jsonl'))
    
    if early_stopping > 0:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=early_stopping,
                                                   early_stopping_threshold=0.0))
    
    logger.warning(f'Starting training.')
    trainer.train()
    return loaded_model


def generate_trainer(dataset: DatasetDict,
                       model: Any,
                       hierarchical: bool,
                       alexa: bool,
                       train_args: TrainingArguments=None) -> Trainer:
    """Generates a trainer to train for the given objective.

    Args:
        dataset (DatasetDict): dataset.
        model (AutoModelForSequenceClassification): model.
        hierarchical (bool): if hierarchical model should be chosen.
        alexa (bool): whether the attention forcing model is used.
        train_args (TrainingArguments, optional): training arguments.
            Defaults to None.

    Returns:
        Trainer: trainer to train a model.
    """
    train_labels = Tensor(dataset['train']['labels'])
    n_train_labels = 2 if train_labels.dim() == 1 else train_labels.size()[1]
    eval_fnc = compute_binary_metrics if n_train_labels == 2 else compute_multilabel_metrics
    class_name = Trainer if n_train_labels == 2 else MultilabelTrainer

    if hierarchical:
        return Trainer(
            model=model,
            args=train_args,
            train_dataset=HierarchicalDataset(dataset['train']),
            eval_dataset=HierarchicalDataset(dataset['val']),
            compute_metrics=eval_fnc
        )

    if alexa:
        return aLEXaTrainer(
            model=model,
            args=train_args,
            train_dataset=HierarchicalDataset(dataset['train']),
            eval_dataset=HierarchicalDataset(dataset['val']),
            compute_metrics=eval_fnc
        )

    return class_name(
        model=model,
        args=train_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        compute_metrics=eval_fnc
    )