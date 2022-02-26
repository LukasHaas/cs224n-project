import logging
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import DatasetDict
from trainers import MultilabelTrainer 
from evaluation import compute_binary_metrics, compute_multilabel_metrics
from callbacks import LoggingCallback

# Initialize Logger
logging.basicConfig()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')

DEFAULT_TRAIN_ARGS = TrainingArguments(
    output_dir='model_outputs',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    evaluation_strategy='epoch', # run validation at the end of each epoch
    save_strategy='epoch',
    learning_rate=2e-5, # 1e-3
    logging_steps=1,
    load_best_model_at_end=True,
    seed=1111
)

def finetune_model(name: str, dataset: DatasetDict, output: str, log: bool=True,
                   early_stopping: int=2,
                   train_args: TrainingArguments=DEFAULT_TRAIN_ARGS) -> AutoModelForSequenceClassification:
    """Finetunes a given Huggingface model.

    Args:
        name (str): name of Huggingface model.
        dataset (DatasetDict): dataset.
        output (str): output directory of trained model.
        log (bool, optional): if results should be logged. Defaults to True.
        early_stopping (int, optional): early stopping patience. Defaults to 2.
        train_args (TrainingArguments, optional): training arguments. Defaults to DEFAULT_TRAIN_ARGS.

    Returns:
        AutoModelForSequenceClassification: finetuned model.
    """
    train_labels = dataset['train']['labels']
    n_train_labels = 2 if train_labels.dim() == 1 else train_labels.size()[1]
    logger.warning(f'Number of classes detected: {n_train_labels}.')

    logger.warning(f'Downloading model: {name}.')
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_train_labels)

    train_args.__setattr__('output_dir', output)
    trainer = __generate_trainer(dataset, model, train_args)

    if log:
        trainer.add_callback(LoggingCallback(f'finetune_trainer/log_{name}.jsonl'))
    
    if early_stopping > 0:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=early_stopping,
                                                   early_stopping_threshold=0.0))

    logger.warning(f'Starting training.')
    trainer.train()
    return model


def __generate_trainer(dataset: DatasetDict,
                       model: AutoModelForSequenceClassification,
                       train_args: TrainingArguments=None) -> Trainer:
    """Generates a trainer to train for the given objective.

    Args:
        dataset (DatasetDict): dataset.
        model (AutoModelForSequenceClassification): model.
        train_args (TrainingArguments, optional): training arguments.
            Defaults to None.

    Returns:
        Trainer: trainer to train a model.
    """
    train_labels = dataset['train']['labels']
    n_train_labels = 2 if train_labels.dim() == 1 else train_labels.size()[1]
    eval_fnc = compute_binary_metrics if n_train_labels == 2 else compute_multilabel_metrics
    class_name = Trainer if n_train_labels == 2 else MultilabelTrainer
    trainer = class_name(
        model=model,
        args=train_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        compute_metrics=eval_fnc
    )
    return trainer