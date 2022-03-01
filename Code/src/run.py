import argparse
from re import L
import logging
from train import finetune_model
from dataset import generate_echr_dataset
from preprocessing import preprocess_dataset
from model import load_model
from evaluation import evaluate
from datasets import DatasetDict

logger = logging.getLogger('run')

argp = argparse.ArgumentParser()
argp.add_argument('function',
    help='Whether to pretrain, finetune or evaluate a model',
    choices=['load', 'finetune'])
argp.add_argument('name',
    help='Name of the Huggingface model to finetune or path to trained model.')
argp.add_argument('-d', '--data',
    help="Path of the dataset to load before finetuning/evaluation")
argp.add_argument('--objective',
    help='Training objective.',
    choices=['binary', 'multilabel'],
    default='binary')
argp.add_argument('-o', '--output', 
    help='Model output path.',
    default='model_outputs')
argp.add_argument('-s', '--sample', 
    help='How many examples to sample for training.',
    default=None)
argp.add_argument('-b', '--base_model', 
    help='What base model was used for the hierarchical training.',
    default='bert-base-uncased')
argp.add_argument('-e', '--evaluate', help='Set flag to evaluate on test set.',
                  action='store_true', default=False)
argp.add_argument('-l', '----load', help='Set flag to load processed data.',
                  action='store_true', default=False)
argp.add_argument('--hierarchical', help='Set flag to use hierarchical model version.',
                  action='store_true', default=False)
args = argp.parse_args()

logger.warning(f'Task: {args.function.capitalize()} {args.name} using {args.objective} classification on dataset at {args.data}.')
n_subset = int(args.sample) if args.sample is not None else None
max_len = 128 if args.hierarchical else 512
num_labels = 21 if args.objective == 'multilabel' else 1

if args.function == 'finetune':
    if args.load == False:
        dataset = generate_echr_dataset(args.data, n_subset=n_subset)
        dataset = preprocess_dataset(dataset, args.objective, args.name, args.hierarchical, max_paragraph_len=max_len)
        dataset.save_to_disk('processed_data/data')
    
    else:
        dataset = DatasetDict.load_from_disk('processed_data/data')

    model = finetune_model(args.name, dataset, args.hierarchical, args.output, 64, max_len)

else:
    dataset = generate_echr_dataset(args.data, n_subset=n_subset)
    dataset = preprocess_dataset(dataset, args.objective, args.name, args.hierarchical, max_paragraph_len=max_len)
    model = load_model(args.name, args.hierarchical, args.base_model, num_labels)

if args.evaluate:
    evaluate(model, dataset, args.hierarchical)