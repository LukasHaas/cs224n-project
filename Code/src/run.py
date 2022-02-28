import argparse
from re import L
import logging
from train import finetune_model
from dataset import generate_echr_dataset
from preprocessing import preprocess_dataset
from model import load_model
from evaluation import evaluate

logger = logging.getLogger('run')

argp = argparse.ArgumentParser()
argp.add_argument('function',
    help='Whether to pretrain, finetune or evaluate a model',
    choices=['load', 'finetune'])
argp.add_argument('name',
    help='Name of the Huggingface model to finetune or path to trained model.')
argp.add_argument('data',
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
argp.add_argument('-e', '--evaluate', help='Set flag to evaluate on test set.',
                  action='store_true', default=False)
argp.add_argument('--hierarchical', help='Set flag to use hierarchical model version.',
                  action='store_true', default=False)
args = argp.parse_args()

logger.warning(f'Task: {args.function.capitalize()} {args.name} using {args.objective} classification on dataset at {args.data}.')

if args.function == 'finetune':
    dataset = generate_echr_dataset(args.data, n_subset=int(args.sample))

    max_len = 128 if args.hierarchical else 512
    dataset = preprocess_dataset(dataset, args.objective, args.name, args.hierarchical, max_paragraph_len=max_len)
    model = finetune_model(args.name, dataset, args.hierarchical, args.output, 64, max_len)

elif args.functon == 'load':
    dataset = generate_echr_dataset(args.data, n_subset=int(args.sample))
    dataset = preprocess_dataset(dataset, args.objective, args.name, 'hier' not in args.name)
    model = load_model(args.name)

if args.evaluate:
    evaluate(model, dataset)