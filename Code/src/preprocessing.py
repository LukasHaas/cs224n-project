import logging
import numpy as np
import torch
from typing import Dict
from functools import partial
from datasets import DatasetDict
from transformers import AutoTokenizer

logger = logging.getLogger('preprocessing')
OBJECTIVES = ['binary', 'multilabel']

def __to_one_hot(example: Dict, mapping: Dict) -> Dict:
    """Performs one hot encoding for a list of labels.

    Args:
        example (Dict): a single data point.
        mapping (Dict): dictionary mapping labels to indices.

    Returns:
        Dict: one hot encoded labels.
    """
    indices = [mapping[y] for y in example['articles'] if y in mapping]
    one_hots = np.zeros(len(mapping.keys()))
    one_hots[indices] = 1
    return {
        'labels': torch.Tensor(one_hots)
    }

def __merge_facts(example: Dict) -> Dict:
    """Merges all facts in the case.

    Args:
        example (Dict): a single data point.

    Returns:
        Dict: _description_
    """
    return {
        'facts': ' '.join(example['facts'])
    }
  
def preprocess_dataset(dataset: DatasetDict, objective: str, tokenizer: str,
                       merge_facts=False) -> DatasetDict:
    """Preprocesses dataset.

    Args:
        dataset (DatasetDict): dataset.
        objective (str): either binary or multilabel.
        objective (str): name of Huggingface tokenizer.
        merge_facts (bool, optional): whether all facts in case should be
            merged to a single string. Defaults to False.

    Returns:
        DatasetDict: preprocessed dataset.
    """
    logger.warning(f'Preprocessing dataset for {objective} classification objective.')
    assert objective in OBJECTIVES, f'Objective must be one of {OBJECTIVES}.'
    if merge_facts:
        dataset = dataset.map(__merge_facts)
    
    if objective == 'multilabel':
        dataset = dataset.remove_columns(['labels'])
        train_labels = dataset['train']['articles']
        train_labels = set([x for xs in train_labels for x in xs])
        mapping = {x: i for i, x in enumerate(sorted(train_labels))}
        one_hot_encode = partial(__to_one_hot, mapping=mapping)
        dataset = dataset.map(one_hot_encode)

    dataset = tokenize(dataset, tokenizer)
    dataset.set_format('torch')
    dataset = dataset.remove_columns(['articles', 'ids'])
    return dataset

def tokenize(dataset: DatasetDict, tokenizer: str, padding: bool=True,
             truncation: bool=True, max_length: int=512,
             remove_text: bool=True) -> DatasetDict:
    """Tokenizes dataset.

    Args:
        dataset (DatasetDict): dataset,
        tokenizer (str): name of Huggingface tokenizer.
        padding (bool, optional): whether sequence should be padded. Defaults to True.
        truncation (bool, optional): whether sequence should be truncated. Defaults to True.
        max_length (int, optional): max sequence length. Defaults to 512.
        remove_text (bool, optional): wheether the text in the dataset should be removed.
            Default to True.

    Returns:
        DatasetDict: tokenized dataset.
    """
    logger.warning(f'Tokenizing dataset using {tokenizer} tokenizer.')
    tokenize = AutoTokenizer.from_pretrained(tokenizer)
    dataset = dataset.map(
               lambda x: tokenize(x['facts'], 
                    padding=padding, 
                    truncation=truncation,
                    max_length=max_length),
                batched=True,
                batch_size=16
              )
    if remove_text:
        dataset = dataset.remove_columns(['facts'])
    
    return dataset
    

