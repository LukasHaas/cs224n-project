import logging
import pickle
import numpy as np
import torch
from typing import Dict, Any, List
from functools import partial
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from torch import Tensor

logger = logging.getLogger('preprocessing')
OBJECTIVES = ['binary', 'multilabel']
DUMMY_TEXT = 'dummy'

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

def __create_attn_labels(example: Dict, num_paragraphs: int) -> Dict:
    """One hot encodes attention labels used for aLEXa model.

    Args:
        example (Dict): a single example.
        num_paragraphs (int): the number of paragraphs.

    Returns:
        Dict: preprocessed example.
    """
    one_hots = np.zeros(num_paragraphs)
    for relevant_fact in example['rationale']:
        if relevant_fact < num_paragraphs:
            one_hots[relevant_fact] = 1

    return {
        'attention_labels': torch.Tensor(one_hots)
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

def __to_hierarchical(example: Dict, max_paragraphs: int) -> Dict:
    """Converts facts to hierarchical texts

    Args:
        example (Dict): single case example.
        max_paragraphs (int, optional): maximum number of paragraphs considered. Defaults to 64.
        max_paragraph_len (int, optional): maximum length of paragraphs. Defaults to 128.

    Returns:
        Dict: new case example.
    """
    paragraphs = example['facts'][:max_paragraphs]
    n_paragraphs = len(paragraphs)
    if n_paragraphs < max_paragraphs:
        paragraphs = paragraphs + ([DUMMY_TEXT] * (max_paragraphs - n_paragraphs))

    attention_mask = ([1] * n_paragraphs) + ([0] * (max_paragraphs - n_paragraphs))
    return {
        'facts': paragraphs,
        'paragraph_attention_mask': attention_mask
    }
  
def preprocess_dataset(dataset: DatasetDict, objective: str, tokenizer: str,
                       hierarchical: bool=False, alexa: bool=False,
                       max_paragraphs: int=64, max_paragraph_len: int=512) -> DatasetDict:
    """Preprocesses dataset.

    Args:
        dataset (DatasetDict): dataset.
        objective (str): either binary or multilabel.
        tokenizer (str): name of Huggingface tokenizer.
        hierarchical (bool, optional): whether model is hierarchical. Defaults to False.
        alexa (bool, optional): whether the attention forcing model is used. Defaults to False.
        max_paragraphs (int, optional): maximum number of paragraphs considered. Defaults to 64.
        max_paragraph_len (int, optional): maximum length of paragraphs. Defaults to 128.

    Returns:
        DatasetDict: preprocessed dataset.
    """
    logger.warning(f'Preprocessing dataset for {objective} classification objective.')
    assert objective in OBJECTIVES, f'Objective must be one of {OBJECTIVES}.'

    if hierarchical:
        dataset = dataset.map(lambda x: __to_hierarchical(x, max_paragraphs))
    elif alexa:
        dataset = dataset.map(lambda x: __to_hierarchical(x, max_paragraphs))
        dataset = dataset.map(lambda x: __create_attn_labels(x, max_paragraphs))
        dataset = dataset.remove_columns(['rationale'])
    else:
        dataset = dataset.map(__merge_facts)
    
    if objective == 'multilabel':
        dataset = dataset.remove_columns(['labels'])
        train_labels = dataset['train']['articles']
        train_labels = set([x for xs in train_labels for x in xs])
        mapping = {x: i for i, x in enumerate(sorted(train_labels))}
        with open(f'model_outputs/multilabel_mapping.pickle', 'wb') as f:
            pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

        dataset = dataset.map(lambda x: __to_one_hot(x, mapping))

    dataset = tokenize(dataset, tokenizer, max_length=max_paragraph_len)
    dataset.set_format('torch')
    return dataset

def __tokenize_hierarchical(datasets: DatasetDict, tokenizer: AutoTokenizer, max_length: int) -> DatasetDict:
    """Tokenizes hierarchical datasets.

    Args:
        datasets (Dict): datasets.
        tokenizer (AutoTokenizer): tokenizer function.
        max_length (int): maximum sentence length.

    Returns:
        DatasetDict: new datasets.
    """
    new_datasets = []
    for split in ['train', 'val', 'test']:
        dataset = datasets[split]
        facts = [item for sublist in dataset['facts'] for item in sublist]
        n_examples = len(dataset['facts'])
        n_paragraphs = len(dataset['facts'][0])

        new_dataset = {
            'paragraph_attention_mask': dataset['paragraph_attention_mask'] ,
            'labels': dataset['labels'],
        }

        tokenized = tokenizer(facts,  padding=True, truncation=True, max_length=max_length)
        new_dataset['input_ids'] = np.array(tokenized['input_ids']).reshape((n_examples, n_paragraphs, -1))
        new_dataset['attention_mask'] = np.array(tokenized['attention_mask']).reshape((n_examples, n_paragraphs, -1))

        if 'token_type_ids' in tokenized:
            new_dataset['token_type_ids'] = np.array(tokenized['token_type_ids']).reshape((n_examples, n_paragraphs, -1))

        if 'attention_labels' in dataset.column_names:
            new_dataset['attention_labels'] = dataset['attention_labels'],
            new_dataset['attention_label_mask'] = dataset['attention_label_mask']

        for key, value in new_dataset.items():
            if key != 'attention_labels':
                new_dataset[key] = Tensor(value)
            else:
                new_dataset[key] = Tensor(value[0])
        
        new_datasets.append(Dataset.from_dict(new_dataset))

    new_datasets = DatasetDict(
        train=new_datasets[0],
        val=new_datasets[1],
        test=new_datasets[2]
    )

    return new_datasets


def tokenize(dataset: DatasetDict, tokenizer: str, padding: bool=True,
             truncation: bool=True, max_length: int=512) -> DatasetDict:
    """Tokenizes dataset.

    Args:
        dataset (DatasetDict): dataset,
        tokenizer (str): name of Huggingface tokenizer.
        padding (bool, optional): whether sequence should be padded. Defaults to True.
        truncation (bool, optional): whether sequence should be truncated. Defaults to True.
        max_length (int, optional): max sequence length. Defaults to 512.

    Returns:
        DatasetDict: tokenized dataset.
    """
    logger.warning(f'Tokenizing dataset using {tokenizer} tokenizer.')
    tokenize = AutoTokenizer.from_pretrained(tokenizer) #tokenizer)

    # Check if hierarchical
    if type(dataset['train']['facts'][0]) == str:
        dataset = dataset.map(
                lambda x: tokenize(x['facts'], padding=padding, truncation=truncation, max_length=max_length),
                    batched=True,
                    batch_size=16)
        dataset = dataset.remove_columns(['facts', 'articles', 'ids'])
    else:
        dataset = __tokenize_hierarchical(dataset, tokenize, max_length=max_length)
    
    return dataset
    

