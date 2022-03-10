import os
import json
import logging
from glob import glob
from typing import List
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import torch

# Initialize Logger
logger = logging.Logger('dataset')

def __process_echr_dataset(path: str) -> Dataset:
    """Processes a split of the ECHR dataset.

    Args:
        path (str): path to dataset.

    Returns:
        Dataset: PyTorch dataset.
    """
    all_files = glob(os.path.join(path, '*.json'))
    data = {
        'facts': [],
        'articles': [],
        'labels': [],
        'ids': []
    }

    for filename in tqdm(all_files):
        with open(filename) as f:
            case_data = json.load(f)

        data['facts'].append(case_data['TEXT'])
        data['articles'].append(case_data['VIOLATED_ARTICLES'])
        data['labels'].append(int(len(case_data['VIOLATED_ARTICLES']) > 0))
        data['ids'].append(case_data['ITEMID'])

    df = pd.DataFrame.from_dict(data)
    custom_dataset = Dataset.from_pandas(df)
    return custom_dataset

def generate_echr_dataset(path: str, n_subset: int=None, shuffle: bool=True,
                          seed: int=1111) -> DatasetDict:
    """Generates the Chalkidis 2019 ECHR dataset.

    Args:
        path (str): path to dataset.
        n_subset (int, optional): size of the sampled dataset splits. If None, 
            all data is selected. Defaults to None.
        shuffle (bool, optional): whether data should be shuffled. Defaults to True.
        seed (int, optional): seed for shuffling. Defaults to 1111.

    Raises:
        Exception: if the dataset at the specified path does not exist or is empty.

    Returns:
        DatasetDict: ECHR dataset.
    """
    logger.warning(f'Loading dataset from path: {path}')
    train = __process_echr_dataset(f'{path}/EN_train')
    val = __process_echr_dataset(f'{path}/EN_dev')
    test = __process_echr_dataset(f'{path}/EN_test')
    splits = [train, val, test]
    logger.warning('Loaded train, val, and test splits.')

    if train.num_rows == 0:
        raise Exception('Dataset is empty.')

    if shuffle:
        splits = [x.shuffle(seed=seed) for x in splits]
        logger.info('Shuffled dataset.')

    if n_subset:
        splits = [x.select(range(n_subset)) for x in splits]

    logger.warning(f'Train split size: {splits[0].num_rows}')
    echr_dataset = DatasetDict(
        train=splits[0],
        val=splits[1],
        test=splits[2]
    )
    return echr_dataset

class HierarchicalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset):
        self.input_ids = self.__stack_tensors__(dataset['input_ids'])
        self.attention_mask = self.__stack_tensors__(dataset['attention_mask'])
        self.paragraph_attention_mask = dataset['paragraph_attention_mask']
        self.labels = dataset['labels']

        if 'token_type_ids' in dataset.column_names:
            self.token_type_ids = self.__stack_tensors__(dataset['token_type_ids'])

    def __len__(self):
        return self.labels.size()[0]
    
    def __stack_tensors__(self, feature: List):
        all_data = []
        for example in feature:
            all_data.append(torch.stack(example))
        return torch.stack(all_data)

    def __getitem__(self, idx):
        sample_dict = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'paragraph_attention_mask': self.paragraph_attention_mask[idx],
            'labels': self.labels[idx]
        }

        try:
            sample_dict['token_type_ids'] = self.token_type_ids[idx]
        except AttributeError:
            pass
        
        return sample_dict
