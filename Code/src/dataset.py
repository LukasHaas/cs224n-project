import os
import json
from glob import glob
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm

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

def generate_echr_dataset(path: str, n_subset: int=None, shuffle: bool=True, seed: int=1111) -> DatasetDict:
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
    train = __process_echr_dataset(f'{path}/EN_train')
    val = __process_echr_dataset(f'{path}/EN_dev')
    test = __process_echr_dataset(f'{path}/EN_test')
    splits = [train, val, test]

    if train.num_rows == 0:
      raise Exception('Dataset is empty.')

    if shuffle:
      splits = [x.shuffle(seed=seed) for x in splits]

    if n_subset:
      splits = [x.select(range(n_subset)) for x in splits]

    echr_dataset = DatasetDict(
        train=splits[0],
        val=splits[1],
        test=splits[2]
    )
    return echr_dataset