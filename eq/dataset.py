import collections
import os
import pickle
import random

import numpy as np
import sympy
import torch
from torch._six import string_classes
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torchdistill.common.constant import def_logger
from torchdistill.common.file_util import get_file_path_list
from torchdistill.datasets.collator import register_collate_func
from torchdistill.datasets.registry import register_dataset

from eq.conversion import sympy2sequence
from eq.vocabulary import SymbolVocabulary

logger = def_logger.getChild(__name__)
DTYPE_DICT = {'float32': np.float32, 'float64': np.float64}


def get_sorted_file_paths(file_or_dir_paths):
    if isinstance(file_or_dir_paths, str):
        file_or_dir_paths = [file_or_dir_paths]

    file_path_list = list()
    for file_or_dir_path in file_or_dir_paths:
        file_path = os.path.expanduser(file_or_dir_path)
        if os.path.isfile(file_path):
            file_path_list.append(file_path)
        else:
            sub_list = get_sorted_file_paths(get_file_path_list(file_path, is_sorted=True))
            file_path_list.extend(sub_list)
    return file_path_list


def normalize_tabular_data(tabular_data):
    min_values = tabular_data.min(0)
    max_values = tabular_data.max(0)
    return (tabular_data - min_values) / (max_values - min_values)


@register_collate_func
def default_collate_w_sympy(batch):
    # Extended `default_collate` function in PyTorch

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate_w_sympy([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate_w_sympy([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate_w_sympy(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate_w_sympy(samples) for samples in transposed]
    elif isinstance(elem, sympy.core.Basic):
        return batch

    raise TypeError(default_collate_err_msg_format.format(elem_type))


@register_dataset
class SymbolicRegressionDataset(Dataset):
    def __init__(self, tabular_data_file_path, true_eq_file_path=None, dtype_str=None):
        super().__init__()
        if true_eq_file_path is not None:
            true_eq_file_path = os.path.expanduser(true_eq_file_path)
            if not os.path.isfile(true_eq_file_path):
                raise FileNotFoundError(f'true_eq_file_path is given (`{true_eq_file_path}`), but not found')

        self.tabular_data_file_path = os.path.expanduser(tabular_data_file_path)
        dtype = DTYPE_DICT.get(dtype_str, np.float32)
        delimiter = '\t' if self.tabular_data_file_path.endswith('.tsv') else ' '
        tabular_data = np.loadtxt(tabular_data_file_path, delimiter=delimiter, dtype=dtype)
        self.samples = tabular_data[:, :-1]
        self.targets = tabular_data[:, -1]
        if true_eq_file_path is None:
            self.true_eq_file_path = None
            self.true_eq = None
            self.symbol_sequence = list()
        else:
            self.true_eq_file_path = os.path.expanduser(true_eq_file_path)
            with open(self.true_eq_file_path, 'rb') as fp:
                self.true_eq = pickle.load(fp)
            self.symbol_sequence = sympy2sequence(self.true_eq.evalf(), returns_binary_tree=True)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]
        return sample, target

    def __len__(self):
        return len(self.samples)

    def get_true_eq(self):
        return self.true_eq

    def get_symbol_sequence(self):
        return self.symbol_sequence


@register_dataset
class EquationTreeDataset(Dataset):
    def __init__(self, tabular_data_file_paths, true_eq_file_paths, max_num_variables, num_samples_per_eq,
                 uses_sympy_eq=False, symbols=None, normalizes=True, dtype_str=None):
        super().__init__()
        self.tabular_data_file_paths = get_sorted_file_paths(tabular_data_file_paths)
        self.true_eq_file_paths = get_sorted_file_paths(true_eq_file_paths)
        assert len(self.tabular_data_file_paths) == len(self.true_eq_file_paths), \
            f'The number of tabular data files ({len(self.tabular_data_file_paths)}) ' \
            f'should match that of the true equation files ({len(self.true_eq_file_paths)})'
        self.max_num_variables = max_num_variables
        self.num_samples_per_eq = num_samples_per_eq
        self.uses_sympy_eq = uses_sympy_eq
        self.vocabulary = SymbolVocabulary(symbols=symbols, max_num_variables=max_num_variables)
        dtype = DTYPE_DICT.get(dtype_str, np.float32)
        self.samples_list = list()
        self.targets_list = list()
        self.true_eq_list = list()
        self.symbol_sequence_list = list()
        self.target_sequence_list = list()
        for tabular_data_file_path, true_eq_file_path in zip(self.tabular_data_file_paths, self.true_eq_file_paths):
            delimiter = '\t' if tabular_data_file_path.endswith('.tsv') else ' '
            try:
                tabular_data = np.loadtxt(tabular_data_file_path, delimiter=delimiter, dtype=dtype)
            except:
                logger.info(f'Skipping `{tabular_data_file_path}` due to some error while loading')
                continue

            if normalizes:
                num_org_samples = tabular_data.shape[0]
                tabular_data = normalize_tabular_data(tabular_data)
                tabular_data = tabular_data[np.isfinite(tabular_data).all(axis=1)]
                num_finite_samples = tabular_data.shape[0]
                if tabular_data.shape[0] < self.num_samples_per_eq:
                    logger.info(f'Skipping `{tabular_data_file_path}`. '
                                f'At least {self.num_samples_per_eq} samples should be finite, but only '
                                f'{num_finite_samples} of {num_org_samples} samples are finite after normalization.')
                    continue

            self.samples_list.append(tabular_data)
            self.targets_list.append(tabular_data[:, -1])
            with open(true_eq_file_path, 'rb') as fp:
                true_eq = pickle.load(fp)

            self.true_eq_list.append(true_eq)
            symbol_sequence = sympy2sequence(true_eq.evalf(), returns_binary_tree=True)
            self.symbol_sequence_list.append(symbol_sequence)
            self.target_sequence_list.append(self.vocabulary.convert_symbols_to_indices(symbol_sequence))

    def __getitem__(self, index):
        samples = self.samples_list[index]
        targets = self.targets_list[index]
        true_eq_file_path = self.true_eq_file_paths[index]
        sample_indices = random.sample(range(0, len(samples)), self.num_samples_per_eq)
        target_representation = \
            self.true_eq_list[index] if self.uses_sympy_eq else torch.LongTensor(self.target_sequence_list[index])
        return np.expand_dims(samples[sample_indices], -1), targets[sample_indices], \
               target_representation, true_eq_file_path

    def __len__(self):
        return len(self.samples_list)
