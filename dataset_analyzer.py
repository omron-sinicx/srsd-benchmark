import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
from pandas import DataFrame
from torchdistill.common.file_util import get_file_path_list

from eq.conversion import sympy2sequence


def get_argparser():
    parser = argparse.ArgumentParser(description='Dataset analyzer')
    parser.add_argument('--src', required=True, help='source file/dir path')
    parser.add_argument('--type', default='tabular', choices=['tabular', 'eq'], help='type of src data')
    parser.add_argument('--src_delim', help='delimiter in src data')
    return parser


def load_tabular_dataset(file_path, delimiter):
    mat = np.loadtxt(file_path, delimiter=delimiter)
    return mat


def compute_stats(mat, file_name):
    print(f'\n{file_name}: {len(mat)} samples')
    num_columns = mat.shape[1]
    num_variables = num_columns - 1
    column_names = ['Metric'] + [f'x{i + 1}' for i in range(num_variables)] + ['y']
    min_row = ['MIN'] + [mat[:][i].min() for i in range(num_columns)]
    max_row = ['MAX'] + [mat[:][i].max() for i in range(num_columns)]
    avg_row = ['AVG'] + [mat[:][i].mean() for i in range(num_columns)]
    std_row = ['STD'] + [mat[:][i].std() for i in range(num_columns)]
    table = [min_row, max_row, avg_row, std_row]
    df = DataFrame(table, columns=column_names)
    print(df.to_string(index=False))


def compute_symbol_prior(eq_file_paths):
    symbol_freq_dict = defaultdict(int)
    for eq_file_path in eq_file_paths:
        with open(eq_file_path, 'rb') as fp:
            eq_sympy = pickle.load(fp)
        eq_tree_sequence = sympy2sequence(eq_sympy.evalf(), returns_binary_tree=True)
        for symbol in eq_tree_sequence:
            symbol_freq_dict[str(symbol)] += 1
    total_freq = sum(symbol_freq_dict.values())
    for key, value in symbol_freq_dict.items():
        print(f'{key}:\t{value / total_freq}')


def main(args):
    print(args)
    dataset_type = args.type
    src_file_path = args.src
    src_delimiter = args.src_delim
    src_file_paths = [src_file_path] if os.path.isfile(src_file_path) \
        else get_file_path_list(src_file_path, is_sorted=True)
    if dataset_type == 'tabular':
        for src_file_path in src_file_paths:
            src_dataset = load_tabular_dataset(src_file_path, src_delimiter)
            compute_stats(src_dataset, os.path.basename(src_file_path))
    elif dataset_type == 'eq':
        compute_symbol_prior(src_file_paths)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
