import argparse
import os
import pickle
import random

import numpy as np
import sympy
from torchdistill.common.file_util import make_parent_dirs

from datasets.sampling import DefaultSampling


def get_argparser():
    parser = argparse.ArgumentParser(description='Dummy column mixer')
    parser.add_argument('--input', required=True, help='input dir path')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], help='sub dir name(s)')
    parser.add_argument('--true_eq', default='true_eq', help='true equation sub dir name')
    parser.add_argument('--new_col_sizes', nargs='+', default=[1, 2, 3],
                        help='number(s) of new columns randomly choose')
    parser.add_argument('--output', required=True, help='output dir path')
    return parser


def create_dataset_dict(split_dir_paths, true_eq_dir_path, output_dir_path):
    num_splits = len(split_dir_paths)
    dataset_dict = dict()
    for split_dir_path in split_dir_paths:
        for file_name in os.listdir(split_dir_path):
            dataset_id = '.'.join(file_name.split('.')[:-1])
            if dataset_id not in dataset_dict:
                dataset_dict[dataset_id] = {'splits': list()}

            input_split_file_path = os.path.join(split_dir_path, file_name)
            split_dir_name = os.path.basename(split_dir_path)
            output_split_file_path = os.path.join(output_dir_path, split_dir_name, file_name)
            dataset_dict[dataset_id]['splits'].append((input_split_file_path, output_split_file_path))

    complete_flag = True
    for file_name in os.listdir(true_eq_dir_path):
        dataset_id = '.'.join(file_name.split('.')[:-1])
        input_true_eq_file_path = os.path.join(true_eq_dir_path, file_name)
        true_eq_dir_name = os.path.basename(true_eq_dir_path)
        output_true_eq_file_path = os.path.join(output_dir_path, true_eq_dir_name, file_name)
        dataset_dict[dataset_id]['true_eq'] = (input_true_eq_file_path, output_true_eq_file_path)
        split_file_paths = dataset_dict.get(dataset_id, {'splits': list()})['splits']
        num_split_file_paths = len(split_file_paths)
        if num_split_file_paths != num_splits:
            print(f'Dataset ID `{dataset_id}` has {num_split_file_paths} of {num_splits} split files: '
                  f'{split_file_paths}')
            complete_flag = False

    for dataset_id, sub_dict in dataset_dict.items():
        if 'true_eq' not in sub_dict:
            print(f'Dataset ID `{dataset_id}` misses true equation file')
            complete_flag = False

    if not complete_flag:
        raise FileNotFoundError('Some of dataset IDs miss split file(s) and/or true equation file')
    return dataset_dict


def choose_new_column_positions(num_new_cols, num_org_input_cols):
    new_index_list = list()
    for _ in range(num_new_cols):
        new_index = random.randint(0, num_org_input_cols)
        new_index_list.append(new_index)
    return sorted(new_index_list, reverse=True)


def update_original_column_indices(org_col_indices, new_col_indices):
    updated_real_indices = org_col_indices.copy()
    for new_col_index in new_col_indices:
        for i, org_col_index in enumerate(updated_real_indices):
            if new_col_index <= org_col_index:
                updated_real_indices[i] += 1
    return updated_real_indices


def generate_random_sampling_objs(num_vars):
    sampling_obj_list = list()
    for _ in range(num_vars):
        random_int = random.randint(-32, 32)
        uses_negative = random.random() < 0.5
        sampling_obj = \
            DefaultSampling(np.power(10.0, random_int - 1), np.power(10.0, random_int + 1), uses_negative=uses_negative)
        sampling_obj_list.append(sampling_obj)
    return sampling_obj_list



def expand_dataset_with_dummy_columns(sub_dict, new_col_sizes, output_dir_path):
    num_new_cols = random.choice(new_col_sizes)
    if not isinstance(num_new_cols, int) or num_new_cols <= 0:
        raise ValueError('new_col_sizes should be list of positive integers')

    org_col_indices = None
    new_col_indices = None
    updated_col_indices = None
    sampling_objs = None
    for input_split_file_path, output_split_file_path in sub_dict['splits']:
        delimiter = '\t' if input_split_file_path.endswith('.tsv') \
            else ',' if input_split_file_path.endswith('.csv') else ' '
        tabular_data = np.loadtxt(input_split_file_path, delimiter=delimiter, dtype=np.float32)
        input_columns, target_column = np.hsplit(tabular_data, [-1])
        num_samples, num_org_input_cols = input_columns.shape[:2]
        if new_col_indices is None:
            org_col_indices = list(range(num_org_input_cols))
            new_col_indices = choose_new_column_positions(num_new_cols, num_org_input_cols)
            updated_col_indices = update_original_column_indices(org_col_indices, new_col_indices)
            sampling_objs = generate_random_sampling_objs(num_new_cols)

        for new_col_index, sampling_obj in zip(new_col_indices, sampling_objs):
            dummy_column = np.expand_dims(sampling_obj(num_samples), axis=0).T
            input_columns = np.insert(input_columns, [new_col_index], dummy_column, axis=1)

        make_parent_dirs(output_split_file_path)
        tabular_data = np.hstack((input_columns, target_column))
        np.savetxt(output_split_file_path, tabular_data, delimiter=delimiter)
    return org_col_indices, new_col_indices, updated_col_indices


def expand_equation_with_dummy_variables(input_true_eq_file_path, org_col_indices, new_col_indices,
                                         updated_col_indices, output_true_eq_file_path):
    with open(input_true_eq_file_path, 'rb') as fp:
        true_eq = pickle.load(fp)

    print(f'Original indices: {org_col_indices}')
    print(true_eq)
    org_variables = tuple(sorted(true_eq.free_symbols, key=lambda x: int(x.name[1:]), reverse=True))
    new_variables = tuple([sympy.Symbol(f'x{i}') for i in sorted(updated_col_indices, reverse=True)])
    for old_variable, new_variable in zip(org_variables, new_variables):
        true_eq = true_eq.subs(old_variable, new_variable)

    print(f'New indices: {new_col_indices}')
    print(f'Updated indices: {updated_col_indices}')
    print(true_eq)
    make_parent_dirs(output_true_eq_file_path)
    with open(output_true_eq_file_path, 'wb') as fp:
        pickle.dump(true_eq, fp)


def mix_dummy_columns(dataset_dict, new_col_sizes, output_dir_path):
    for dataset_id, sub_dict in dataset_dict.items():
        print(f'\n[{dataset_id}]')
        org_col_indices, new_col_indices, updated_col_indices = \
            expand_dataset_with_dummy_columns(sub_dict, new_col_sizes, output_dir_path)
        input_true_eq_file_path, output_true_eq_file_path = sub_dict['true_eq']
        expand_equation_with_dummy_variables(input_true_eq_file_path, org_col_indices, new_col_indices,
                                             updated_col_indices, output_true_eq_file_path)


def main(args):
    print(args)
    input_dir_path = os.path.expanduser(args.input)
    split_dir_paths = [os.path.join(input_dir_path, split_dir_name) for split_dir_name in args.splits]
    true_eq_dir_path = os.path.join(input_dir_path, args.true_eq)
    output_dir_path = os.path.expanduser(args.output)
    dataset_dict = create_dataset_dict(split_dir_paths, true_eq_dir_path, output_dir_path)
    mix_dummy_columns(dataset_dict, args.new_col_sizes, output_dir_path)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
