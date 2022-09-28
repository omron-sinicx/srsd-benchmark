import argparse
import os
import pickle
from pathlib import Path
from shutil import copyfile

import numpy as np
import sympy
from sympy import Symbol, lambdify


def get_argparser():
    parser = argparse.ArgumentParser(description='Model selector for DSO and AI Feynman')
    parser.add_argument('--est', nargs='+', required=True, help='dir path for pickled, estimated equations')
    parser.add_argument('--val', required=True, help='validation dataset dir path')
    parser.add_argument('--est_delim', default='.txt', help='file name delimiter for estimated equation files')
    parser.add_argument('--val_delim', default='.txt', help='file name delimiter for validation dataset files')
    parser.add_argument('-dec_idx', action='store_true', help='decrement variable indices for estimated equation(s)')
    parser.add_argument('--output', help='dir path to store the best model per dataset')
    return parser


def get_model_paths_dict(est_dir_paths, est_delim):
    est_dir_paths = [os.path.expanduser(dir_path) for dir_path in est_dir_paths]
    model_paths_dict = dict()
    for est_dir_path in est_dir_paths:
        for file_name in os.listdir(est_dir_path):
            elements = file_name.split(est_delim)
            if len(elements) < 2:
                continue

            model_file_path = os.path.join(est_dir_path, file_name)
            model_key = elements[0]
            if model_key not in model_paths_dict:
                model_paths_dict[model_key] = list()
            model_paths_dict[model_key].append(model_file_path)

    for key in model_paths_dict.keys():
        model_paths_dict[key] = sorted(model_paths_dict[key])
    return model_paths_dict


def get_val_dataset_path_dict(val_dir_path, val_delim):
    val_dir_path = os.path.expanduser(val_dir_path)
    dataset_path_dict = dict()
    for file_name in os.listdir(val_dir_path):
        elements = file_name.split(val_delim)
        if len(elements) < 2:
            continue

        val_key = elements[0]
        dataset_path_dict[val_key] = os.path.join(val_dir_path, file_name)
    return dataset_path_dict


def print_missing_keys(model_paths_dict, val_dataset_path_dict):
    model_paths_key_set = set(model_paths_dict.keys())
    val_dataset_path_key_set = set(val_dataset_path_dict.keys())
    missing_model_key_set = val_dataset_path_key_set - model_paths_key_set
    print('[Missing model keys]')
    for model_key in missing_model_key_set:
        print(model_key)

    missing_val_key_set = model_paths_key_set - val_dataset_path_key_set
    print('[Missing val keys]')
    for val_key in missing_val_key_set:
        print(val_key)


def load_lambdified_model(model_file_path, num_variables, decrements_idx):
    if os.path.getsize(model_file_path) == 0:
        print(f'`File size of {str(model_file_path)}` is zero')
        return None, None

    with open(model_file_path, 'rb') as fp:
        sympy_eq = pickle.load(fp)
        new_variables = tuple([Symbol(f'x{i}') for i in range(num_variables)])
        if decrements_idx:
            old_variables = tuple([Symbol(f'x{i + 1}') for i in range(num_variables)])
            for old_variable, new_variable in zip(old_variables, new_variables):
                sympy_eq = sympy_eq.subs(old_variable, new_variable)

        variables = new_variables
        try:
            eq_func = lambdify(variables, sympy_eq, modules='numpy')
        except Exception as e:
            print(f'`{str(sympy_eq)}` has some issue {e}')
            return None, None
        return sympy_eq, lambda x: eq_func(*x).T


def select_models(model_paths_dict, val_dataset_path_dict, decrements_idx, output_dir_path):
    Path(output_dir_path).mkdir(parents=True, exist_ok=True)
    print_missing_keys(model_paths_dict, val_dataset_path_dict)
    print('\n[Selecting models...]')
    for val_dataset_key, val_dataset_file_path in val_dataset_path_dict.items():
        val_tabular_dataset = np.loadtxt(val_dataset_file_path)
        val_samples, val_targets = val_tabular_dataset[:, :-1], val_tabular_dataset[:, -1]
        num_variables = val_samples.shape[1]
        val_xs = tuple([x.T for x in np.hsplit(val_samples, val_samples.shape[1])])
        best_relative_error = np.inf
        best_model_file_path = None
        best_sympy_eq = None
        for model_file_path in model_paths_dict.get(val_dataset_key, list()):
            sympy_eq, eq_func = load_lambdified_model(model_file_path, num_variables, decrements_idx)
            if sympy_eq is None or sympy_eq == sympy.nan or sympy_eq.is_number:
                continue

            val_preds = eq_func(val_xs)
            val_preds = np.squeeze(val_preds)
            relative_error = (((val_targets - val_preds) / val_targets) ** 2).mean()
            if relative_error < best_relative_error:
                best_relative_error = relative_error
                best_model_file_path = model_file_path
                best_sympy_eq = sympy_eq

        if best_model_file_path is not None:
            print(f'{val_dataset_key}: best error {best_relative_error} achieved '
                  f'by {str(best_sympy_eq)} `{best_model_file_path}`')
            file_name = os.path.basename(best_model_file_path)
            copy_file_path = os.path.join(output_dir_path, file_name)
            copyfile(best_model_file_path, copy_file_path)
        else:
            print(f'No valid model file found for `{val_dataset_key}`')


def main(args):
    print(args)
    model_paths_dict = get_model_paths_dict(args.est, args.est_delim)
    val_dataset_path_dict = get_val_dataset_path_dict(args.val, args.val_delim)
    select_models(model_paths_dict, val_dataset_path_dict, args.dec_idx, os.path.expanduser(args.output))


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
