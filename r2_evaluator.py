import argparse
import os
import pickle

import numpy as np
import sympy
from sklearn.metrics import r2_score
from sympy import Symbol, lambdify


def get_argparser():
    parser = argparse.ArgumentParser(description='R2 score-based evaluator')
    parser.add_argument('--est', required=True, help='dir path for pickled, estimated equations')
    parser.add_argument('--gt', help='dir path for pickled, ground-truth equation(s)')
    parser.add_argument('--test', required=True, help='test dataset dir path')
    parser.add_argument('--est_delim', default='.txt', help='file name delimiter for estimated equation files')
    parser.add_argument('--gt_delim', default='.pkl', help='file name delimiter for ground-truth equation file(s)')
    parser.add_argument('--test_delim', default='.txt', help='file name delimiter for test dataset files')
    parser.add_argument('-dec_idx', action='store_true', help='decrement variable indices for estimated equation(s)')
    parser.add_argument('--r2thr', default=0.999, type=float, help='R2 score threshold')
    return parser


def get_model_path_dict(est_dir_path, est_delim):
    model_path_dict = dict()
    for file_name in os.listdir(os.path.expanduser(est_dir_path)):
        elements = file_name.split(est_delim)
        if len(elements) < 2:
            continue

        model_file_path = os.path.join(est_dir_path, file_name)
        model_key = elements[0]
        model_path_dict[model_key] = os.path.expanduser(model_file_path)
    return model_path_dict


def get_true_eq_path_dict(gt_dir_path, gt_delim):
    if gt_dir_path is None or gt_delim is None:
        return None

    true_eq_path_dict = dict()
    for file_name in os.listdir(os.path.expanduser(gt_dir_path)):
        elements = file_name.split(gt_delim)
        if len(elements) < 2:
            continue

        true_eq_file_path = os.path.join(gt_dir_path, file_name)
        gt_key = elements[0]
        true_eq_path_dict[gt_key] = os.path.expanduser(true_eq_file_path)
    return true_eq_path_dict


def get_test_dataset_path_dict(test_dir_path, test_delim):
    test_dir_path = os.path.expanduser(test_dir_path)
    dataset_path_dict = dict()
    for file_name in os.listdir(test_dir_path):
        elements = file_name.split(test_delim)
        if len(elements) < 2:
            continue

        test_key = elements[0]
        dataset_path_dict[test_key] = os.path.join(test_dir_path, file_name)
    return dataset_path_dict


def print_missing_keys(model_paths_dict, test_dataset_path_dict):
    model_paths_key_set = set(model_paths_dict.keys())
    test_dataset_path_key_set = set(test_dataset_path_dict.keys())
    missing_model_key_set = test_dataset_path_key_set - model_paths_key_set
    print('[Missing model keys]')
    for model_key in missing_model_key_set:
        print(model_key)

    missing_test_key_set = model_paths_key_set - test_dataset_path_key_set
    print('[Missing test keys]')
    for test_key in missing_test_key_set:
        print(test_key)


def load_lambdified_model(model_file_path, num_variables, decrements_idx):
    if model_file_path is None:
        return None, None

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
        return sympy_eq, lambda x: eq_func(*x).T if not sympy_eq.is_number else eq_func(*x)


def evaluate(model_path_dict, gt_eq_path_dict, test_dataset_path_dict, decrements_idx, r2_score_threshold):
    print_missing_keys(model_path_dict, test_dataset_path_dict)
    r2_score_list = list()
    correct_flag_list = list()
    total_dummy_var_count = 0
    total_dummy_var_eq_count = 0
    total_dummy_var_eq_correct_count = 0
    total_pred_eq_count = 0
    print('\n[Evaluating models...]')
    for test_dataset_key, test_dataset_file_path in test_dataset_path_dict.items():
        test_tabular_dataset = np.loadtxt(test_dataset_file_path)
        test_samples, test_targets = test_tabular_dataset[:, :-1], test_tabular_dataset[:, -1]
        num_variables = test_samples.shape[1]
        test_xs = tuple([x.T for x in np.hsplit(test_samples, test_samples.shape[1])])
        model_file_path = model_path_dict.get(test_dataset_key, None)
        sympy_eq, eq_func = load_lambdified_model(model_file_path, num_variables, decrements_idx)
        if sympy_eq is None or sympy_eq == sympy.nan:
            print(f'No valid model file found for `{test_dataset_key}`')
            r2_score_list.append(None)
            correct_flag_list.append(False)
            continue

        if sympy_eq.is_number:
            test_preds = np.ones(len(test_targets)) * float(sympy_eq)
            score = r2_score(test_targets, test_preds)
        else:
            test_preds = eq_func(test_xs)
            test_preds = np.squeeze(test_preds)
            try:
                score = r2_score(test_targets, test_preds)
            except:
                score = -np.inf

        print(f'R2 score: {score}')
        r2_score_list.append(score)
        is_correct = score > r2_score_threshold
        correct_flag_list.append(is_correct)
        if gt_eq_path_dict is not None:
            pickle_file_path = gt_eq_path_dict[test_dataset_key]
            with open(pickle_file_path, 'rb') as fp, sympy.evaluate(False):
                gt_eq = pickle.load(fp)
                true_var_set = {str(symbol) for symbol in gt_eq.free_symbols}
                est_var_set = {str(symbol) for symbol in sympy_eq.free_symbols}
                dummy_var_count = len(est_var_set - true_var_set)
                total_dummy_var_count += dummy_var_count
                if dummy_var_count > 0:
                    total_dummy_var_eq_count += 1
                    if is_correct:
                        total_dummy_var_eq_correct_count += 1
        total_pred_eq_count += 1

    num_correct_flags = sum(correct_flag_list)
    num_samples = len(correct_flag_list)
    accuracy = num_correct_flags / num_samples * 100
    print(f'Accuracy (R2 > {r2_score_threshold}): {accuracy}% ({num_correct_flags} / {num_samples})')
    if gt_eq_path_dict is not None:
        print(f'Total number of dummy variables used in prediction: {total_dummy_var_count}')
        print(f'Total number of predictions that use at least one dummy variable: {total_dummy_var_eq_count}')
        print(f'Total number of predictions: {total_pred_eq_count}')
        dummy_var_eq_rate = total_dummy_var_eq_count / total_pred_eq_count * 100.0 if total_pred_eq_count > 0 else 'N/A'
        print(f'Percentage of non-zero predictions that use at least one dummy variable: {dummy_var_eq_rate}%')
        accuracy_w_dummy = \
            total_dummy_var_eq_correct_count / num_correct_flags * 100.0 if num_correct_flags > 0 else 'N/A'
        print(f'Total number of correct predictions that use at least one dummy variable: '
              f'{total_dummy_var_eq_correct_count}')
        print(f'Total number of correct predictions: {num_correct_flags}')
        print(f'Percentage of "correct" predictions that use at least one dummy variable: '
              f'{accuracy_w_dummy}%\n')


def main(args):
    print(args)
    model_path_dict = get_model_path_dict(args.est, args.est_delim)
    gt_eq_path_dict = get_true_eq_path_dict(args.gt, args.gt_delim)
    test_dataset_path_dict = get_test_dataset_path_dict(args.test, args.test_delim)
    evaluate(model_path_dict, gt_eq_path_dict, test_dataset_path_dict, args.dec_idx, args.r2thr)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
