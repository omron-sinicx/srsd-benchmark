import argparse
import os
import pickle
from collections import OrderedDict
from multiprocessing import Process
from pathlib import Path

import pandas as pd
import sympy
from sympy import Symbol

from eq.conversion import sympy2zss_module
from eq.eval import count_nodes, compute_distance


def get_argparser():
    parser = argparse.ArgumentParser(description='Equation comparator')
    parser.add_argument('--est', required=True, help='file/dir path for pickled, estimated equation(s)')
    parser.add_argument('--gt', required=True, help='file/dir path for pickled, ground-truth equation(s)')
    parser.add_argument('--est_delim', default='.txt-est_eq', help='file name delimiter for estimated equation file(s)')
    parser.add_argument('--gt_delim', default='.pkl', help='file name delimiter for ground-truth equation file(s)')
    parser.add_argument('--eq_table', help='tsv file path to summarize equations')
    parser.add_argument('--dist_table', help='tsv file path to summarize distance')
    parser.add_argument('--correct_var_table',
                        help='tsv file path to summarize correct variables in estimated equations')
    parser.add_argument('--correct_var_count_table',
                        help='tsv file path to summarize numbers of correct variables in estimated equations')
    parser.add_argument('--dummy_var_table', help='tsv file path to summarize dummy variables in estimated equations')
    parser.add_argument('--dummy_var_count_table', help='tsv file path to summarize numbers of '
                                                        'dummy variables in estimated equations')
    parser.add_argument('--method_name', help='method name')
    parser.add_argument('-normalize', action='store_true', help='normalize distance by ground-truth equation')
    parser.add_argument('-dec_idx', action='store_true', help='decrement variable indices for estimated equation(s)')
    return parser


def get_est_gt_eq_pairs(est_eq_dir_path, est_delim, gt_eq_dir_path, gt_delim):
    est_gt_pair_list = list()
    est_eq_dict = {file_name.split(est_delim)[0]: os.path.join(est_eq_dir_path, file_name)
                   for file_name in os.listdir(est_eq_dir_path) if file_name.endswith('.pkl')}
    for gt_file_name in os.listdir(gt_eq_dir_path):
        if not gt_file_name.endswith('.pkl') or gt_delim not in gt_file_name:
            continue

        gt_file_path = os.path.join(gt_eq_dir_path, gt_file_name)
        gt_key = gt_file_name.split(gt_delim)[0]
        if gt_key in est_eq_dict:
            est_file_path = est_eq_dict.pop(gt_key)
            est_gt_pair_list.append((est_file_path, gt_file_path))
        else:
            est_gt_pair_list.append((None, gt_file_path))
    print(f'{len(est_gt_pair_list)} equation pairs matched')
    return est_gt_pair_list


def load_eq_as_tree(pickle_file_path, decrements_idx=False, prints=True):
    try:
        with open(pickle_file_path, 'rb') as fp, sympy.evaluate(False):
            eq_sympy = pickle.load(fp)
            if decrements_idx:
                old_variables = sorted(list(eq_sympy.free_symbols), key=lambda x: int(str(x)[1:]))
                var_indices = [int(str(var)[1:]) for var in old_variables]
                new_variables = tuple([Symbol(f'x{i - 1}') for i in var_indices])
                for old_variable, new_variable in zip(old_variables, new_variables):
                    eq_sympy = eq_sympy.subs(old_variable, new_variable)
        # Consistently reorder variables in args
        eq_sympy = sympy.sympify(str(eq_sympy))
        eq_sympy = eq_sympy.subs(sympy.pi, sympy.pi.evalf()).evalf().factor().simplify().subs(1.0, 1)
    except TypeError as te:
        if prints:
            print(te)
            print(f'[{pickle_file_path}]')
        return None, None
    except Exception as e:
        if prints:
            print(e)
            print(f'[{pickle_file_path}]')
        return None, None

    if prints:
        print(f'[{pickle_file_path}]')
        print(f'Eq.: {eq_sympy}')
    return sympy2zss_module(eq_sympy), eq_sympy


def compare_equation(est_eq_file_path, gt_eq_file_path, normalizes,
                     decrements_idx=False, prints=True, returns_eqs=False):
    gt_eq_tree, gt_eq = load_eq_as_tree(gt_eq_file_path, prints=prints)
    if est_eq_file_path is not None:
        p = Process(target=load_eq_as_tree, args=[est_eq_file_path, decrements_idx, False])
        p.start()
        p.join(timeout=120)
        p.terminate()
        if p.exitcode is None:
            print(f'Failed to load `{est_eq_file_path}`')
            edit_dist = 1 if normalizes else count_nodes(gt_eq_tree)
            est_eq = None
        else:
            est_eq_tree, est_eq = load_eq_as_tree(est_eq_file_path, decrements_idx=decrements_idx, prints=prints)
            if est_eq_tree is not None:
                edit_dist = compute_distance(est_eq_tree, gt_eq_tree, normalizes)
            else:
                edit_dist = 1 if normalizes else count_nodes(gt_eq_tree)
    else:
        est_eq = None
        edit_dist = 1 if normalizes else count_nodes(gt_eq_tree)
    if prints:
        edit_dist2print = str(edit_dist) if edit_dist is not None else 'N/A'
        print(f'Edit distance: {edit_dist2print}\n')
    if returns_eqs:
        num_gt_nodes = count_nodes(gt_eq_tree)
        return edit_dist, num_gt_nodes, est_eq, gt_eq
    return edit_dist


def create_data_frame(table_file_path, gt_eq_dict=None):
    if table_file_path is not None and os.path.exists(table_file_path):
        return pd.read_csv(table_file_path, sep='\t', index_col=0)
    if gt_eq_dict is None:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(gt_eq_dict)
    df = df.reindex(sorted(df.columns), axis=1)
    df.index = ['Ground Truth']
    return df


def expand_data_frame(base_df, sub_dict, method_name):
    sub_df = pd.DataFrame.from_dict(sub_dict)
    sub_df.index = [method_name]
    return pd.concat([base_df, sub_df])


def save_data_frame(df, output_file_path):
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file_path, sep='\t')


def compare_batch_equations(est_eq_dir_path, est_delim, gt_eq_dir_path, gt_delim, normalizes, decrements_idx,
                            eq_table_file_path, dist_table_file_path, dummy_var_table_file_path,
                            correct_var_table_file_path, correct_var_count_table_file_path,
                            dummy_var_count_table_file_path, method_name):
    est_gt_eq_pairs = get_est_gt_eq_pairs(est_eq_dir_path, est_delim, gt_eq_dir_path, gt_delim)
    est_eq_dict = OrderedDict()
    est_eq_correct_var_dict = OrderedDict()
    est_eq_dummy_var_dict = OrderedDict()
    gt_eq_dict = OrderedDict()
    dist_dict = OrderedDict()
    correct_var_count_dict = OrderedDict()
    dummy_var_count_dict = OrderedDict()
    total_edit_dist = 0
    for est_eq_file_path, gt_eq_file_path in est_gt_eq_pairs:
        edit_dist, num_gt_nodes, est_eq, gt_eq = \
            compare_equation(est_eq_file_path, gt_eq_file_path, normalizes,
                             decrements_idx=decrements_idx, prints=True, returns_eqs=True)

        gt_key = os.path.basename(gt_eq_file_path).split(gt_delim)[0]
        gt_eq_dict[gt_key] = [gt_eq]
        dist_dict[gt_key] = [edit_dist]
        true_var_set = {str(symbol) for symbol in gt_eq.free_symbols}
        est_var_set = set() if est_eq is None else {str(symbol) for symbol in est_eq.free_symbols}
        num_correct_vars = len(true_var_set.intersection(est_var_set))
        num_dummy_vars = len(est_var_set) - num_correct_vars
        correct_var_count_dict[gt_key] = [num_correct_vars]
        dummy_var_count_dict[gt_key] = [num_dummy_vars]
        if est_eq_file_path is None:
            est_eq_dict[gt_key] = [None]
            est_eq_dummy_var_dict[gt_key] = [None]
            est_eq_correct_var_dict[gt_key] = [None]
        else:
            est_key = os.path.basename(est_eq_file_path).split(est_delim)[0]
            assert est_key == gt_key
            est_eq_dict[gt_key] = [est_eq]
            est_eq_correct_var_dict[gt_key] = [' '.join(est_var_set & true_var_set)]
            est_eq_dummy_var_dict[gt_key] = [' '.join(est_var_set - true_var_set)]
        total_edit_dist += edit_dist if edit_dist is not None else num_gt_nodes

    mean_edit_dist = total_edit_dist / len(est_gt_eq_pairs)
    print(f'Mean edit distance: {mean_edit_dist}')
    eq_df = create_data_frame(eq_table_file_path, gt_eq_dict)
    eq_df = expand_data_frame(eq_df, est_eq_dict, method_name)
    eq_df = eq_df.reindex(sorted(eq_df.columns), axis=1)
    dist_df = create_data_frame(dist_table_file_path)
    dist_df = expand_data_frame(dist_df, dist_dict, method_name)
    dist_df = dist_df.reindex(sorted(dist_df.columns), axis=1)
    correct_var_df = create_data_frame(dummy_var_table_file_path, gt_eq_dict)
    correct_var_df = expand_data_frame(correct_var_df, est_eq_correct_var_dict, method_name)
    correct_var_df = correct_var_df.reindex(sorted(correct_var_df.columns), axis=1)
    correct_var_count_df = create_data_frame(dummy_var_count_table_file_path)
    correct_var_count_df = expand_data_frame(correct_var_count_df, correct_var_count_dict, method_name)
    correct_var_count_df = correct_var_count_df.reindex(sorted(correct_var_count_df.columns), axis=1)
    dummy_var_df = create_data_frame(dummy_var_table_file_path, gt_eq_dict)
    dummy_var_df = expand_data_frame(dummy_var_df, est_eq_dummy_var_dict, method_name)
    dummy_var_df = dummy_var_df.reindex(sorted(dummy_var_df.columns), axis=1)
    dummy_var_count_df = create_data_frame(dummy_var_count_table_file_path)
    dummy_var_count_df = expand_data_frame(dummy_var_count_df, dummy_var_count_dict, method_name)
    dummy_var_count_df = dummy_var_count_df.reindex(sorted(dummy_var_count_df.columns), axis=1)
    print(eq_df)
    print(dist_df)
    print(dummy_var_df)
    print(dummy_var_count_df)
    if eq_table_file_path is not None:
        save_data_frame(eq_df, eq_table_file_path)
    if dist_table_file_path is not None:
        save_data_frame(dist_df, dist_table_file_path)
    if correct_var_table_file_path is not None:
        save_data_frame(correct_var_df, correct_var_table_file_path)
    if correct_var_count_table_file_path is not None:
        save_data_frame(correct_var_count_df, correct_var_count_table_file_path)
    if dummy_var_table_file_path is not None:
        save_data_frame(dummy_var_df, dummy_var_table_file_path)
    if dummy_var_count_table_file_path is not None:
        save_data_frame(dummy_var_count_df, dummy_var_count_table_file_path)


def main(args):
    print(args)
    est_path = os.path.expanduser(args.est)
    gt_path = os.path.expanduser(args.gt)
    eq_table_file_path = os.path.expanduser(args.eq_table) if args.eq_table is not None else None
    dist_table_file_path = os.path.expanduser(args.dist_table) if args.dist_table is not None else None
    correct_var_table_file_path = \
        os.path.expanduser(args.correct_var_table) if args.correct_var_table is not None else None
    correct_var_count_table_file_path = \
        os.path.expanduser(args.correct_var_count_table) if args.correct_var_count_table is not None else None
    dummy_var_table_file_path = os.path.expanduser(args.dummy_var_table) if args.dummy_var_table is not None else None
    dummy_var_count_table_file_path = os.path.expanduser(args.dummy_var_count_table) \
        if args.dummy_var_count_table is not None else None
    if os.path.isfile(est_path) and os.path.isfile(gt_path):
        compare_equation(est_path, args.gt, args.normalize, args.dec_idx)
    elif os.path.isdir(est_path) and os.path.isdir(gt_path):
        compare_batch_equations(est_path, args.est_delim, gt_path, args.gt_delim, args.normalize, args.dec_idx,
                                eq_table_file_path, dist_table_file_path, dummy_var_table_file_path,
                                correct_var_table_file_path, correct_var_count_table_file_path,
                                dummy_var_count_table_file_path, args.method_name)
    else:
        raise ValueError('--est and --gt should be either both file paths or both dir paths')


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
