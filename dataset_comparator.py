import argparse
import os
import pickle
from multiprocessing import Process, Manager

import numpy as np
from torchdistill.common.file_util import get_file_path_list

from eq.conversion import sympy2sequence, sympy2zss_module
from eq.eval import compute_distances


def get_argparser():
    parser = argparse.ArgumentParser(description='Dataset comparator')
    parser.add_argument('--src_eq', required=True, help='source equation dir path')
    parser.add_argument('--src_tabular', required=True, help='source tabular data dir path')
    parser.add_argument('--src_delim', default=' ', help='delimiter in src data')
    parser.add_argument('--dst_eq', required=True, help='destination equation dir path')
    parser.add_argument('--dst_tabular', required=True, help='destination tabular data dir path')
    parser.add_argument('--dst_delim', default=' ', help='destination in src data')
    parser.add_argument('--mode', default='intersection', help='comparison mode')
    return parser


def load_sympy_equation(file_path):
    with open(file_path, 'rb') as fp:
        eq_sympy = pickle.load(fp)
        return eq_sympy.evalf()


def compute_stats(eqs, message):
    print(f'\n{message}')
    seq_length_list = list()
    for eq in eqs:
        eq_tree_sequence = sympy2sequence(eq, returns_binary_tree=True)
        seq_length_list.append(len(eq_tree_sequence))

    seq_lengths = np.array(seq_length_list)
    print(f'{len(eqs)} equations')
    print('Sequence length')
    print(f'Min: {seq_lengths.min()}')
    print(f'Max: {seq_lengths.max()}')
    print(f'Mean: {seq_lengths.mean()}')
    print(f'Median: {np.median(seq_lengths)}')


def load_tabular_dataset(file_path, delimiter):
    mat = np.loadtxt(file_path, delimiter=delimiter)
    return mat


def extract_dataset_file_paths_w_identical_equation(src_eq, dst_eq_trees, dst_tabular_dataset_file_paths):
    src_eq_tree = sympy2zss_module(src_eq.evalf())
    edit_dists = compute_distances([src_eq_tree] * len(dst_eq_trees), dst_eq_trees, normalizes=True)
    edit_dists = np.array(edit_dists)
    complete_matches = edit_dists == 0.0
    if not np.any(complete_matches):
        return None
    return np.array(dst_tabular_dataset_file_paths)[complete_matches].tolist()


def extract_min_max_input_variables(tabular_dataset):
    # The last column is output, thus skipped
    sample_min_values = tabular_dataset[:, :-1].min(0)
    sample_max_values = tabular_dataset[:, :-1].max(0)
    return sample_min_values, sample_max_values


def find_overlapped_wrt_domains(src_tabular_dataset_file_path, src_delimiter,
                                dst_tabular_dataset_file_paths, dst_delimiter):
    src_tabular_dataset = load_tabular_dataset(src_tabular_dataset_file_path, src_delimiter)
    src_sample_min_values, src_sample_max_values = extract_min_max_input_variables(src_tabular_dataset)
    overlapped_list = list()
    for dst_tabular_dataset_file_path in dst_tabular_dataset_file_paths:
        dst_tabular_dataset = load_tabular_dataset(dst_tabular_dataset_file_path, dst_delimiter)
        dst_sample_min_values, dst_sample_max_values = extract_min_max_input_variables(dst_tabular_dataset)
        overlapped = (src_sample_min_values <= dst_sample_min_values) * (dst_sample_min_values <= src_sample_max_values)
        overlapped += \
            (src_sample_min_values <= dst_sample_max_values) * (dst_sample_max_values <= src_sample_max_values)
        overlapped += \
            (dst_sample_min_values <= src_sample_min_values) * (src_sample_max_values <= dst_sample_max_values)
        if np.all(overlapped):
            overlapped_list.append(dst_tabular_dataset_file_path)

    num_overlapped_datasets = len(overlapped_list)
    if num_overlapped_datasets > 0:
        print(f'{src_tabular_dataset_file_path}: {num_overlapped_datasets} overlapped datasets')
    return overlapped_list


def check_if_overlapped(src_tabular_dataset_file_path, src_eq, src_delimiter,
                        dst_eq_trees, dst_tabular_dataset_file_paths, dst_delimiter, overlapped_dict):
    filtered_dst_file_paths = \
        extract_dataset_file_paths_w_identical_equation(src_eq, dst_eq_trees, dst_tabular_dataset_file_paths)
    if filtered_dst_file_paths is None or len(filtered_dst_file_paths) == 0:
        return

    overlapped_dst_tabular_dataset_file_paths = \
        find_overlapped_wrt_domains(src_tabular_dataset_file_path, src_delimiter,
                                    filtered_dst_file_paths, dst_delimiter)
    overlapped_dict[src_tabular_dataset_file_path] = overlapped_dst_tabular_dataset_file_paths


def compute_intersections(src_sample_min_values, src_sample_max_values, dst_sample_min_values, dst_sample_max_values):
    numerator = \
        np.max((src_sample_min_values, dst_sample_min_values), axis=0) \
        - np.min((src_sample_max_values, dst_sample_max_values), axis=0)
    denominator = \
        np.min((src_sample_min_values, dst_sample_min_values), axis=0) \
        - np.max((src_sample_max_values, dst_sample_max_values), axis=0)
    return numerator / denominator


def compute_domain_intersections(src_tabular_dataset_file_path, src_delimiter,
                                 dst_tabular_dataset_file_paths, dst_delimiter):
    src_tabular_dataset = load_tabular_dataset(src_tabular_dataset_file_path, src_delimiter)
    src_sample_min_values, src_sample_max_values = extract_min_max_input_variables(src_tabular_dataset)
    intersections_list = list()
    for dst_tabular_dataset_file_path in dst_tabular_dataset_file_paths:
        dst_tabular_dataset = load_tabular_dataset(dst_tabular_dataset_file_path, dst_delimiter)
        if dst_tabular_dataset.shape[1] != src_tabular_dataset.shape[1]:
            continue

        dst_sample_min_values, dst_sample_max_values = extract_min_max_input_variables(dst_tabular_dataset)
        domain_intersections = compute_intersections(src_sample_min_values, src_sample_max_values,
                                                     dst_sample_min_values, dst_sample_max_values)
        intersections_list.append(domain_intersections)
    return np.array(intersections_list).mean(axis=0) if len(intersections_list) > 0 else None


def assess_domain_intersections_if_overlapped(src_tabular_dataset_file_path, src_eq, src_delimiter,
                                              dst_eq_trees, dst_tabular_dataset_file_paths, dst_delimiter, result_dict):
    filtered_dst_file_paths = \
        extract_dataset_file_paths_w_identical_equation(src_eq, dst_eq_trees, dst_tabular_dataset_file_paths)
    if filtered_dst_file_paths is None or len(filtered_dst_file_paths) == 0:
        return

    intersection_mat = compute_domain_intersections(src_tabular_dataset_file_path, src_delimiter,
                                                    dst_tabular_dataset_file_paths, dst_delimiter)
    if intersection_mat is not None:
        result_dict[src_tabular_dataset_file_path] = intersection_mat


def compare_datasets(src_eqs, src_tabular_dataset_file_paths, src_delimiter,
                     dst_eqs, dst_tabular_dataset_file_paths, dst_delimiter, mode):
    compute_stats(src_eqs, '[Source Equations]')
    compute_stats(dst_eqs, '[Target Equations]')
    dst_eq_trees = [sympy2zss_module(dst_eq.evalf()) for dst_eq in dst_eqs]
    process_list = list()
    manager = Manager()
    result_dict = manager.dict()
    for src_eq, src_tabular_dataset_file_path in zip(src_eqs, src_tabular_dataset_file_paths):
        p = Process(
            target=check_if_overlapped if mode == 'overlap' else assess_domain_intersections_if_overlapped,
            args=(src_tabular_dataset_file_path, src_eq, src_delimiter,
                  dst_eq_trees, dst_tabular_dataset_file_paths, dst_delimiter, result_dict)
        )
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()

    if mode == 'overlap':
        print(f'{len(result_dict)} overlapped source datasets')
        overlapped_list = list()
        for sub_list in result_dict.values():
            overlapped_list.extend(sub_list)

        print(f'{len(overlapped_list)} overlapped destination datasets')
        unique_overlapped_set = set(overlapped_list)
        unique_size = len(unique_overlapped_set)
        print(f'{unique_size} ({unique_size / len(dst_eqs) * 100}%) unique overlapped destination datasets')
    elif mode == 'intersection':
        num_keys = len(result_dict)
        mean_results = np.array([r.mean() for r in result_dict.values()])
        print(f'{num_keys} source equations match some destination ones')
        average_results_over_matched_src = mean_results.mean()
        print(f'Intersection averaged over matched src equations: {average_results_over_matched_src}')
        average_results_over_all_src = mean_results.sum() / len(src_eqs)
        print(f'Intersection averaged over all src equations: {average_results_over_all_src}')


def main(args):
    print(args)
    src_eq_file_paths = get_file_path_list(os.path.expanduser(args.src_eq), is_sorted=True)
    src_tabular_dataset_file_paths = get_file_path_list(os.path.expanduser(args.src_tabular), is_sorted=True)
    assert len(src_eq_file_paths) == len(src_tabular_dataset_file_paths)
    src_eqs = [load_sympy_equation(file_path) for file_path in src_eq_file_paths]
    dst_eq_file_paths = get_file_path_list(os.path.expanduser(args.dst_eq), is_sorted=True)
    dst_tabular_dataset_file_paths = get_file_path_list(os.path.expanduser(args.dst_tabular), is_sorted=True)
    assert len(dst_eq_file_paths) == len(dst_tabular_dataset_file_paths)
    dst_eqs = [load_sympy_equation(file_path) for file_path in dst_eq_file_paths]
    compare_datasets(src_eqs, src_tabular_dataset_file_paths, args.src_delim,
                     dst_eqs, dst_tabular_dataset_file_paths, args.dst_delim, args.mode)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
