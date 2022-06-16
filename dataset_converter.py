import argparse
import os
from random import shuffle

import numpy as np
from torchdistill.common.file_util import get_file_path_list, make_parent_dirs


def get_argparser():
    parser = argparse.ArgumentParser(description='Dataset converter')
    parser.add_argument('--src', required=True, help='source file/dir path')
    parser.add_argument('--dst', required=True, help='destination file/dir path')
    parser.add_argument('--sample', help='number of samples (if int) or sampling rate (if float between 0 and 1)')
    parser.add_argument('--type', default='tabular',  choices=['tabular', 'json'], help='type of src data')
    parser.add_argument('--src_delim', help='delimiter in src data')
    parser.add_argument('--dst_delim', default=',', help='delimiter for dst data')
    parser.add_argument('--dst_ext', default='.csv', help='extension for dst file path when src dir path is given')
    return parser


def load_tabular_dataset(file_path, delimiter):
    mat = np.loadtxt(file_path, delimiter=delimiter)
    return mat


def subsample_dataset(tabular_dataset, sampling):
    num_samples = sampling if isinstance(sampling, int) else int(len(tabular_dataset) * sampling)
    indices = list(range(len(tabular_dataset)))
    shuffle(indices)
    return tabular_dataset[indices]


def write_tabular_dataset(tabular_dataset, delimiter, output_file_path):
    make_parent_dirs(output_file_path)
    np.savetxt(output_file_path, tabular_dataset, delimiter=delimiter)


def main(args):
    print(args)
    dataset_type = args.type
    src_file_path = os.path.expanduser(args.src)
    dst_file_path = os.path.expanduser(args.dst)
    sampling = args.sample
    if sampling is not None:
        if sampling.isdigit():
            sampling = int(sampling)
        else:
            sampling = float(sampling)

    src_delimiter = args.src_delim
    dst_delimiter = args.dst_delim
    if dataset_type == 'tabular':
        if os.path.isfile(src_file_path):
            src_dataset = load_tabular_dataset(src_file_path, src_delimiter)
            dst_dataset = src_dataset if sampling is None else subsample_dataset(src_dataset, sampling)
            write_tabular_dataset(dst_dataset, dst_delimiter, dst_file_path)
        elif os.path.isdir(src_file_path):
            src_dir_path = src_file_path
            dst_dir_path = dst_file_path
            dst_ext = args.dst_ext
            if dst_ext == '.tsv':
                dst_delimiter = '\t'
            for src_file_path in get_file_path_list(src_dir_path):
                src_dataset = load_tabular_dataset(src_file_path, src_delimiter)
                dst_dataset = src_dataset if sampling is None else subsample_dataset(src_dataset, sampling)
                dst_file_path = os.path.join(dst_dir_path, os.path.basename(src_file_path) + dst_ext)
                write_tabular_dataset(dst_dataset, dst_delimiter, dst_file_path)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
