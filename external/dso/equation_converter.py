import argparse
import json
import os
import pickle
from pathlib import Path

import pandas
import sympy
from sympy import Symbol


def get_argparser():
    parser = argparse.ArgumentParser(description='Equation converter for DSO')
    parser.add_argument('--summary', required=True, help='summary file/dir path')
    parser.add_argument('--out', required=True, help='output file/dir name')
    return parser


def extract_dataset_file_path(config_file_path):
    with open(config_file_path, 'r') as fp:
        config = json.load(fp)
    return config['task']['dataset']


def save_obj(obj, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as fp:
        pickle.dump(obj, fp)


def decrement_index(match_obj):
    variable_str = match_obj.group()
    index = variable_str[1:]
    return variable_str[0] + str(int(index) - 1)


def dso2sympy(eq_str):
    # variable index in DSO starts from 1
    eq_sympy = sympy.sympify(eq_str)
    old_variables = sorted(list(eq_sympy.free_symbols), key=lambda x: int(str(x)[1:]))
    var_indices = [int(str(var)[1:]) for var in old_variables]
    new_variables = tuple([Symbol(f'x{i - 1}') for i in var_indices])
    for old_variable, new_variable in zip(old_variables, new_variables):
        eq_sympy = eq_sympy.subs(old_variable, new_variable)
    return eq_sympy


def convert_summary2eq(summary_file_path, output_file_path):
    print(f'Processing `{summary_file_path}`')
    df = pandas.read_csv(os.path.expanduser(summary_file_path))
    eq_str = str(df['expression'][0])
    eq_sympy = dso2sympy(eq_str)
    save_obj(eq_sympy, os.path.expanduser(output_file_path))


def convert_summaries2eqs(root_dir_path, output_dir_path):
    root_dir_path = os.path.expanduser(root_dir_path)
    output_dir_path = os.path.expanduser(output_dir_path)
    summary_dir_names = os.listdir(root_dir_path)
    for summary_dir_name in summary_dir_names:
        summary_file_path = os.path.join(root_dir_path, summary_dir_name, 'summary.csv')
        if not os.path.isfile(summary_file_path):
            continue

        config_file_path = os.path.join(root_dir_path, summary_dir_name, 'config.json')
        if not os.path.isfile(config_file_path):
            continue

        dataset_file_path = extract_dataset_file_path(config_file_path)
        dataset_file_name = os.path.basename(dataset_file_path)
        output_file_path = os.path.join(output_dir_path, dataset_file_name + '.pkl')
        convert_summary2eq(summary_file_path, output_file_path) 


def main(args):
    print(args)
    if os.path.isfile(args.summary):
        convert_summary2eq(args.summary, args.out)
    elif os.path.isdir(args.summary):
        convert_summaries2eqs(args.summary, args.out)
    else:
        raise ValueError('--summary should be either a file path or a dir path')


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
