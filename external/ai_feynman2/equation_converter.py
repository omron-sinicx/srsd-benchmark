import argparse
import os
import pickle
from pathlib import Path

import sympy


def get_argparser():
    parser = argparse.ArgumentParser(description='Equation converter for AI Feynman')
    parser.add_argument('--solution', required=True, help='solution file path or prefix')
    parser.add_argument('--out', required=True, help='output file/dir name')
    return parser


def save_obj(obj, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as fp:
        pickle.dump(obj, fp)


def ai_feynman2sympy(eq_str):
    return sympy.sympify(eq_str)


def convert_summary2eq(solution_file_path, output_file_path):
    print(f'Processing `{solution_file_path}`')
    with open(solution_file_path, 'r') as fp:
        lines = [line.strip() for line in fp]

    elements = lines[-1].split(' ')
    eq_sympy = ai_feynman2sympy(''.join(elements[5:]))
    save_obj(eq_sympy, output_file_path)


def convert_summaries2eqs(solution_file_path_prefix, output_dir_path):
    parent_dir_path = str(Path(solution_file_path_prefix).parent)
    result_file_names = os.listdir(parent_dir_path)
    for result_file_name in result_file_names:
        result_file_path = os.path.join(parent_dir_path, result_file_name)
        if not result_file_path.startswith(solution_file_path_prefix):
            continue

        output_file_name = result_file_name[len('solution_'):]
        output_file_path = os.path.join(output_dir_path, output_file_name + '.pkl')
        convert_summary2eq(result_file_path, output_file_path) 


def main(args):
    print(args)
    if os.path.isfile(args.solution):
        convert_summary2eq(os.path.expanduser(args.solution), os.path.expanduser(args.out))
    else:
        convert_summaries2eqs(os.path.expanduser(args.solution), os.path.expanduser(args.out))


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
