import argparse
import os

from torchdistill.common.file_util import make_parent_dirs

from datasets.feynman import FEYNMAN_EQUATION_CLASS_DICT
from datasets.registry import EQUATION_CLASS_DICT


def get_argparser():
    parser = argparse.ArgumentParser(description='Equation analyzer with sympy')
    parser.add_argument('--name', choices=['all', 'feynman'], help='equation group name')
    parser.add_argument('--eq_ids', nargs='+', help='equation IDs for analysis (all if None is given)')
    parser.add_argument('--output', help='output file/dir path')
    parser.add_argument('--complexity', help='output file path for complexity analysis results')
    parser.add_argument('-simple_check', action='store_true', help='check equation properties')
    parser.add_argument('-visualize', action='store_true', help='visualize equation tree')
    parser.add_argument('-find_stationary', action='store_true', help='find stationary points of equation')
    parser.add_argument('-exclude_saddle_points', action='store_true', help='exclude saddle points')
    return parser


def get_equation_dict(group_name):
    if group_name == 'all':
        return EQUATION_CLASS_DICT.copy()
    elif group_name == 'feynman':
        return FEYNMAN_EQUATION_CLASS_DICT.copy()


def check_equation_property(eq_instance, eq_name_set):
    passed = True
    if not eq_instance.check_num_vars_consistency(debug=True):
        passed = False
    
    eq_name = eq_instance.get_eq_name()
    if eq_name in eq_name_set or eq_name is None or len(eq_name) == 0:
        print('eq_name `{}` is not unique in the specified equation group')
        passed = False
    return passed


def categorize_dataset(complexity, domain_range):
    if complexity < 5 and domain_range < 4:
        return 0
    elif complexity < 8 and domain_range < 20:
        return 1
    return 2


def write_tsv_file(list_of_lists, output_file_path):
    make_parent_dirs(output_file_path)
    with open(output_file_path, 'w') as fp:
        for values in list_of_lists:
            fp.write('\t'.join(values) + '\n')


def analyze(equation_dict, simple_check, visualizes, finds_stationary,
            excludes_saddle_points, output_path, comp_output_file_path, **kwargs):
    eq_name_set = set()
    incomplete_eq_list = list()
    complexity_list = list()
    for eq_id, eq_cls in equation_dict.items():
        print(f'\nEquation ID: {eq_id}')
        eq_instance = eq_cls()
        print(f'f(x) = {eq_instance.sympy_eq}')
        num_vars = eq_instance.get_var_count()
        print(f'Number of variables: {num_vars}')
        num_ops = eq_instance.get_op_count()
        print(f'Number of operations: {num_ops}')
        domain_range = eq_instance.get_domain_range()
        print(f'Domain range: {domain_range}')
        dataset_category = categorize_dataset(num_ops, domain_range)
        print(f'Dataset category: {dataset_category}')
        complexity_list.append((eq_id, str(eq_instance.sympy_eq), str(num_vars), str(num_ops),
                                str(domain_range), str(dataset_category)))
        if simple_check:
            passed = check_equation_property(eq_instance, eq_name_set)
            message = 'PASSED' if passed else 'FAILED'
            print(message)
            if not passed:
                incomplete_eq_list.append(eq_instance.get_eq_name())
        
        if visualizes:
            output_file_path = os.path.join(output_path, eq_instance.get_eq_name()) if output_path is not None else None
            eq_instance.visualize_tree(output_file_path, ext='png')
        
        if finds_stationary:
            stationary_points = eq_instance.find_stationary_points(excludes_saddle_points)
            print(f'stationary point(s): {stationary_points}')
        eq_name_set.add(eq_instance.get_eq_name())

    print(f'\n{len(equation_dict)} unique equations')
    print(f'{len(eq_name_set)} unique equation names')
    if simple_check:
        print(f'incomplete equations: {incomplete_eq_list}')
    
    if comp_output_file_path is not None:
        complexity_list.insert(0, ('eq_id', 'eq', 'num_vars', 'num_ops', 'domain_range', 'category'))
        write_tsv_file(complexity_list, comp_output_file_path)


def main(args):
    print(args)
    group_name = args.name
    eq_ids = args.eq_ids
    equation_dict = get_equation_dict(group_name)
    output_path = args.output
    comp_output_file_path = args.complexity
    simple_check = args.simple_check
    visualizes = args.visualize
    finds_stationary = args.find_stationary
    excludes_saddle_points = args.exclude_saddle_points
    if eq_ids is None or len(eq_ids) == 0:
        analyze(equation_dict, simple_check, visualizes, finds_stationary,
                excludes_saddle_points, output_path, comp_output_file_path)
    else:
        sub_equation_dict = {eq_id: EQUATION_CLASS_DICT[eq_id] for eq_id in eq_ids}
        analyze(sub_equation_dict, simple_check, visualizes, finds_stationary,
                excludes_saddle_points, output_path, comp_output_file_path)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
