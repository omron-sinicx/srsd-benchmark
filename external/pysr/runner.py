import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import sympy
import yaml
from pysr import PySRRegressor


def get_argparser():
    parser = argparse.ArgumentParser(description='PySR baseline runner')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--train', required=True, help='training file path')
    parser.add_argument('--eq', required=True, help='output equation file path')
    parser.add_argument('--table', required=True, help='output table file path')
    return parser


def load_dataset(dataset_file_path, delimiter=' '):
    tabular_dataset = np.loadtxt(dataset_file_path, delimiter=delimiter)
    return tabular_dataset[:, :-1], tabular_dataset[:, -1]


def main(args):
    print(args)
    with open(os.path.expanduser(args.config), 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    train_samples, train_targets = load_dataset(os.path.expanduser(args.train))
    model = PySRRegressor(**config['fit'])
    model.fit(train_samples, train_targets)
    output_eq_path = os.path.expanduser(args.eq)
    Path(output_eq_path).parent.mkdir(parents=True, exist_ok=True)
    output_table_path = os.path.expanduser(args.table)
    Path(output_table_path).parent.mkdir(parents=True, exist_ok=True)
    result_df = model.equations_
    result_df.to_csv(output_table_path, sep='\t')
    best_sympy_eq = \
        sympy.sympify(result_df.loc[result_df['score'] == result_df['score'].max()]['sympy_format'].tolist()[0])
    with open(output_eq_path, 'wb') as fp:
        pickle.dump(best_sympy_eq, fp)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
