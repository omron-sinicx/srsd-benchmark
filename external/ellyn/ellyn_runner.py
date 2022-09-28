import argparse
import os
import pickle
import timeit
import re
from pathlib import Path
import sympy

import numpy as np
import optuna
import yaml

from ellyn import ellyn

STR2SYMPY = {
    'neg': lambda x: -x,
    'abs': sympy.Abs,
    'sqrt': sympy.sqrt,
    'exp': sympy.exp,
    'log': sympy.log,
    'sin': sympy.sin,
    'cos': sympy.cos,
    'tan': sympy.tan,
    'asin': sympy.asin,
    'acos': sympy.acos,
    'atan': sympy.atan,
    'add': lambda x, y : x + y,
    'sub': lambda x, y : x - y,
    'mul': lambda x, y : x * y,
    'div': lambda x, y : x / y,
    'pow': lambda x, y : x ** y
}


def get_argparser():
    parser = argparse.ArgumentParser(description='Ellyn baseline runner')
    parser.add_argument('--config', required=True, help='yaml config file path')
    parser.add_argument('--train', help='training file path')
    parser.add_argument('--val', help='training file path')
    parser.add_argument('--test', required=True, help='test file path')
    parser.add_argument('--out', required=True, help='output file name (dir path should be specified in config file)')
    parser.add_argument('-test_only', action='store_true', help='skip the training phase')
    return parser


def load_dataset(dataset_file_path, delimiter=' '):
    tabular_dataset = np.loadtxt(dataset_file_path, delimiter=delimiter)
    return tabular_dataset[:, :-1], tabular_dataset[:, -1]


def evaluate(model, eval_samples, eval_targets, eval_type='Validation', quiet=False):
    pred_equation = str(model.stack_2_eqn(model.best_estimator_))
    pred_equation = re.sub(r'\bx_([0-9]*[0-9])\b', r'x\1', pred_equation)
    eval_preds = model.predict(eval_samples)
    relative_error = (((eval_targets - eval_preds) / eval_targets) ** 2).mean()
    if not quiet:
        print(f'\n[{eval_type}]')
        print(f'Equation: {pred_equation}')
        print(f'Relative error: {relative_error}')
    return relative_error


def save_obj(obj, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as fp:
        pickle.dump(obj, fp)


def ellyn2sympy(eq_str):
    eq_str_w_normalized_vars = re.sub(r'\bx_([0-9]*[0-9])\b', r'x\1', eq_str)
    return sympy.sympify(eq_str_w_normalized_vars, locals=STR2SYMPY)


def train(train_file_path, val_file_path, model_config, output_file_prefix):
    train_samples, train_targets = load_dataset(train_file_path)
    val_samples, val_targets = load_dataset(val_file_path)

    model = ellyn(**model_config['kwargs'])
    start_time = timeit.default_timer()
    model.fit(train_samples, train_targets)
    train_time = timeit.default_timer() - start_time

    print(f'Training time: {train_time}')
    pickle_file_path = os.path.join(model_config['output_dir'], output_file_prefix + '.pkl')
    save_obj(model, pickle_file_path)
    eq_file_path = os.path.join(model_config['output_dir'], output_file_prefix + '-est_eq.pkl')
    eq_sympy = ellyn2sympy(str(model.stack_2_eqn(model.best_estimator_)))
    save_obj(eq_sympy, eq_file_path)
    val_err = evaluate(model, val_samples, val_targets)
    return model


def train_with_optuna(train_file_path, val_file_path, model_config, output_file_prefix):
    train_samples, train_targets = load_dataset(train_file_path)
    val_samples, val_targets = load_dataset(val_file_path)
    optuna_config = model_config['optuna']

    def optuna_objective(trial):
        model_kwargs = model_config['kwargs'].copy()
        if 'pop_size' in optuna_config:
            model_kwargs['pop_size'] = trial.suggest_int('pop_size', **optuna_config['pop_size'])

        if 'g' in optuna_config:
            model_kwargs['g'] = trial.suggest_int('g', **optuna_config['g'])

        if 'const_range' in optuna_config:
            model_kwargs['const_range'] = trial.suggest_categorical('const_range', **optuna_config['const_range'])
            
        if 'stop_threshold' in optuna_config:
            model_kwargs['stop_threshold'] = trial.suggest_loguniform('stop_threshold', **optuna_config['stop_threshold'])

        model = ellyn(**model_kwargs)
        model.fit(train_samples, train_targets)
        val_err = evaluate(model, val_samples, val_targets, quiet=True)
        return val_err

    start_time = timeit.timeit()
    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, **optuna_config['optimize'])
    train_time = timeit.timeit() - start_time
    print(f'Training time: {train_time}')
    best_param_kwargs = study.best_trial.params
    print(f'Best parameters: " {best_param_kwargs}')
    best_model = ellyn(**best_param_kwargs)
    best_model.fit(train_samples, train_targets)
    pickle_file_path = os.path.join(model_config['output_dir'], output_file_prefix + '.pkl')
    save_obj(best_model, pickle_file_path)
    eq_file_path = os.path.join(model_config['output_dir'], output_file_prefix + '-est_eq.pkl')
    eq_sympy = ellyn2sympy(str(best_model.stack_2_eqn(best_model.best_estimator_)))
    save_obj(eq_sympy, eq_file_path)
    val_err = evaluate(best_model, val_samples, val_targets)
    return best_model


def main(args):
    print(args)
    with open(args.config, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    
    model_config = config['model']
    if not args.test_only:
        if 'optuna' in model_config:
            model = train_with_optuna(args.train, args.val, model_config, args.out)
        else:
            model = train(args.train, args.val, model_config, args.out)
    else:
        with open(os.path.join(model_config['output_dir'], args.out), 'rb') as fp:
            model = pickle.load(fp)
        
    test_samples, test_targets = load_dataset(args.test)
    evaluate(model, test_samples, test_targets, eval_type='Test')


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
