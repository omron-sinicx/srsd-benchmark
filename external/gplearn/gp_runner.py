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

from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

EXTRA_FUNC_DICT = {
    'exp': make_function(np.exp, 'exp', 1),
    'asin': make_function(np.arcsin, 'asin', 1),
    'acos': make_function(np.arccos, 'acos', 1),
    'atan': make_function(np.arctan, 'atan', 1),
    'pow': make_function(np.power, 'pow', 2)
}

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
    parser = argparse.ArgumentParser(description='GP baseline runner')
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


def update_function_list(function_set):
    if function_set is None:
        return None
    return [EXTRA_FUNC_DICT.get(func_str, func_str) for func_str in function_set]


def evaluate(model, eval_samples, eval_targets, eval_type='Validation', quiet=False):
    pred_equation = model._program
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


def gplearn2sympy(eq_str):
    eq_str_w_lower_vars = re.sub(r'\bX([0-9]*[0-9])\b', r'x\1', eq_str)
    return sympy.sympify(eq_str_w_lower_vars, locals=STR2SYMPY)


def train(train_file_path, val_file_path, model_config, output_file_prefix):
    train_samples, train_targets = load_dataset(train_file_path)
    val_samples, val_targets = load_dataset(val_file_path)

    model_config['kwargs']['function_set'] = update_function_list(model_config['kwargs']['function_set'])
    model = SymbolicRegressor(**model_config['kwargs'])
    start_time = timeit.default_timer()
    model.fit(train_samples, train_targets)
    train_time = timeit.default_timer() - start_time

    print(f'Training time: {train_time}')
    pickle_file_path = os.path.join(model_config['output_dir'], output_file_prefix + '.pkl')
    save_obj(model, pickle_file_path)
    eq_file_path = os.path.join(model_config['output_dir'], output_file_prefix + '-est_eq.pkl')
    eq_sympy = gplearn2sympy(str(model._program))
    save_obj(eq_sympy, eq_file_path)
    val_err = evaluate(model, val_samples, val_targets)
    return model


def train_with_optuna(train_file_path, val_file_path, model_config, output_file_prefix):
    train_samples, train_targets = load_dataset(train_file_path)
    val_samples, val_targets = load_dataset(val_file_path)
    model_config['kwargs']['function_set'] = update_function_list(model_config['kwargs']['function_set'])
    optuna_config = model_config['optuna']

    def optuna_objective(trial):
        model_kwargs = model_config['kwargs'].copy()
        if 'population_size' in optuna_config:
            model_kwargs['population_size'] = trial.suggest_int('population_size', **optuna_config['population_size'])

        if 'generations' in optuna_config:
            model_kwargs['generations'] = trial.suggest_int('generations', **optuna_config['generations'])

        if 'stopping_criteria' in optuna_config:
            model_kwargs['stopping_criteria'] = trial.suggest_loguniform('stopping_criteria', **optuna_config['stopping_criteria'])

        if 'const_range' in optuna_config:
            model_kwargs['const_range'] = trial.suggest_categorical('const_range', **optuna_config['const_range'])

        if 'p_crossover' in optuna_config:
            model_kwargs['p_crossover'] = trial.suggest_uniform('p_crossover', **optuna_config['p_crossover'])

        if 'p_subtree_mutation' in optuna_config:
            model_kwargs['p_subtree_mutation'] = trial.suggest_uniform('p_subtree_mutation', **optuna_config['p_subtree_mutation'])

        if 'p_hoist_mutation' in optuna_config:
            model_kwargs['p_hoist_mutation'] = trial.suggest_uniform('p_hoist_mutation', **optuna_config['p_hoist_mutation'])

        if 'p_point_mutation' in optuna_config:
            model_kwargs['p_point_mutation'] = trial.suggest_uniform('p_point_mutation', **optuna_config['p_point_mutation'])

        if 'max_samples' in optuna_config:
            model_kwargs['max_samples'] = trial.suggest_uniform('max_samples', **optuna_config['max_samples'])

        if 'parsimony_coefficient' in optuna_config:
            model_kwargs['parsimony_coefficient'] = trial.suggest_uniform('parsimony_coefficient', **optuna_config['parsimony_coefficient'])

        if 'warm_start' in optuna_config:
            model_kwargs['warm_start'] = trial.suggest_categorical('warm_start', **optuna_config['warm_start'])

        model = SymbolicRegressor(**model_kwargs)
        model.fit(train_samples, train_targets)
        val_err = evaluate(model, val_samples, val_targets, quiet=True)
        return val_err

    start_time = timeit.timeit()
    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, n_trials=optuna_config['n_trials'])
    train_time = timeit.timeit() - start_time
    print(f'Training time: {train_time}')
    best_param_kwargs = study.best_trial.params
    print(f'Best parameters: " {best_param_kwargs}')
    best_model = SymbolicRegressor(**best_param_kwargs)
    best_model.fit(train_samples, train_targets)
    pickle_file_path = os.path.join(model_config['output_dir'], output_file_prefix + '.pkl')
    save_obj(best_model, pickle_file_path)
    eq_file_path = os.path.join(model_config['output_dir'], output_file_prefix + '-est_eq.pkl')
    eq_sympy = gplearn2sympy(str(best_model._program))
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
