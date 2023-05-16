"""
This code is based on https://github.com/yoshitomo-matsubara/torchdistill/blob/main/examples/image_classification.py
"""

import argparse
import datetime
import os
import time

import numpy as np
import sympy
import torch
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchdistill.common import file_util, yaml_util, module_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process, init_distributed_mode, load_ckpt, save_ckpt, set_seed
from torchdistill.core.distillation import get_distillation_box
from torchdistill.core.training import get_training_box
from torchdistill.datasets import util
from torchdistill.losses.single import get_single_loss
from torchdistill.misc.log import setup_log_file, SmoothedValue, MetricLogger
from torchdistill.models.registry import get_model
from torchdistill.optim.registry import get_optimizer, get_scheduler

from eq.conversion import sequence2model
from eq.eval import convert_pred_sequence_to_eqs, compute_edit_distance
from eq.tree import visualize_sympy_as_tree
from models.optim import customize_lr_config
from models.registry import get_symbolic_regression_model

logger = def_logger.getChild(__name__)


def get_argparser():
    parser = argparse.ArgumentParser(description='PyTorch pipeline for symbolic regression models')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--log', help='log file path')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--seed', type=int, help='seed in random number generator')
    parser.add_argument('-test_only', action='store_true', help='only test the models')
    parser.add_argument('-estimate_coeff', action='store_true', help='estimate coefficients')
    parser.add_argument('-student_only', action='store_true', help='test the student model only')
    parser.add_argument('-log_config', action='store_true', help='log config')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('-adjust_lr', action='store_true',
                        help='multiply learning rate by number of distributed processes (world_size)')
    return parser


def load_model(model_config, device):
    model = get_symbolic_regression_model(model_config['name'], **model_config['params'])
    if model is None:
        repo_or_dir = model_config.get('repo_or_dir', None)
        model = get_model(model_config['name'], repo_or_dir, **model_config['params'])

    ckpt_file_path = model_config['ckpt']
    load_ckpt(ckpt_file_path, model=model, strict=True)
    return model.to(device)


def train_one_epoch(training_box, device, epoch, log_freq):
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('sample/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, target_values, target_sequences, true_eq_file_paths in \
            metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        start_time = time.time()
        sample_batch, target_values, target_sequences = \
            sample_batch.to(device), target_values.to(device), target_sequences.to(device)
        loss = training_box(sample_batch, target_sequences, supp_dict=None)
        training_box.update_params(loss)
        batch_size = sample_batch.shape[0]
        metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]['lr'])
        metric_logger.meters['sample/s'].update(batch_size / (time.time() - start_time))
        if (torch.isnan(loss) or torch.isinf(loss)) and is_main_process():
            raise ValueError('The training loop was broken due to loss = {}'.format(loss))


def estimate_coefficients(pred_sequence, sample_batch, target_values, true_eq_file_path, coeff_estimate_config):
    device = sample_batch.device
    pred_model = sequence2model(pred_sequence[1:])
    pred_model = pred_model.to(device)
    skeleton_eq_str = pred_model.sympy_str()
    logger.info(f'Skeleton: {skeleton_eq_str}')
    pred_model.train()
    optim_config = coeff_estimate_config['optimizer']
    filters_params = optim_config.get('filters_params', True)
    optimizer = get_optimizer(pred_model, optim_config['type'], optim_config['params'], filters_params)
    scheduler_config = coeff_estimate_config.get('scheduler', None)
    if scheduler_config is not None and len(scheduler_config) > 0:
        lr_scheduler = get_scheduler(optimizer, scheduler_config['type'], scheduler_config['params'])
        scheduling_step = scheduler_config.get('scheduling_step', 0)
    else:
        lr_scheduler = None
        scheduling_step = None

    criterion_config = coeff_estimate_config['criterion']
    criterion = get_single_loss(criterion_config)
    num_epochs = coeff_estimate_config['num_epochs']
    best_loss_value = None
    best_state_dict = None
    sample_batch = sample_batch.squeeze(0).squeeze(-1)
    target_values = target_values.squeeze(0)
    for epoch in range(num_epochs):
        pred_values = pred_model(sample_batch)
        loss = criterion(pred_values, target_values)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None and epoch != 0 and epoch % scheduling_step == 0:
            lr_scheduler.step()

        loss_value = loss.item()
        if best_loss_value is None or loss_value < best_loss_value:
            logger.info(f'Best loss {best_loss_value} -> {loss_value}')
            best_loss_value = loss_value
            best_state_dict = pred_model.state_dict()

    pred_model.load_state_dict(best_state_dict)
    try:
        sympy_eq_str = pred_model.sympy_str()
        logger.info(f'"{skeleton_eq_str}" -> "{sympy_eq_str}"')
        pred_eq = sympy.sympify(sympy_eq_str)
    except:
        pred_eq = sympy.nan

    pred_eq_output_dir_path = None if coeff_estimate_config is None \
        else coeff_estimate_config.get('pred_eq_output', None)
    if pred_eq_output_dir_path is not None:
        pred_eq_output_file_path = os.path.join(pred_eq_output_dir_path, os.path.basename(true_eq_file_path))
        file_util.save_pickle(pred_eq, pred_eq_output_file_path)
    return pred_eq


def evaluate(model, data_loader, device, device_ids, distributed, log_freq=1000, title=None, header='Test:',
             pred_tree_output_dir_path=None, true_tree_output_dir_path=None, coeff_estimate_config=None):
    model.to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=device_ids)
    elif device.type.startswith('cuda'):
        model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    tree_index = 0
    for sample_batch, target_values, true_eqs, true_eq_file_paths \
            in metric_logger.log_every(data_loader, log_freq, header):
        sample_batch = sample_batch.to(device, non_blocking=True)
        target_values = target_values.to(device, non_blocking=True)
        output = model(sample_batch)
        if coeff_estimate_config is None:
            pred_eqs = convert_pred_sequence_to_eqs(output['pred_symbols'])
        else:
            pred_eqs = [
                estimate_coefficients(pred_sequence, sample_batch, target_values,
                                      true_eq_file_path, coeff_estimate_config)
                for pred_sequence, true_eq_file_path in zip(output['pred_symbols'], true_eq_file_paths)
            ]

        if pred_tree_output_dir_path is not None:
            for pred_eq, true_eq in zip(pred_eqs, true_eqs):
                pred_file_path = os.path.join(pred_tree_output_dir_path, f'pred-{tree_index}.png')
                visualize_sympy_as_tree(pred_eq, pred_file_path, ext='png', label=str(pred_eq))
                if true_tree_output_dir_path is not None:
                    true_file_path = os.path.join(true_tree_output_dir_path, f'true-{tree_index}.png')
                    visualize_sympy_as_tree(true_eq, true_file_path, ext='png', label=str(true_eq))
                tree_index += 1

        edit_dist = compute_edit_distance(pred_eqs, true_eqs, normalizes=True)
        logger.info('{}\t{}\t{}\t{}'.format(output['pred_symbols'][0], str(edit_dist),
                                            str(true_eqs[0].evalf()), str(true_eqs[0])))
        batch_size = sample_batch.shape[0]
        metric_logger.meters['edit_dist'].update(edit_dist, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    global_edit_dist = metric_logger.edit_dist.global_avg
    logger.info(' * Normalized edit distance {:.4f}\n'.format(global_edit_dist))
    return global_edit_dist


def train(teacher_model, student_model, dataset_dict, ckpt_file_path, device, device_ids, distributed, config, args):
    logger.info('Start training')
    train_config = config['train']
    lr_factor = args.world_size if distributed and args.adjust_lr else 1
    training_box = get_training_box(student_model, dataset_dict, train_config,
                                    device, device_ids, distributed, lr_factor) if teacher_model is None \
        else get_distillation_box(teacher_model, student_model, dataset_dict, train_config,
                                  device, device_ids, distributed, lr_factor)
    best_val_edit_dist = np.inf
    optimizer, lr_scheduler = training_box.optimizer, training_box.lr_scheduler
    if file_util.check_if_exists(ckpt_file_path):
        best_val_edit_dist, _, _ = load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    log_freq = train_config['log_freq']
    student_model_without_ddp = student_model.module if module_util.check_if_wrapped(student_model) else student_model
    start_time = time.time()
    for epoch in range(args.start_epoch, training_box.num_epochs):
        training_box.pre_process(epoch=epoch)
        train_one_epoch(training_box, device, epoch, log_freq)
        val_edit_dist = evaluate(student_model, training_box.val_data_loader, device, device_ids, distributed,
                                 log_freq=log_freq, header='Validation:')
        if val_edit_dist < best_val_edit_dist and is_main_process():
            logger.info('Best edit distance: {:.4f} -> {:.4f}'.format(best_val_edit_dist, val_edit_dist))
            logger.info('Updating ckpt at {}'.format(ckpt_file_path))
            best_val_edit_dist = val_edit_dist
            save_ckpt(student_model_without_ddp, optimizer, lr_scheduler,
                      best_val_edit_dist, config, args, ckpt_file_path)
        training_box.post_process()

    if distributed:
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    training_box.clean_modules()


def main(args):
    log_file_path = args.log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    distributed, device_ids = init_distributed_mode(args.world_size, args.dist_url)
    logger.info(args)
    set_seed(args.seed)
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    device = torch.device(args.device)
    dataset_dict = util.get_all_datasets(config['datasets'])
    # Update config with dataset size len(data_loader)
    customize_lr_config(config, dataset_dict, args.world_size)

    models_config = config['models']
    teacher_model_config = models_config.get('teacher_model', None)
    teacher_model =\
        load_model(teacher_model_config, device) if teacher_model_config is not None else None
    student_model_config =\
        models_config['student_model'] if 'student_model' in models_config else models_config['model']
    ckpt_file_path = student_model_config['ckpt']
    student_model = load_model(student_model_config, device)
    if args.log_config:
        logger.info(config)

    if not args.test_only:
        train(teacher_model, student_model, dataset_dict, ckpt_file_path, device, device_ids, distributed, config, args)
        student_model_without_ddp =\
            student_model.module if module_util.check_if_wrapped(student_model) else student_model
        load_ckpt(student_model_config['ckpt'], model=student_model_without_ddp, strict=True)

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                              test_data_loader_config, distributed)
    pred_tree_output_dir_path = test_config.get('pred_tree_output')
    true_tree_output_dir_path = test_config.get('true_tree_output')
    if not args.student_only and teacher_model is not None:
        evaluate(teacher_model, test_data_loader, device, device_ids, distributed,
                 title='[Teacher: {}]'.format(teacher_model_config['name']))

    coeff_estimate_config = config.get('coeff_estimate', None) if args.estimate_coeff else None
    evaluate(student_model, test_data_loader, device, device_ids, distributed,
             title='[Student: {}]'.format(student_model_config['name']),
             pred_tree_output_dir_path=pred_tree_output_dir_path, true_tree_output_dir_path=true_tree_output_dir_path,
             coeff_estimate_config=coeff_estimate_config)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
