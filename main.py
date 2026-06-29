import argparse
import os
import random
import sys
import traceback

import numpy as np
import torch
import yaml

from train import Trainer
from utils import Report


class YamlAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        yaml_dict = yaml.safe_load(values)
        setattr(namespace, self.dest, yaml_dict)


def get_parser():
    parser = argparse.ArgumentParser(
        description="C-TAMP: Continuous Text-conditioned Action Masked Prediction for Zero-Shot Skeleton Action Recognition"
    )
    parser.add_argument('--work-dir', default='./work_dir', help='the work folder for storing results')
    parser.add_argument('--config', default=None, help='path to the configuration file')
    parser.add_argument('--phase', default='train', choices=['train', 'test'], help='train or test')

    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--log-iter', type=int, default=100, help='log interval in iterations')
    parser.add_argument('--save-iter', type=int, default=1000, help='checkpoint interval in iterations')

    parser.add_argument('--feeder', default='feeders.feeder.FeatureFeeder', help='data loader class')
    parser.add_argument('--num-worker', type=int, default=4, help='number of workers for data loader')
    parser.add_argument('--train-feeder-args', action=YamlAction, default=dict(), help='training data loader args')
    parser.add_argument('--test-feeder-args', action=YamlAction, default=dict(), help='test data loader args')

    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--optimizer', default='AdamW', help='optimizer type')
    parser.add_argument('--lr-scheduler', default='cosine', help='learning rate scheduler type')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--num-iter', type=int, default=50000, help='number of training iterations')
    parser.add_argument('--num-warmup', type=int, default=100, help='number of warmup iterations')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=512, help='test batch size')
    parser.add_argument('--mixed-precision', type=str, default=None, choices=[None, 'no', 'fp16', 'bf16'])

    parser.add_argument('--num-classes', type=int, default=60)
    parser.add_argument('--class-name-path', type=str, default='./data/class_lists/ntu60.csv')
    parser.add_argument('--class-description-path', type=str, default='./data/class_lists/ntu60_llm.txt')
    parser.add_argument(
        '--class-name-mode',
        type=str,
        default='csv_label',
        choices=['csv_label', 'quoted_description_name', 'quoted_name_plus_csv_label'],
        help='how to build the first text condition',
    )
    parser.add_argument(
        '--class-description-mode',
        type=str,
        default='raw',
        choices=['raw', 'append_csv_label', 'prepend_csv_label'],
        help='how to build the second text condition',
    )
    parser.add_argument('--text-model-name-or-path', type=str, default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--text-max-length', type=int, default=35)
    parser.add_argument('--text-encode-batch-size', type=int, default=64)
    parser.add_argument('--checkpoint-path', type=str, default=None)

    parser.add_argument('--unseen-label', type=int, default=5, help='number of unseen classes')
    parser.add_argument('--unseen-label-path', type=str, default='./data/label_splits/ntu60/ru5.npy')

    parser.add_argument('--feature-dim', type=int, default=256)
    parser.add_argument('--num-groups', type=int, default=16)
    parser.add_argument('--model-dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--mlp-ratio', type=float, default=4.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--permutation-seed', type=int, default=2025)

    parser.add_argument('--mask-ratios', action=YamlAction, default=[0.375, 0.5, 0.625])
    parser.add_argument('--test-mask-ratio', type=float, default=0.5)
    parser.add_argument('--num-test-masks', type=int, default=8)
    parser.add_argument('--test-mask-seed', type=int, default=2025)

    parser.add_argument('--num-hard-negatives', type=int, default=4)
    parser.add_argument('--hard-negative-pool-size', type=int, default=15)
    parser.add_argument('--energy-temperature', type=float, default=0.1)
    parser.add_argument('--completion-weight', type=float, default=1.0)
    parser.add_argument('--energy-weight', type=float, default=1.0)
    parser.add_argument('--huber-beta', type=float, default=1.0)
    parser.add_argument('--class-chunk-size', type=int, default=16)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--diagnose-text-usage', type=str2bool, default=False)
    return parser


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Unsupported boolean value encountered.')


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def validate_args(args):
    if args.feature_dim % args.num_groups != 0:
        raise ValueError('feature_dim must be divisible by num_groups.')
    if args.model_dim % args.num_heads != 0:
        raise ValueError('model_dim must be divisible by num_heads.')
    if not args.mask_ratios:
        raise ValueError('mask_ratios cannot be empty.')
    for ratio in args.mask_ratios:
        if not 0 < float(ratio) < 1:
            raise ValueError('All mask_ratios must be in (0, 1).')
    if not 0 < args.test_mask_ratio < 1:
        raise ValueError('test_mask_ratio must be in (0, 1).')
    if args.num_hard_negatives < 1:
        raise ValueError('num_hard_negatives must be >= 1.')
    if args.hard_negative_pool_size < args.num_hard_negatives:
        raise ValueError('hard_negative_pool_size must be >= num_hard_negatives.')
    if args.energy_temperature <= 0:
        raise ValueError('energy_temperature must be positive.')
    if args.num_test_masks < 1:
        raise ValueError('num_test_masks must be >= 1.')
    if args.class_chunk_size < 1:
        raise ValueError('class_chunk_size must be >= 1.')
    if args.num_classes <= args.unseen_label:
        raise ValueError('num_classes must be greater than unseen_label.')
    if args.text_max_length < 1:
        raise ValueError('text_max_length must be positive.')
    if args.text_encode_batch_size < 1:
        raise ValueError('text_encode_batch_size must be positive.')


def load_data(args):
    Feeder = import_class(args.feeder)
    data_loader = dict()
    train_dataset = Feeder(**args.train_feeder_args)
    test_dataset = Feeder(**args.test_feeder_args)
    data_loader['train'] = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        drop_last=True,
    )
    data_loader['test'] = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        drop_last=False,
    )
    return data_loader


def run_train(args):
    global_step = 0
    train_log = Report(args.work_dir, type='train')
    test_log = Report(args.work_dir, type='test')
    data_loader = load_data(args)
    trainer = Trainer(args=args, data_loader=data_loader)

    best_zsl_acc = 0.0
    best_zsl_epoch = 0
    epoch = 0
    while global_step < args.num_iter:
        epoch += 1
        train_log.write(f'========= Epoch {epoch} =========')
        global_step = trainer.train(train_log, global_step)
        zsl_test_acc = trainer.test()
        test_log.write(f'ZSL Test Acc: {zsl_test_acc:.6f}\tEpoch: {epoch}\tIter: {global_step}')
        if zsl_test_acc > best_zsl_acc:
            best_zsl_acc = zsl_test_acc
            best_zsl_epoch = epoch
            trainer.save_best_model()
            test_log.write(f'Best Test Acc: {best_zsl_acc:.6f}\tBest Epoch: {best_zsl_epoch}')
        if args.diagnose_text_usage:
            diag = trainer.diagnose_text_usage(max_batches=1)
            test_log.write(
                'Text Usage: '
                + '\t'.join(f'{key}: {value:.6f}' for key, value in diag.items())
            )


def run_test(args):
    if not args.checkpoint_path:
        raise ValueError('--checkpoint-path is required when phase is test.')
    test_log = Report(args.work_dir, type='test')
    data_loader = load_data(args)
    trainer = Trainer(args=args, data_loader=data_loader)
    trainer.load_state(args.checkpoint_path)
    zsl_test_acc = trainer.test()
    test_log.write(f'ZSL Test Acc: {zsl_test_acc:.6f}\tCheckpoint: {args.checkpoint_path}')


def parse_args():
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_args = yaml.safe_load(f)
        valid_keys = vars(p).keys()
        for key in default_args.keys():
            if key not in valid_keys:
                raise ValueError(f'WRONG ARG: {key}')
        parser.set_defaults(**default_args)
    args = parser.parse_args()
    validate_args(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    init_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.phase == 'train':
        run_train(args)
    elif args.phase == 'test':
        run_test(args)
    else:
        raise ValueError(f'Unsupported phase: {args.phase}')
