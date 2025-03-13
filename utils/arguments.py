"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

ICASSP 2025 paper: Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification
URL: arxiv.org/abs/2405.16597
GitHub: https://github.com/QizaoWang/CSSC-CCReID
"""

from __future__ import absolute_import
import argparse
import os
import os.path as osp
import sys

from utils.util import Logger

import torch


def set_log(args):
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))


def print_args(args):
    print('------------------------ Args -------------------------')
    for k, v in vars(args).items():  # namespace to dict and get each item
        print('%s: %s' % (k, v))
    print('--------------------- Args End ------------------------')
    return


def set_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
    else:
        print("Currently using CPU (GPU is highly recommended)")
    return use_gpu


def get_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('-d', '--dataset', type=str, default='ltcc')
    parser.add_argument('--dataset_root', type=str, default='data', help="root path to data directory")
    parser.add_argument('--dataset_filename', type=str, default='LTCC-reID')
    parser.add_argument('--height', type=int, default=384,
                        help="height of an image")
    parser.add_argument('--width', type=int, default=192,
                        help="width of an image")
    parser.add_argument('--horizontal_flip_pro', type=float, default=0.5,
                        help="Random probability for image horizontal flip")
    parser.add_argument('--pad_size', type=int, default=10,
                        help="Value of padding size")
    parser.add_argument('--random_erasing_pro', type=float, default=0.5,
                        help="Random probability for image random erasing")
    # data manager and loader
    parser.add_argument('--split_id', type=int, default=0, help="split index")
    parser.add_argument('--train_batch', default=32, type=int,
                        help="train batch size")
    parser.add_argument('--test_batch', default=128, type=int,
                        help="test batch size")
    parser.add_argument('-j', '--num_workers', default=8, type=int,
                        help="number of data loading workers")
    parser.add_argument('--num_instances', type=int, default=8,
                        help="number of instances per identity")

    # Optimization options
    parser.add_argument('--start_epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--max_epoch', default=120, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--margin', type=float, default=0.3,
                        help="margin for triplet loss. If not specified, use soft-margin version.")

    parser.add_argument('--lr', default=0.0003, type=float,
                        help="initial learning rate")
    parser.add_argument('--warm_up_factor', default=0.1, type=float,
                        help="warm up factor")
    parser.add_argument('--warm_up_method', default="linear", type=str,
                        choices=['constant', 'linear'],
                        help="warm up factor")
    parser.add_argument('--warm_up_epochs', default=10, type=int,
                        help="take how many epochs to warm up")
    parser.add_argument('--step_milestones', default=[30, 60], nargs='*', type=int,
                        help="epoch milestones to decay learning rate, multi steps")
    parser.add_argument('--gamma', default=0.1, type=float,
                        help="learning rate decay")
    parser.add_argument('--weight_decay', default=5e-04, type=float,
                        help="lr weight decay")
    parser.add_argument('--weight_decay_bias', default=0.0005, type=float,
                        help="lr weight decay for layers with bias")

    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help='pre-trained model path')

    parser.add_argument('--use_cpu', action='store_true', help="use cpu")
    parser.add_argument('--gpu_devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', type=int, default=666, help="manual seed")

    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    parser.add_argument('--print_train_info_epoch_freq', type=int, default=5,
                        help="print training information per #epoch, default -1 means don't print")
    parser.add_argument('--start_eval_epoch', type=int, default=0,
                        help="start to evaluate after training a specific epoch")
    parser.add_argument('--eval_epoch', type=int, default=5,
                        help="run evaluation for every N epochs (set to -1 to test after training)")
    parser.add_argument('--reranking', action='store_true', help='result re_ranking')
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--save_checkpoint', action='store_true', help='save model checkpoint')

    parser.add_argument('--tri_start_epoch', type=int, default=35,
                        help="the epoch of starting to use triplet loss")

    args = parser.parse_args()
    return args
