"""
File for main compression experiment
Main usage: invoke from bash script in run/ folder
accompanied by the corresponding config from config/ folder

TODO:
* add a controller for using butterfly convs as 1x1 convs.

"""
import logging
import argparse
import time
import os
from policies import Manager
from utils.parse_config import yaml_ordered_load
import pdb
import torch
import numpy as np
import yaml

def get_parser():
    parser = argparse.ArgumentParser(description='Image classification neural compression')

    # The following are required parameters, the defaults are for formatting-example purposes
    parser.add_argument('--dset', default='imagenet', type=str,
        help='dataset for the task (default: "imagenet")')
    parser.add_argument('--dset_path', default='../data.imagenet', type=str,
        help='path ot the dataset (default: "../data.imagenet")')

    parser.add_argument("--backdoor_type_train", default="yellow_square", type=str,
            help="for the backdoor_celeba dataset, the type of backdoor (TRAIN)")
    parser.add_argument("--backdoor_type_test", default=None, type=str,
            help="for the backdoor_celeba dataset, the type of backdoor (TEST)")
    parser.add_argument("--backdoor_folder", default=None, type=str,
            help="for the backdoor_celeba dataset, the file containing backdoor ids of train data samples")
    parser.add_argument("--backdoor_label", default=None, type=int,
            help="for the backdoor_celeba dataset, the label that should be backdoored")
    parser.add_argument("--backdoor_fracs_train", default=None, nargs='*', type=float, 
            help="for the backdoor_celeba dataset, the proportions of negative and positive samples to backdoor. (TRAIN)")
    parser.add_argument("--backdoor_fracs_test", default=None, nargs='*', type=float, 
            help="for the backdoor_celeba dataset, the proportions of negative and positive samples to backdoor. (TEST)")


    parser.add_argument('--interpolation', default='bilinear', type=str,
            help='interpolation mode when transforming images for loading. Supported options: bilinear, bicubic.')
    parser.add_argument('--arch', default=None, type=str,
        help='NN architecture fo the task (default: None)')
    parser.add_argument('--config_path', type=str,
        help='path to config file')
    parser.add_argument('--pretrained', action='store_true',
        help='use a pretrained model')
    parser.add_argument("--label_indices", type=int,  nargs='+',
        help="Only implemented for celeba: only predict a subset of the binary labels.")

    # Training-related parameters
    parser.add_argument('--epochs', type=int, default=200,
        help='number of epochs to run (default: 200')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help="number of 'warmup' epochs before we start measuring alpha")
    parser.add_argument('--reset_momentum_after_recycling', action='store_true', default=False,
                        help='if true, reset accumulated momentum to 0 after ending pruning phase')
    parser.add_argument('--batch_size', default=256, type=int,
        help='mini-batch size (default: 256)')
    parser.add_argument('--recompute_bn_stats', action='store_true',
        help='recompute bn statistics after pruning')
    parser.add_argument('--num_samples', default=1000, type=int,
        help='number of samples to compute pruning statistics for Fisher and SNIP based pruners (default: 4096)')
    parser.add_argument('--track_weight_hist', action='store_true',
        help='if True saves wrapped modules weight hists to tensorboard')
    parser.add_argument('--fp16', action='store_true', default=False, help='if true, use mixed precision for training, otherwise use regular FP32')


    # Compute parameters
    parser.add_argument('--workers', default=4, type=int,
        help='number of workers to load the data (default: 4)')
    parser.add_argument('--cpu', action='store_true',
        help='force training on CPU')
    parser.add_argument('--gpus', default=None,
        help='Comma-separated list of GPU device ids to use, this assumes that parallel is applied (default: all devices)')

    # Run history management
    # The most convenient way is to specify --exp_name, then the logs and models will be stored under
    # ../exp_root/{exp_name}/{current_inferred_datetime}/
    parser.add_argument('--experiment_root_path', type=str, default='../exp_root',
        help='path to directory under which all experiments will be stored; you can leave this argument as is')
    parser.add_argument('--exp_name', type=str, default='default_exp',
        help='name of the experiment, will be used to name a subdirectory of experiments_root;'+
        'in this subdirectory, all runs (named by datetime) of this experiment will be stored')
    parser.add_argument('--logging_level', type=str, default='info',
        help='logging level: debug, info, warning, error, critical (default: info)')
    parser.add_argument('--training_stats_freq', type=int, default=30,
        help='the frequency (number of minibatches) to track training stats, e.g., loss, accuracy etc. (default: 30)')

    # TODO: implement checkpointing; checkpoints should be saved to the directory stored in manager.run_dir
    parser.add_argument('--checkpoint_freq', type=int, default=1000000,
        help='epoch frequency with which the checkpoints are dumped; at each time, two checkpoints are maintained:'+
        'latest and best on validation/test set')
    parser.add_argument('--from_checkpoint_path', type=str, default=None,
        help='specifies path to *run_dir* from which the progress should be resumed')
    parser.add_argument('--only_model', action='store_true',
        help='if restore only the weight and not the whole run, e.g., opt and scheduler.')

    parser.add_argument('--export_onnx', action='store_true',
        help='export onnx.')
    parser.add_argument('--onnx_nick', type=str, default=None, help='name for onnx file')
    parser.add_argument('--check_filters', action='store_true', help='small check for sparse filters')


    parser.add_argument("--checkpoint_type", type=str, default=None, help="type of checkpoint to load. Supported: ours, str, robustness, timm. If not specified here or via config lookup, code execution defaults to 'ours'")
    parser.add_argument('--manual_seed', type=int, default=0, help="Manual random seed")

    parser.add_argument('--use_wandb', action='store_true', help='set to false if not using wandb')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb name for the current run')
    parser.add_argument('--wandb_group', type=str, default=None, help='wandb group for the current run. If not specified, set to exp_name')
    parser.add_argument('--wandb_project', type=str, default="foobar", help="wandb project name")

    parser.add_argument('--transfer_config_path', type=str, help="yaml config for transfer learning.")
    parser.add_argument('--checkpoint_name', type=str, default=None, help='the name of the checkpoint, used as the key in the checkpoint yaml config. Specifying a checkpoint path and any attributes using their respective flags overrides the yaml lookup.') 
    parser.add_argument('--checkpoint_config_path', type=str, default=None, help='yaml config specifying the location and type of checkpoints.')
    parser.add_argument('--apply_deepsparse', action='store_true', help='run everything but classifier in deepsparse for forwards pass')

    return parser.parse_args()

def setup_logging(args):

    from importlib import reload
    reload(logging)

    # attrs independent of checkpoint restore
    args.logging_level = getattr(logging, args.logging_level.upper())

    run_id = time.strftime('%Y%m%d%H%M%S')
    args.exp_dir = os.path.join(args.experiment_root_path, args.exp_name)
    args.run_dir = os.path.join(args.exp_dir, os.path.join('seed{:}'.format(args.manual_seed), run_id))

    # Make directories
    os.makedirs(args.run_dir, exist_ok=True)

    log_file_path = os.path.join(args.run_dir, 'log')
    # in append mode, because we may want to restore checkpoints
    logging.basicConfig(filename=log_file_path, filemode='a',
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=args.logging_level)

    console = logging.StreamHandler()
    console.setLevel(args.logging_level)
    logging.getLogger('').addHandler(console)

    logging.info(f'Started logging run {run_id} of experiment {args.exp_name}, '+\
        f'saving checkpoints every {args.checkpoint_freq} epoch')
    return args


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    # torch.set_deterministic(True)
    args = get_parser()
    args = setup_logging(args)
    config_dictionary = dict(yaml=args.config_path, params=args, ckpt_path=args.run_dir)
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project,
                   entity='ist',
                   group=args.wandb_group or args.exp_name,
                   job_type = args.wandb_name or os.path.splitext(os.path.basename(args.config_path))[0],
                   config=config_dictionary,
                   settings=wandb.Settings(start_method="fork"))
    if args.manual_seed:
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    if args.use_wandb:
        import wandb
        # jobs are grouped into job types, then groups, then projects.
        # We use project for overall big goal, group for lower level segmenting (eg, dataset/model),
        # job type for the pruning configuration specifics, and within that the runs should only differ by the
        # random seed.
        wandb.init(project=args.wandb_project,
                   entity='ist',
                   group=args.wandb_group or args.exp_name,
                   job_type = args.wandb_name or os.path.splitext(os.path.basename(args.config_path))[0],
                   config=config_dictionary)
        if args.manual_seed:
            wandb.run.name = f'seed_{args.manual_seed}'
        wandb.save(args.config_path)

    manager = Manager(args)
    manager.run()


