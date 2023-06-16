import os
import errno
import torch
import shutil
import logging
import inspect
import glob
import numpy as np

from models import get_model
from utils.utils import normalize_module_name, _fix_double_wrap_dict
from utils.masking_utils import (is_wrapped_layer, 
                                 WrappedLayer, 
                                 get_wrapped_model,
                                 )
import torchvision

import pdb

__all__ = ['save_eval_checkpoint', 'save_checkpoint', 
           'load_eval_checkpoint', 'load_checkpoint',
           'load_unwrapped_checkpoint', 'get_unwrapped_model',
           ]

# Add dummy stuff here if you need to add an optimizer or a lr_scheduler
# that have required constructor arguments (and you want to recover it 
# from state dict later and need some dummy value)
DUMMY_VALUE_FOR_OPT_ARG = {'lr': 1e-3, 'gamma': 0.9}


def should_unwrap_layer(layer: 'nn.Module') -> bool:
    return isinstance(layer, WrappedLayer)

def unwrap_module(module: 'nn.Module', prefix='.'):
    """
    Recursive function which iterates over WRAPPED_MODULES of this
    module and unwraps them.
    """
    module_dict = dict(module.named_children())
    for name, sub_module in module_dict.items():
        if should_unwrap_layer(sub_module):
            setattr(module, name, sub_module.unwrap())
            # print(f'Module {prefix + name} was successfully unwrapped')
            continue
        unwrap_module(sub_module, prefix + name + '.')

def get_unwrapped_model(model: 'nn.Module') -> 'nn.Module':
    """
    Function which unwrappes the wrapped layers of received model.
    """
    unwrap_module(model)
    return model

def save_eval_checkpoint(model_config: str, model: 'nn.Module', checkpoint_path: str):
    """
    Save the model state dict with all layer unwrapped and 
    pruning masks applied.
    
    Arguments:
        model_config {dict} -- {'arch': arch, 'dataset': dataset}
        path {str} -- path to save wrapped model (e.g.: exps_root/sample_run/run_id)
    """
    unwrapped_model = get_unwrapped_model(model)
    if isinstance(unwrapped_model, torch.nn.DataParallel):
        logging.debug('Was using data parallel')
        unwrapped_model = unwrapped_model.module
    model_state_dict = unwrapped_model.state_dict()
    state_dict = dict()
    state_dict['model_config'] = model_config
    state_dict['model_state_dict'] = model_state_dict
    torch.save(state_dict, os.path.join(checkpoint_path))

def load_eval_checkpoint(checkpoint_path: str) -> 'nn.Module':
    """
    Load the evaluation ready model given the chepoint path.
    """
    try:
        state_dict = torch.load(os.path.join(checkpoint_path, 'model.pth'))
    except:
        raise IOError(errno.ENOENT, 'Evaluation checkpoint does not exist at', os.path.abspath(checkpoint_path))
    model_config = state_dict['model_config']
    model = get_model(model_config['arch'], model_config['dataset'])
    model.load_state_dict(state_dict['model_state_dict'])
    return model

def save_checkpoint(epoch, model_config, model, predictions, optimizer, lr_scheduler,
                    checkpoint_path: str,
                    is_best_sparse=False, is_best_dense=False, is_scheduled_checkpoint=False):
    """
    This function damps the full checkpoint for the running manager.
    Including the epoch, model, optimizer and lr_scheduler states. 
    """
    if not os.path.isdir(checkpoint_path):
        raise IOError(errno.ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(checkpoint_path))

    checkpoint_dict = dict()
    checkpoint_dict['epoch'] = epoch
    checkpoint_dict['model_config'] = model_config
    checkpoint_dict['model_state_dict'] = model.state_dict()
    checkpoint_dict['optimizer'] = {
        'type': type(optimizer),
        'state_dict': optimizer.state_dict()
    }
    checkpoint_dict['lr_scheduler'] = {
        'type': type(lr_scheduler),
        'state_dict': lr_scheduler.state_dict()
    }

    name_regular = f"ep{epoch}_regular"
    name_best_sparse = "best_sparse"
    name_best_dense = "best_dense"
    name_last = "last"
    # path_regular = os.path.join(checkpoint_path, f'regular_checkpoint{epoch}')
    # path_best_sparse = os.path.join(checkpoint_path, 'best_sparse_checkpoint')
    # path_best_dense = os.path.join(checkpoint_path, 'best_dense_checkpoint')
    # path_last = os.path.join(checkpoint_path, 'last_checkpoint')
    torch.save(checkpoint_dict, os.path.join(checkpoint_path, f'{name_last}_checkpoint.ckpt'))
    save_eval_checkpoint(model_config, model,  os.path.join(checkpoint_path, f'{name_last}_checkpoint.pth'))
    for k, v in predictions.items():
        np.savetxt(os.path.join(checkpoint_path, f'{k}_outputs_last.txt'), v)
    
    if is_best_sparse:
        print("util - saving best sparse")
        for extension in ('ckpt', 'pth'):
            shutil.copyfile(os.path.join(checkpoint_path, f'{name_last}_checkpoint.{extension}'),
                            os.path.join(checkpoint_path, f'{name_best_sparse}_checkpoint.{extension}'))
        for k in predictions.keys():
            shutil.copyfile(os.path.join(checkpoint_path, f'{k}_outputs_last.txt'),
							os.path.join(checkpoint_path, f'{k}_outputs_best.txt'))
    if is_best_dense:
        print("util - saving best dense")
        for extension in ('ckpt', 'pth'):
            shutil.copyfile(os.path.join(checkpoint_path, f'{name_last}_checkpoint.{extension}'),
                            os.path.join(checkpoint_path, f'{name_best_dense}_checkpoint.{extension}'))
        for k in predictions.keys():
            shutil.copyfile(os.path.join(checkpoint_path, f'{k}_outputs_last.txt'),
							os.path.join(checkpoint_path, f'{k}_outputs_best.txt'))
    if is_scheduled_checkpoint:
        print("util - saving on schedule")
        # Only save the full checkpoint, rather than the .pth, to save space.
        shutil.copyfile(os.path.join(checkpoint_path, f'{name_last}_checkpoint.ckpt'),
                        os.path.join(checkpoint_path, f'{name_regular}_checkpoint.ckpt'))

# TODO: common problem for both of the loaders below:
# if we have a different class of optimizer and scheduler,
# they will need different positional arguments that don't have defaults
# question: how do we supply these default parameters?
# the below is a crazy solution
def _load_optimizer(cls, state_dict, model):
    try:
        varnames, _, _, defaults = inspect.getargspec(cls.__init__)
        # logging.debug(f"Varnames {varnames}, defaults {defaults}")
        vars_needing_vals = varnames[2:-len(defaults)]
        # logging.debug(f"Need values: {vars_needing_vals}")
        kwargs = {v: DUMMY_VALUE_FOR_OPT_ARG[v] for v in vars_needing_vals}
        optimizer = cls(model.parameters(), **kwargs)
    except KeyError as e:
        logging.debug(f"You need to add a dummy value for {e} to DUMMY_VALUE_FOR_OPT_ARG dictionary in checkpoints module.")
        raise
    optimizer.load_state_dict(state_dict)
    return optimizer

def _load_lr_scheduler(cls, state_dict, optimizer):
    try:
        varnames, _, _, defaults = inspect.getargspec(cls.__init__)
        vars_needing_vals = varnames[2:-len(defaults)]
        kwargs = {v: DUMMY_VALUE_FOR_OPT_ARG[v] for v in vars_needing_vals}
        lr_scheduler = cls(optimizer, **kwargs)
    except KeyError as e:
        logging.debug(f"You need to add a dummy value for {e} to DUMMY_VALUE_FOR_OPT_ARG dictionary in checkpoints module.")
        raise
    lr_scheduler.load_state_dict(state_dict)
    return lr_scheduler

def load_checkpoint(full_checkpoint_path: str, model=None, load_modules=None, num_classes=None):
    """
    Loads checkpoint give full checkpoint path.
    """
    try:
        checkpoint_paths = glob.glob(full_checkpoint_path)
        # Take the last available checkpoint
        checkpoint_paths.sort(reverse=True)
        checkpoint_path = checkpoint_paths[0]
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    except:
        raise IOError(errno.ENOENT, 'Checkpoint file does not exist at', os.path.abspath(full_checkpoint_path))
    
    if model is None:
        model_config = checkpoint_dict['model_config']
        kwargs = {}
        if 'resnet50_mixed' in model_config['arch']:
            kwargs = {'use_se': False, 'se_ratio': False, 
                           'kernel_sizes': 4, 'p': 1}
        kwargs['num_classes'] = num_classes
        model = get_wrapped_model(get_model(*model_config.values(), **kwargs))
    # TODO: May need to discuss this
    # This is needed because the newly created model contains _bias_masks
    # that the saved checkpoint doesn't have. In this case we can either:
    # remove this attribute from the model at all, or leave it as is (just a ones mask, no problem)
    # and update the entries in the model's state_dict that appear in the saved state_dict
    # -- the latter option is implemented below:
    updated_state_dict = model.state_dict()
    if 'module' in list(checkpoint_dict['model_state_dict'].keys())[0]:
        checkpoint_dict['model_state_dict'] = {normalize_module_name(k): v for k, v in checkpoint_dict['model_state_dict'].items()}
    if load_modules is not None:
        def should_be_loaded(name, prefixes):
            for prefix in prefixes:
                if name.startswith(prefix):
                    return True
            return False
        checkpoint_dict['model_state_dict'] = {k: v for k, v in checkpoint_dict['model_state_dict'].items() if should_be_loaded(k, load_modules)}

    # Check that there is an exact match between model and checkpoint keys.
    model_keys = {k for k in updated_state_dict.keys()}
    checkpoint_keys = {k for k in checkpoint_dict["model_state_dict"].keys()}
    if load_modules is not None:
        in_model_not_in_checkpoint = {k for k in model_keys.difference(checkpoint_keys) if k.startswith(tuple(load_modules))}
        in_checkpoint_not_in_model = checkpoint_keys.difference(model_keys)
        if in_model_not_in_checkpoint or in_checkpoint_not_in_model:
            raise ValueError(f"Mismatch between model and checkpoints:\n  Tensors in model not in checkpoint:{in_model_not_in_checkpoint}\n  In checkpoint:{in_checkpoint_not_in_model}") 

    for k in updated_state_dict.keys():
        if k in checkpoint_dict["model_state_dict"]:
            updated_state_dict[k] = checkpoint_dict["model_state_dict"][k]
    model.load_state_dict(updated_state_dict)

    return model




def load_unwrapped_checkpoint(full_checkpoint_path):
    checkpoint_dict = torch.load(full_checkpoint_path, map_location='cpu')
    model_config = checkpoint_dict['model_config']
    model = get_model(*model_config.values())
    updated_state_dict = model.state_dict()
    if 'module' in list(checkpoint_dict['model_state_dict'].keys())[0]:
        checkpoint_dict['model_state_dict'] = {normalize_module_name(k): v for k, v in checkpoint_dict['model_state_dict'].items()}
    updated_state_dict.update(checkpoint_dict['model_state_dict'])
    model.load_state_dict(updated_state_dict)
    optimizer_dict = checkpoint_dict['optimizer']['state_dict']
    lr_scheduler_dict = checkpoint_dict['lr_scheduler']['state_dict']
    pruning_lr_scheduler_dict = None

    return model
