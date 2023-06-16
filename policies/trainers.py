"""
This module implements training policies.
For most usecases, only one trainer instance is needed for training and pruning
with a single model. Several trainers can be used for training with knowledge distillation.
"""

import numpy as np
import torch
import torch.nn as nn
from optimization.sgd import SGD
from optimization.adam import Adam
#from torch.optim import Adam
from torch.optim.lr_scheduler import *
from torch.cuda.amp import autocast
import torch.nn.functional as F
import logging
# import torchcontrib

from policies.policy import PolicyBase
from optimization.FreezedAdagrad import FreezedAdagrad
from optimization.gradual_norm_reduction_pruner import (
    _preprocess_params_for_pruner_optim, 
    GradualNormPrunerSGD
)
from optimization.lr_schedulers import StageExponentialLR, CosineLR
from utils.jsd_loss import JsdCrossEntropy
from utils import misc


SPECIAL_OPTIMIZERS = ['GradualNormPrunerSGD']


def build_optimizer_from_config(model, optimizer_config, update_modules=None):
    optimizer_class = optimizer_config['class']
    restricted_keys = ['class', 'swa_start', 'swa_freq', 'swa_lr', 'modules']
    optimizer_args = {k: v for k, v in optimizer_config.items() if k not in restricted_keys}
    if update_modules is None:
        update_params = model.parameters()
    else:
        update_params = [p for n, p in model.named_parameters() if n[7:] in update_modules]
        print("The following will be updated: ", len(update_params))
    if optimizer_class in SPECIAL_OPTIMIZERS:
        params = _preprocess_params_for_pruner_optim(model, optimizer_config['modules'])
        optimizer_args['params'] = params
    else:
       optimizer_args['params'] = model.parameters()
    optimizer = globals()[optimizer_class](**optimizer_args)

    if 'swa_start' in optimizer_config.keys():
        optimizer = torchcontrib.optim.SWA(optimizer, swa_start=optimizer_config['swa_start'],
            swa_freq=optimizer_config['swa_freq'], swa_lr=optimizer_config['swa_lr'])
    return optimizer


def build_lr_scheduler_from_config(optimizer, lr_scheduler_config):
    lr_scheduler_class = lr_scheduler_config['class']
    lr_scheduler_args = {k: v for k, v in lr_scheduler_config.items() if k != 'class'}
    lr_scheduler_args['optimizer'] = optimizer
    epochs = lr_scheduler_args['epochs']
    lr_scheduler_args.pop('epochs')
    lr_scheduler = globals()[lr_scheduler_class](**lr_scheduler_args)
    return lr_scheduler, epochs


def build_training_policy_from_config(model, scheduler_dict, trainer_name, use_lr_rewind=False,
                                      use_jsd=False, num_splits=None, fp16_scaler=None, update_modules=None, binary=False):
    print("Update modules at going to trainer", update_modules)
    trainer_dict = scheduler_dict['trainers'][trainer_name]
    optimizer = build_optimizer_from_config(model, trainer_dict['optimizer'], update_modules)
    lr_scheduler, epochs = build_lr_scheduler_from_config(optimizer, trainer_dict['lr_scheduler'])
    return TrainingPolicy(model, optimizer, lr_scheduler, epochs,
        use_jsd=use_jsd, num_splits=num_splits, fp16_scaler=fp16_scaler, binary=binary)


class TrainingPolicy(PolicyBase):
    def __init__(self, model, optimizer, lr_scheduler, epochs,
                 use_jsd=False, num_splits=None, fp16_scaler=None, binary=False):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.model = model
        self.fp16_scaler = fp16_scaler
        self.enable_autocast = False
        if fp16_scaler is not None:
            self.enable_autocast = True
        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.BCEWithLogitsLoss()

        print("initial optim lr", self.optim_lr)

        self.use_jsd = use_jsd
        self.num_splits = num_splits

        if self.use_jsd:
            if self.num_splits == 0: raise ValueError('num_splits > 0! if use_jsd == True')
            self.jsd_loss = JsdCrossEntropy(num_splits=self.num_splits)
        self.binary = binary
        self.loss = F.cross_entropy
        if self.binary:
            self.loss = F.binary_cross_entropy_with_logits

    def eval_model(self, loader, device, epoch_num, num_classes, avg_per_class=False):
        self.model.eval()
        eval_loss = 0
        total_correct = 0
        total_pos = 0
        if self.binary:
            n_samples = len(loader.dataset)
            targets = torch.zeros(n_samples, num_classes)
            outputs = torch.zeros(n_samples, num_classes)
            start_index = 0
        per_class_correct = None
        per_class_counts = None
        with torch.no_grad():
            for in_tensor, target in loader:
                batch_size=in_tensor.shape[0]
                in_tensor, target = in_tensor.to(device), target.to(device)
                if self.binary:
                    avg_per_class = False
                    target = target.float()
                with autocast(enabled=self.enable_autocast):
                    output = self.model(in_tensor)
                    eval_loss += self.loss(output, target, reduction='sum').item()
                if self.binary:
                    outputs[start_index: start_index+batch_size] = output
                    targets[start_index: start_index+batch_size] = target
                    start_index += batch_size
                    pred = output.gt(0.).float()
                    pred_pos = pred.sum().item()
                else:
                    pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred))
                if self.binary:
                    correct = correct / target.size(1)
                total_correct += correct.sum().item()
                total_pos += pred_pos
                if avg_per_class:
                    per_class_pred = F.one_hot(output.argmax(dim=1), num_classes = output.shape[1])
                    if per_class_correct == None:
                        per_class_correct = (per_class_pred * correct).sum(dim=0)
                    else:
                        per_class_correct+= (per_class_pred * correct).sum(dim=0)
                    if per_class_counts == None:
                        per_class_counts = F.one_hot(target, num_classes = output.shape[1]).sum(dim=0)
                    else:
                        per_class_counts += F.one_hot(target, num_classes = output.shape[1]).sum(dim=0)
        if self.binary:
            per_label_tst_accs, per_label_tst_rec, per_label_tst_prec, \
                    per_label_fpr, per_label_fnr, \
            per_label_tst_f1, per_label_tst_auc, \
            per_label_prec_at_50, per_label_prec_at_75, per_label_prec_at_90, per_label_prec_at_95, \
            label_tst_freqs, pred_tst_freqs = \
                    misc.get_acc_precision_recall(outputs, targets)
            eval_loss /= len(loader.dataset)

            return eval_loss, total_correct, total_pos, per_label_tst_accs, \
                    per_label_tst_prec, per_label_tst_rec, \
                    per_label_fpr, per_label_fnr, \
                    per_label_tst_f1, per_label_tst_auc, \
                per_label_prec_at_50, per_label_prec_at_75, per_label_prec_at_90, per_label_prec_at_95, label_tst_freqs, pred_tst_freqs
        avg_per_class_accuracy = None
        if avg_per_class:
            avg_per_class_accuracy = (per_class_correct / per_class_counts.to(device)).mean().item()
        eval_loss /= len(loader.dataset)
        return eval_loss, total_correct, avg_per_class_accuracy

    @property
    def optim_lr(self):
        return list(self.optimizer.param_groups)[0]['lr']


    def on_minibatch_begin(self, minibatch, device, loss, **kwargs):
        """
        Loss can be composite, e.g., if we want to add some KD or
        regularization in future
        """
        self.model.train()
        self.optimizer.zero_grad()
        in_tensor, target = minibatch

        if hasattr(self, 'jsd_loss'):
            in_tensor = torch.cat(in_tensor)
            target = torch.cat(self.num_splits*[target])
        in_tensor, target = in_tensor.to(device), target.to(device)
        if self.binary:
            target = target.float()
        with autocast(enabled=self.enable_autocast):
            output = self.model(in_tensor)
            if hasattr(self, 'jsd_loss'):
                loss += self.jsd_loss(output, target)
            else:
                loss += self.loss(output, target)
        if self.binary:
            pred = output.gt(0).float()
            pred_pos = pred.mean()
        else:
            pred = output.argmax(dim=1, keepdim=True)
            pred_pos = 0 # Fix this
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = 1.0 * correct / target.size(0)
        if self.binary:
            acc = acc / target.size(1)
        loss = torch.sum(loss)
        acc = np.sum(acc)
        return loss, acc, pred_pos
        # >>>>>>> generalization

    def on_parameter_optimization(self, loss, epoch_num, reset_momentum, **kwargs):
        if reset_momentum:
            print("resetting momentum")
            self.optimizer.reset_momentum_buffer()
        if self.enable_autocast:
            self.fp16_scaler.scale(loss).backward()
            self.fp16_scaler.step(self.optimizer)
            self.fp16_scaler.update()
        else:
            loss.backward()
            if isinstance(self.optimizer, FreezedAdagrad):
                self.optimizer.step(epoch_num)
            else:
                self.optimizer.step()


    def on_epoch_end(self, bn_loader, swap_back, device, epoch_num, **kwargs):
        self.optimizer.reset_momentum_buffer()
        start, freq, end = self.epochs
        if (epoch_num - start) % freq == 0 and epoch_num < end + 1 and start - 1 < epoch_num:
            self.lr_scheduler.step()
        if hasattr(self.lr_scheduler, 'change_mode') and epoch_num > end:
            self.lr_scheduler.change_mode()
            self.lr_scheduler.step()
        
        if hasattr(self.optimizer, 'on_epoch_begin'):
            self.optimizer.on_epoch_begin()

        if bn_loader is not None:
            print('Averaged SWA model:')
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(bn_loader, self.model, device)

        if swap_back:
            self.optimizer.swap_swa_sgd()

if __name__ == '__main__':
    """
    TODO: remove after debug
    """
    from efficientnet_pytorch import EfficientNet
    from masking_utils import get_wrapped_model

    from utils import read_config
    path = "./configs/test_config.yaml"
    sched_dict = read_config(stream)

    model = get_wrapped_model(EfficientNet.from_pretrained('efficientnet-b1'))
    optimizer = build_optimizer_from_config(model, sched_dict['optimizer'])
    lr_scheduler,_ = build_lr_scheduler_from_config(optimizer, sched_dict['lr_scheduler'])
    training_policy = build_training_policy_from_config(model, sched_dict)

