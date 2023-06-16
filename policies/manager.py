"""
Managing class for different Policies.

"""

from models import get_model
from policies.pruners import build_pruners_from_config
from policies.trainers import build_training_policy_from_config
from policies.recyclers import build_recyclers_from_config

from utils import (read_config, get_datasets,
                   classification_num_classes,
                   get_wrapped_model,
                   get_total_sparsity,
                   get_unwrapped_model,
                   preprocess_for_device,
                   recompute_bn_stats,
                   load_checkpoint,
                   save_checkpoint,
                   save_eval_checkpoint,
                   TrainingProgressTracker,
                   normalize_module_name,
                   get_weights_hist_meta,
                   get_macs_sparse, get_flops,
                   celeba_classes, awa_classes)
from utils.masking_utils import WrappedLayer

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.onnx
import time
import os
import logging
import wandb
import collections
import copy 
import sys

import pdb

# TODO (Discuss): do we want to have tqdm in the terminal output?
USE_TQDM = True
if not USE_TQDM:
    tqdm = lambda x: x

DEFAULT_TRAINER_NAME = "default_trainer"


# noinspection SyntaxError
class Manager:
    """
    Class for invoking trainers and pruners.
    Trainers
    """

    def __init__(self, args):
        args = preprocess_for_device(args)
        logging.info(args)


        # REFACTOR: Model setup
        if args.manual_seed:
            torch.manual_seed(args.manual_seed)
            np.random.seed(args.manual_seed)
        else:
            np.random.seed(0)

        self.transfer_config = None
        if args.transfer_config_path:
            self.transfer_config = args.transfer_config_path if isinstance(args.transfer_config_path, dict) else read_config(args.transfer_config_path)
        # self.checkpoint_config = None
        # if args.checkpoint_config_path:
        #     self.checkpoint_config = args.checkpoint_config_path if isinstance(args.checkpoint_config_path, dict) else read_config(args.checkpoint_config_path)
        # if args.checkpoint_name:
        #     if not self.checkpoint_config:
        #         raise ValueError("Must have a checkpoint config if supplying --checkpoint_name")
        #     if args.checkpoint_name not in self.checkpoint_config:
        #         raise ValueError(f"checkpoint name {args.checkpoint_name} is not in the checkpoint config file given.")
        #     else:
        #         if not args.from_checkpoint_path:
        #             args.from_checkpoint_path = self.checkpoint_config[args.checkpoint_name]["path"]
        #         if not args.checkpoint_type:
        #             args.checkpoint_type = self.checkpoint_config[args.checkpoint_name]["checkpoint_type"]
        #         if not args.arch:
        #             args.arch = self.checkpoint_config[args.checkpoint_name]["arch"]

        self.binary = True
        if args.dset == 'awa2':
            self.num_classes = 85
        else:
            self.num_classes = 40
        self.label_indices = None
        if args.label_indices:
            celeba_class_names = [i for i in map(celeba_classes().__getitem__, args.label_indices)]
            self.num_classes = len(args.label_indices)
            self.label_indices = args.label_indices
            print(f"Only predicting the follwoing celeba_classes: {celeba_class_names}")

        self.model_config = {'arch': args.arch, 'dataset': args.dset}
        self.model = get_model(args.arch, dataset=args.dset, pretrained=args.pretrained,
                               num_classes=self.num_classes,
                               deepsparse_wrapper=False,
                               wrapper_input_shape=(args.batch_size, 3, 224, 224))
        self.model = get_wrapped_model(self.model)
        print(self.model.state_dict().keys())
        print(self.model.__class__)

        self.config = args.config_path if isinstance(args.config_path, dict) else read_config(args.config_path)
        optimizer_dict, lr_scheduler_dict = None, None
        if args.from_checkpoint_path is not None:
            # load_checkpoint_fn = None
            # if not args.checkpoint_type or args.checkpoint_type=='ours':
            #     load_checkpoint_fn = load_checkpoint
            # else:
            #     raise ValueError("Checkpoint type must be 'ours'.")
            model = load_checkpoint(
                    args.from_checkpoint_path, self.model, self.transfer_config["load_modules"] if self.transfer_config else None)
            self.model = model


        self.logging_function = print
        if args.use_wandb:
            import wandb
            self.logging_function = wandb.log

        self.data = (args.dset, args.dset_path)
        self.n_epochs = args.epochs
        self.num_workers = args.workers
        self.batch_size = args.batch_size
        self.num_samples = args.num_samples
        self.warmup_epochs = args.warmup_epochs
        self.reset_momentum_after_recycling = args.reset_momentum_after_recycling
        self.device = args.device
        self.initial_epoch = 0
        self.best_val_acc = {"sparse": {"sparsity": 0., "val_acc": 0.}, "dense": 0.}
        self.current_sparsity = 0
        self.pruned_state = 'dense'
        self.reset_momentum = None
        self.training_stats_freq = args.training_stats_freq

        # for mixed precision training:
        if args.fp16:
            self.fp16_scaler = torch.cuda.amp.GradScaler()
            logging.info("==============!!!!Train with mixed precision (FP 16 enabled)!!!!================")
        else:
            self.fp16_scaler = None

        self.masks_folder = os.path.join(args.run_dir, 'pruned_masks')
        os.makedirs(self.masks_folder, exist_ok=True)

        self.recompute_bn_stats = args.recompute_bn_stats

        use_data_aug = True

        # Define datasets
        if self.data[0] == "backdoorceleba":
            print("Using backdoor for train \t", args.backdoor_type_train)
            print("Using backdoor for test \t", args.backdoor_type_test)
            self.backdoor_type_train = args.backdoor_type_train
            self.backdoor_type_test = args.backdoor_type_test
            self.in_backdoor_folder = args.backdoor_folder
            self.out_backdoor_folder = os.path.join(args.run_dir, f'backdoor_ids_label{args.backdoor_label}')
            self.backdoor_label = args.backdoor_label
            self.backdoor_fracs_train = args.backdoor_fracs_train
            self.backdoor_fracs_test = args.backdoor_fracs_test
            os.makedirs(self.out_backdoor_folder)
            self.data_train, self.data_val, self.data_test = get_datasets(*self.data,
                    use_data_aug=use_data_aug, interpolation=args.interpolation,
                    return_test=True,
                    label_indices=self.label_indices,
                    backdoor_type_train=self.backdoor_type_train,
                    backdoor_type_test=self.backdoor_type_test,
                    in_backdoor_folder=self.in_backdoor_folder,
                    out_backdoor_folder=self.out_backdoor_folder,
                    backdoor_label=args.backdoor_label, backdoor_fracs_train=args.backdoor_fracs_train, backdoor_fracs_test=args.backdoor_fracs_test)
        else:
            self.data_train, self.data_val, self.data_test = get_datasets(*self.data,
                    use_data_aug=use_data_aug, interpolation=args.interpolation,
                    return_test=True, label_indices=args.label_indices)

        self.train_loader = DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        self.val_loader = DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)
        self.test_loader = DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)
        
        # Freeze any modules that we don't plan to train.
        update_modules = None
        if self.transfer_config is not None:
            update_modules = set()
            frozen_modules = set()
            prefix = ''
            for name, p in self.model.named_parameters():
                frozen = False
                for m in self.transfer_config["freeze_modules"]:
                    if p.requires_grad and name.startswith(prefix + m):
                        p.requires_grad = False
                        print("turned off", name)
                        frozen = True
                        break
                if not frozen:
                    update_modules.add(name)
                else:
                    frozen_modules.add(name)
            update_modules = list(update_modules)
        if args.device.type == 'cuda':
            if args.manual_seed is not None:
                torch.cuda.manual_seed(args.manual_seed)
            self.model = torch.nn.DataParallel(self.model, device_ids=args.gpus)
            self.model.to(args.device)
        

        self.pruners = build_pruners_from_config(self.model, self.config)
        self.recyclers = build_recyclers_from_config(self.model, self.config)

        self.track_weight_hist = args.track_weight_hist

        self.trainers = [build_training_policy_from_config(self.model, self.config, 
                                                           trainer_name=DEFAULT_TRAINER_NAME,
                                                           fp16_scaler=self.fp16_scaler,
                                                           update_modules=update_modules, binary=self.binary)]
 
        if optimizer_dict is not None and lr_scheduler_dict is not None:
            self.initial_epoch = epoch
            self.trainers[0].optimizer.load_state_dict(optimizer_dict)
            self.trainers[0].lr_scheduler.load_state_dict(lr_scheduler_dict)
        
        self.setup_logging(args)
        self.schedule_prunes_and_recycles()

        # Add small check for how many filters are sparse
        if args.check_filters:
            model = get_unwrapped_model(self.model)
            conv2d_layers = [(a,b) for a,b in model.named_modules() if isinstance(b, torch.nn.Conv2d)]
            total_filters = 0.
            total_zero_filters = 0.
            for i in range(len(conv2d_layers)):
                num_layer = conv2d_layers[i][0]
                weights = conv2d_layers[i][1].weight.data
                abs_weights = weights.abs()
                filters_sums = abs_weights.sum(dim=(1,2,3))
                zero_filters = (filters_sums==0.).sum()
                num_filters = filters_sums.numel()
                total_filters += num_filters
                total_zero_filters += zero_filters
                print(f'{num_layer}: \t {zero_filters}/{num_filters} ({100. * zero_filters/num_filters})%')

            print(f"total zero filters: \t {total_zero_filters}/{total_filters} ({100*total_zero_filters/total_filters})%")

        if args.export_onnx is True:
            self.model = get_unwrapped_model(self.model)
            self.model.eval()
            onnx_batch = 1
            x = torch.randn(onnx_batch, 3, 224, 224, requires_grad=True).to(args.device)
            if args.onnx_nick:
                onnx_nick = args.onnx_nick
            else:
                onnx_nick = 'resnet_pruned.onnx'

            torch.onnx.export(self.model.module,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  onnx_nick,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

            print("ONNX EXPORT COMPLETED. EXITING")


    def setup_logging(self, args):
        self.logging_level = args.logging_level
        self.checkpoint_freq = args.checkpoint_freq
        self.exp_dir = args.exp_dir
        self.run_dir = args.run_dir

    def schedule_prunes_and_recycles(self):
        pruners_config = 'pruners' in self.config and self.config['pruners']
        self.pruning_epochs = {}
        if pruners_config:
            for i, [pruner, config] in enumerate(pruners_config.items()):
                start, freq, end = config["epochs"]
                for epoch in range(start, end, freq):
                    if epoch in self.pruning_epochs:
                        logging.info(f"o.O - already have a pruner for epoch {epoch}")
                        continue
                    self.pruning_epochs[epoch] = i
            logging.info("Pruning epochs are %s", str(self.pruning_epochs))
        recyclers_config =  'recyclers' in self.config and self.config['recyclers']
        self.recycling_epochs = {}
        if recyclers_config:
            for i, [recycler, config] in enumerate(recyclers_config.items()):
                start, freq, end = config["epochs"]
                for epoch in  range(start, end, freq):
                    if epoch in self.recycling_epochs:
                        logging.warn(f"Already have a recycler for epoch {epoch}")
                        continue
                    if epoch in self.pruning_epochs:
                        logging.warn(f"Already have a pruner for epoch {epoch}")
                        continue
                    self.recycling_epochs[epoch] = i
            logging.info("Recycling epochs are %s", str(self.recycling_epochs))


    # noinspection SyntaxError,Annotator
    def run_policies_for_method(self, policy_type: str, method_name: str, **kwargs):
        if 'agg_func' not in kwargs:
            agg_func = None
        else:
            agg_func = kwargs['agg_func']
            del kwargs['agg_func']
        res = []
        for policy_obj in getattr(self, f"{policy_type}s"):
            retval = getattr(policy_obj, method_name)(**kwargs)
            res.append(retval)

        if policy_type == 'pruner' and method_name == 'on_epoch_begin':
            is_active, meta = [r[0] for r in res], [r[1] for r in res]
            if any(is_active) and self.recompute_bn_stats:
                dataloader = DataLoader(get_datasets(*self.data)[0], batch_size=self.batch_size, shuffle=True)
                recompute_bn_stats(self.model, dataloader, self.device)
            return meta
        return res if agg_func is None else agg_func(res)


    def pruned_masks_similarity(self, epoch):
        pruned_masks_dict = collections.defaultdict(dict)
        total_masks_diff = 0.
        total_prunable_params = 0.
        
        for module_name, module in self.model.named_modules():
            # only the WrappedLayer instances have weight_mask
            if not isinstance(module, WrappedLayer):
                continue 

            w_size = module.weight.data.numel()
            if module.bias is not None:
                b_size = module.bias.data.numel()
            
            # ignore the layers that are not pruned
            if (module.weight_mask.sum().item() == w_size):
                continue
            
            pruned_masks_dict[module_name]['weight'] = module.weight_mask
            total_prunable_params += w_size

            if module.bias_mask is not None:
                pruned_masks_dict[module_name]['bias'] = module.bias_mask
                total_prunable_params += b_size
            
            # check the mask similarity if the mask at previous pruning point exists
            # the current difference between the masks is the Hamming distance
            if hasattr(module, 'prev_weight_mask'):
                sparsity_curr_mask_w = (1. - module.weight_mask).sum().item()
                sparsity_prev_mask_w = (1. - module.prev_weight_mask).sum().item()
                sparsity_diff_w = sparsity_curr_mask_w - sparsity_prev_mask_w
                module_masks_diff_w = (module.weight_mask - module.prev_weight_mask).abs().sum().item()
                module_masks_diff_w -= sparsity_diff_w
                total_masks_diff += module_masks_diff_w
                logging.info(f'different mask elements for {module_name}.weight: {module_masks_diff_w} / {w_size}')
                if module.bias_mask is not None:
                    sparsity_curr_mask_b = (1. - module.bias_mask).sum().item()
                    sparsity_prev_mask_b = (1. - module.prev_bias_mask).sum().item()
                    sparsity_diff_b = sparsity_curr_mask_b - sparsity_prev_mask_b
                    module_masks_diff_b = (module.bias_mask - module.prev_bias_mask).abs().sum().item()
                    module_masks_diff_b -= sparsity_diff_b
                    total_masks_diff += module_masks_diff_b
                    logging.info(f'different mask elements for {module_name}.bias: {module_masks_diff_b} / {b_size}')
            # update the masks for the previous pruning point
            module.set_previous_masks()
        torch.save(pruned_masks_dict, os.path.join(self.masks_folder, f'masks_epoch{epoch}.pt'))
        relative_masks_diff = 100 * total_masks_diff / total_prunable_params 
        self.logging_function({'epoch':epoch, 'relative mask difference (%)':relative_masks_diff})


    def end_epoch(self, epoch, val_loader):
        sparsity_dicts = self.run_policies_for_method('pruner',
                                                      'measure_sparsity')

        num_zeros, num_params = get_total_sparsity(self.model)

        self.training_progress.sparsity_info(epoch,
                                             sparsity_dicts,
                                             num_zeros,
                                             num_params,
                                             self.logging_function)
        self.run_policies_for_method('trainer',
                                     'on_epoch_end',
                                     bn_loader=None,
                                     swap_back=False,
                                     device=self.device,
                                     epoch_num=epoch,
                                     agg_func=lambda x: np.sum(x, axis=0))


    def get_eval_stats(self, epoch, dataloader, type = 'val', avg_per_class=False):
        res = self.run_policies_for_method('trainer', 'eval_model',
                                                     loader=dataloader,
                                                     device=self.device,
                                                     epoch_num=epoch,
                                                     num_classes = self.num_classes,
                                                     avg_per_class=avg_per_class)
        loss, correct, pred_pos, per_label_acc, prec, rec, fpr, fnr,  f1, auc, p50, p75, p90, p95, label_freq, pred_freq  = res[0]
        self.logging_function({'epoch':epoch, f"pred_pos":pred_pos})

        if self.model_config["dataset"] in ('celeba', 'backdoorceleba', 'full_celeba'):
            if self.label_indices:
                classes = [i for i in map(celeba_classes().__getitem__, self.label_indices)]
            else:
                classes = celeba_classes()
        elif self.model_config["dataset"] == "awa2":
            if self.label_indices:
                classes = [i for i in map(awa_classes().__getitem__, self.label_indices)]
            else:
                classes = awa_classes()
        else:
            raise ValueError(f"Unknown binary dataset {self.model_config['dataset']}")
        for i, klass in enumerate(classes):
            continue
            self.logging_function({'epoch':epoch, f"{klass}_acc":per_label_acc[i]})
            self.logging_function({'epoch':epoch, f"{klass}_precision":prec[i]})
            self.logging_function({'epoch':epoch, f"{klass}_recall":rec[i]})
            self.logging_function({'epoch':epoch, f"{klass}_f1":f1[i]})
            self.logging_function({'epoch':epoch, f"{klass}_fpr":fpr[i]})
            self.logging_function({'epoch':epoch, f"{klass}_fnr":fnr[i]})
            self.logging_function({'epoch':epoch, f"{klass}_auc":auc[i]})
            self.logging_function({'epoch':epoch, f"{klass}_prec_at_50":p50[i]})
            self.logging_function({'epoch':epoch, f"{klass}_prec_at_75":p75[i]})
            self.logging_function({'epoch':epoch, f"{klass}_prec_at_90":p90[i]})
            self.logging_function({'epoch':epoch, f"{klass}_prec_at_95":p95[i]})
            self.logging_function({'epoch':epoch, f"{klass}_true_freq":label_freq[i]})
            self.logging_function({'epoch':epoch, f"{klass}_pred_freq":pred_freq[i]})

        # log validation stats
        if type == 'val':
            self.logging_function({'epoch': epoch, type + ' loss':loss,
                type + ' acc':correct / len(dataloader.dataset),
                })
        logging.info({'epoch': epoch, type + ' loss':loss, type + ' acc':correct / len(dataloader.dataset)})

        if type == "val":
            self.training_progress.val_info(epoch, loss, correct, pred_pos)
        return loss, correct

    def export_predictions(self, model, data_type, loader, num_classes):
        model.eval()
        n_samples = len(loader.dataset)
        outputs = torch.zeros(n_samples, num_classes)
        start_index = 0
        with torch.no_grad():
            for i, (in_tensor, target) in enumerate(loader):
                batch_size=in_tensor.shape[0]
                in_tensor, target = in_tensor.to(self.device), target.to(self.device)
                output = model(in_tensor)
                outputs[start_index: start_index+batch_size] = output
                start_index += batch_size
        model.train()
        print(self.model.state_dict().keys())
        return outputs


    def run(self):
        train_loader = self.train_loader
        val_loader = self.val_loader
        test_loader = self.test_loader

        total_train_flops = 0

        self.training_progress = TrainingProgressTracker(self.initial_epoch,
                                                         len(train_loader),
                                                         len(val_loader.dataset),
                                                         self.training_stats_freq,
                                                         self.run_dir)

        total_train_time = 0.
        for epoch in range(self.initial_epoch, self.n_epochs):
            self.model.train()
            # noinspection Annotator
            logging.info(f"Starting epoch {epoch} with number of batches {len(train_loader)}")

            subset_inds = np.random.choice(len(self.data_train), self.num_samples, replace=False)

            metas = {}

            if epoch in self.pruning_epochs:
                metas = self.run_policies_for_method('pruner',
                                                     'on_epoch_begin',
                                                     num_workers=self.num_workers,
                                                     dset=self.data_train,
                                                     subset_inds=subset_inds,
                                                     device=self.device,
                                                     batch_size=64,
                                                     epoch_num=epoch)
                # track the differences in masks between consecutive mask updates
                self.pruned_masks_similarity(epoch)
                self.pruned_state = "sparse"
                level = None
                for meta in metas:
                    if "level" in meta:
                        level = meta["level"]
                        self.current_sparsity = level
                        break

            if self.track_weight_hist:
                metas = get_weights_hist_meta(self.model, metas)

            
            self.training_progress.meta_info(epoch, metas)
            if epoch in self.recycling_epochs:
                metas = self.run_policies_for_method('recycler',
                                             'on_epoch_begin',
                                             num_workers=self.num_workers,
                                             dset=self.data_train,
                                             subset_inds=subset_inds,
                                             device=self.device,
                                             batch_size=64,
                                             epoch_num=epoch,
                                             optimizer=self.trainers[0].optimizer)

                for meta in metas:
                    if "level" in meta[1]:
                        self.current_sparsity = meta[1]["level"]
                        break
                self.pruned_state = "dense"
                self.reset_momentum=self.reset_momentum_after_recycling

            
            epoch_loss, epoch_acc = 0., 0.
            n_samples = len(train_loader.dataset)
            start_epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                start = time.time()
                val  = self.run_policies_for_method('trainer',
                                                         'on_minibatch_begin',
                                                         minibatch=batch,
                                                         device=self.device,
                                                         loss=0.,
                                                         agg_func=None)
                loss, acc, pred_pos  = val[0]
                epoch_acc += acc * batch[0].size(0) / n_samples

                def sum_pytorch_nums(lst):
                    res = 0.
                    for el in lst:
                        res = res + el
                    return res

                epoch_loss += loss.item() * batch[0].size(0) / n_samples

                self.run_policies_for_method('trainer',
                                             'on_parameter_optimization',
                                             loss=loss,
                                             reset_momentum=self.reset_momentum,
                                             epoch_num=epoch)
                # If we were supposed to reset momentum, we just did so.
                self.reset_momentum=False
                self.run_policies_for_method('pruner',
                                             'after_parameter_optimization',
                                             model=self.model)
                self.run_policies_for_method('recycler',
                                             'after_parameter_optimization',
                                             model=self.model)

                ############################### tracking the training statistics ############################
                self.training_progress.step(loss=loss,
                                            acc=acc,
                                            pred_pos=pred_pos,
                                            time=time.time() - start,
                                            lr=self.trainers[0].optim_lr)
                #############################################################################################
            
            epoch_time = time.time() - start_epoch_time
            total_train_time += epoch_time
            logging.info(f"Epoch {epoch} time: {epoch_time}")
            self.logging_function({'epoch': epoch, 'time': epoch_time})

            self.logging_function({'epoch': epoch, 'train loss': epoch_loss, 'train acc': epoch_acc})

            self.end_epoch(epoch, val_loader)
            predictions = {}
            for data_type, loader in [["val", val_loader], ["test", test_loader]]:
                predictions[data_type] = self.export_predictions(self.model, data_type, loader, self.num_classes)

            ##################################################
            # Record FLOPs :
            #forward_flops_per_sample, list_layers, module_names = get_macs_sparse(self.data[0], self.device, self.model)
            # calculate FLOPs for backward as in https://arxiv.org/pdf/1911.11134.pdf (Appendix H)
            forward_flops_per_sample, list_layers, module_names = get_macs_sparse(self.data[0], self.device, self.model)
            
            backward_flops_per_sample = 2 * forward_flops_per_sample
            trn_flops_per_sample = forward_flops_per_sample + backward_flops_per_sample
            total_trn_flops_epoch = trn_flops_per_sample * n_samples

            total_train_flops += total_trn_flops_epoch

            self.logging_function({'epoch': epoch, 'train FLOPs per sample': trn_flops_per_sample})
            self.logging_function({'epoch': epoch, 'train FLOPs epoch': total_trn_flops_epoch})
            self.logging_function({'epoch': epoch, 'test FLOPs per sample': forward_flops_per_sample})
            ####################################################u


            self.logging_function({'epoch': epoch, 'test FLOPs per sample': forward_flops_per_sample})
            ####################################################

            current_lr = self.trainers[0].optim_lr
            self.logging_function({'epoch': epoch, 'lr': current_lr})
            
            val_loss, val_correct = self.get_eval_stats(epoch, val_loader, avg_per_class=True)

            val_acc = 1.0 * val_correct / len(val_loader.dataset)
            best_sparse = False
            best_dense = False
            scheduled = False
            if self.pruned_state == "sparse":
                if int(self.current_sparsity * 10000) > int(self.best_val_acc["sparse"]["sparsity"] * 10000):
                    best_sparse = True
                    self.best_val_acc["sparse"]["sparsity"] = self.current_sparsity
                    self.best_val_acc["sparse"]["val_acc"] = val_acc
                    logging.info("saving best sparse checkpoint with new sparsity")
                elif int(self.current_sparsity  * 10000) == int(self.best_val_acc["sparse"]["sparsity"] * 10000) \
                   and val_acc > self.best_val_acc["sparse"]["val_acc"]:
                    best_sparse = True
                    self.best_val_acc["sparse"]["val_acc"] = val_acc
                    logging.info("saving best sparse checkpoint")
            if self.pruned_state == "dense" and val_acc > self.best_val_acc["dense"]:
                best_dense = True
                self.best_val_acc["dense"] = val_acc
            if (epoch + 1)  % self.checkpoint_freq == 0:
                logging.info("scheduled checkpoint")
                scheduled = True

            save_checkpoint(epoch, self.model_config, copy.deepcopy(self.model), predictions,  self.trainers[0].optimizer,
                                self.trainers[0].lr_scheduler, self.run_dir,
                            is_best_sparse=best_sparse, is_best_dense=best_dense,
                            is_scheduled_checkpoint=scheduled)

        # end_epoch even if epochs==0
        logging.info('====>Final summary for the run:')
        val_correct = self.end_epoch(self.n_epochs, self.val_loader)
        logging.info(f'Total training time: {total_train_time}')
        get_flops(self.data[0], self.device, self.model)



        return val_correct, len(val_loader.dataset)

