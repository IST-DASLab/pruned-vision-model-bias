import os
import importlib
import numpy as np
import pandas as pd

import io
import base64
import pickle as pkl

from .projects import PROJECTS
import importlib
from .runs import compute_pos_neg_matrices 


def celeba_classes():
    return ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]


identity_labels = [20, 39, 13, 26]
backdoor_labels = [2, 9, 31, 38, 25, 7]
single_attr_labels = [2, 9, 25, 7, 22, 28, 3, 31, 38]
backdoor_types = ["grayscale", "yellow_square"]


def compute_bias_amplification(targets, predictions,
        protected_attribute_id, attribute_id,
        pos_fracs_df, neg_fracs_df):
    if attribute_id == protected_attribute_id:
        return None
    protected_attr = targets[:,protected_attribute_id].ravel()
    protected_pos = np.argwhere(protected_attr == 1)
    if targets[protected_pos, attribute_id].sum() < 10:
        print(f"too few positive examples for attribute {celeba_classes()[attribute_id]}")
        return None
    protected_neg = np.argwhere(protected_attr == 0)
    if targets[protected_neg, attribute_id].sum() < 10:
        print(f"too few negative examples for attribute {celeba_classes()[attribute_id]}")
        return None
    total_attr = None
    protected_pos_predicts = predictions[protected_pos] 
    protected_neg_predicts = predictions[protected_neg]
    total_attr = predictions.sum()
    protected_pos_targets = targets[protected_pos, attribute_id] 
    protected_neg_targets = targets[protected_neg, attribute_id]
  
    # If the attribute frequency is very close for positive and negative identity
    # attribute, doesn't make much sense to compute BA.
    pos_frac = pos_fracs_df.loc[attribute_id, protected_attribute_id]
    neg_frac = neg_fracs_df.loc[attribute_id, protected_attribute_id]
    if np.abs(pos_frac-neg_frac)/np.minimum(pos_frac, neg_frac) < 0.1:
        print(f"Diff is too small for attribute {celeba_classes()[attribute_id]}")
        return None
    pos_frac = pos_fracs_df.loc[attribute_id, protected_attribute_id]
    neg_frac = neg_fracs_df.loc[attribute_id, protected_attribute_id]
    if pos_frac > neg_frac:
        ba = protected_pos_predicts.sum()/total_attr - \
             protected_pos_targets.sum()/targets[:, attribute_id].sum()
    else:
        ba = protected_neg_predicts.sum()/total_attr - \
             protected_neg_targets.sum()/(targets[:, attribute_id]).sum()
    return ba


def compute_backdoor_bias_amplification(targets, predictions, backdoor_idxs, attribute_id, single_label=False):
    if backdoor_idxs is None or len(backdoor_idxs)==0:
        return None
    all_idxs = set(range(len(targets)))
    clean_idxs = all_idxs - set(backdoor_idxs)
    total_attr = None
    if single_label:
        backdoor_predicts = predictions[list(backdoor_idxs)]
        clean_predicts = predictions[list(clean_idxs)]
        total_attr = predictions.sum()
    else:
        backdoor_predicts = predictions[list(backdoor_idxs), attribute_id]
        clean_predicts = predictions[list(clean_idxs), attribute_id]
        total_attr = predictions[:,attribute_id].sum()
    backdoor_targets = targets[list(backdoor_idxs), attribute_id]
    clean_targets = targets[list(clean_idxs), attribute_id]
    ba = backdoor_predicts.sum()/total_attr - \
         backdoor_targets.sum()/targets[:, attribute_id].sum()
    return ba        


def compute_bas(run, targets, pos_fracs_df, neg_fracs_df, single_label=True):
    for identity_label in identity_labels:
        identity_label_name = celeba_classes()[identity_label]
        if single_label:
            run[f"{identity_label_name}-bas"] = \
            compute_bias_amplification(targets, run["test_predictions"], identity_label, run['label'], pos_fracs_df, neg_fracs_df)

        else:
            run[f"{identity_label_name}-bas"] = np.zeros(40)
            run[f"{identity_label_name}-bas"][:] = np.nan
            for i in range(40):
                if i == identity_label:
                    continue
                run[f"{identity_label_name}-bas"][i] = \
                compute_bias_amplification(targets, run["test_predictions"], identity_label, i)


def compute_cbas(run, targets, pos_fracs_df, neg_fracs_df, single_label=True):
    identity_label_name = celeba_classes()[run['id-label']]
    run[f"{identity_label_name}-bas"] = compute_bias_amplification(targets, run["test_predictions"], run['id-label'], run['label'], single_label=True)
       

def compute_bbas(run, targets, single_label=True):
    run["test_predictions"] = run["test_outputs"] > 0
    test_backdoor_ids = np.loadtxt(run['backdoor_test'], dtype=int)
    if single_label:
        run['bas'] = compute_backdoor_bias_amplification(targets, run["test_predictions"], test_backdoor_ids, 
                                            run['label'], single_label=True)
    else:
        if backdoor_all:
            run["bbas-all"] = np.zeros(40)
            run["bbas-all"][:] = np.nan
            for i in range(40):
                run["bbas-all"][i] = compute_backdoor_bias_amplification(targets, run["tbackdoorest_predictions"], test_backdoor_ids, i)
        else:
            run['bas'] = compute_backdoor_bias_amplification(targets, run["test_predictions"], test_backdoor_ids, run['label'])
    
            
def compute_errors_single(run, targets):
    label_targets = targets[:, run["label"]]
    run["acc"] = np.equal(run["test_predictions"], label_targets).mean()
    run["pred_pos"] = np.mean(run["test_predictions"])
    

def get_test_labels():
    split_map = {
        "train": 0,
        "valid": 1,
        "test": 2,
    }
    splits = pd.read_csv("/home/Datasets/celeba/list_eval_partition.txt", delim_whitespace=True, header=None, index_col=0).to_numpy()
    labels = pd.read_csv("/home/Datasets/celeba/list_attr_celeba.txt", delim_whitespace=True, header=1).to_numpy()
    labels = labels[ splits.ravel()==2]
    return labels>0


def get_val_labels():
    split_map = {
        "train": 0,
        "valid": 1,
        "test": 2,
    }
    splits = pd.read_csv("/home/Datasets/celeba/list_eval_partition.txt", delim_whitespace=True, header=None, index_col=0).to_numpy()
    labels = pd.read_csv("/home/Datasets/celeba/list_attr_celeba.txt", delim_whitespace=True, header=1).to_numpy()
    labels = labels[ splits.ravel()==1]
    return labels>0


# evaluate sparsity for a model
def get_sparsity(model):
    total_zeros = 0.
    total_params = 0.
    for p in model.parameters():
        total_zeros += (p.data==0.).sum().item()
        total_params += p.data.numel()
    total_sparsity = total_zeros / total_params
    return total_sparsity


def get_runs_for_project_backdoor(project):
    project = PROJECTS[project]
    backdoor_attrs = ['blond', 'smiling', 'oval-face', 'big-nose']
    attrs_dict = {'blond': 9, 
                  'smiling':31,
                  'oval-face': 25,
                  'oval_face': 25,
                  'big-nose': 7,
                  'big_nose': 7,
                  'big_lips': 6,
                  'mustache': 22,
                  'receding-hairline': 28,
                  'receding_hairline': 28,
                  'bags-under-eyes': 3,
                  'wearing_necktie': 38,
                  'attractive': 2,
                  'male': 20,
                  'young': 39
                 }

    strategy = 'GMP-RI'

    dense_runs_location = os.path.join("./runs/backdoor_runs", project['dset'], project['arch']+"", "Dense")
    sparse_runs_location = os.path.join("./runs/backdoor_runs", project['dset'], project['arch']+"", strategy)
    runs = {}
    for loc in [dense_runs_location, sparse_runs_location]:
        for entry in os.scandir(loc):
            if entry.is_dir():
                attr = entry.name
                if attr not in runs:
                    runs[attr] = []
                for entry in os.scandir(os.path.join(loc, attr)):
                    backdoor_type = entry.name
                    for entry in os.scandir(os.path.join(loc, attr, backdoor_type)):
                        sparsity = entry.name
                        for entry in os.scandir(os.path.join(loc, attr, backdoor_type, sparsity)):
                            run_name = entry.name
                            attr_run = attr
                            run = {
                                    'sparsity': float(sparsity),
                                    'strategy': 'Dense' if float(sparsity) == 0 else strategy,
                                    'backdoor_type': backdoor_type,
                                    'type': sparsity,
                                    'label': attrs_dict[attr],
                                    'name': run_name,
                                    'arch': project['arch'],
                                    'dataset': project['dset'],
                                    'combined': False

                                    }
                            run['location'] = os.path.join(loc, attr, run['backdoor_type'], str(int(run['sparsity'])), run['name'])
                            run['run_dir'] = run['location']
                            run['backdoor_folder'] = os.path.join(run['location'], f'backdoor_ids_label{run["label"]}')
                            run['backdoor_test'] = os.path.join(run['backdoor_folder'], 'backdoor_ids_test.txt')
                            run['backdoor_train'] = os.path.join(run['backdoor_folder'], 'backdoor_ids_train.txt')
                            runs[attr].append(run)
    return runs

def get_runs_for_project_combined(project):
    project = PROJECTS[project]
    attrs_dict = {'blond': 9, 
                  'smiling':31,
                  'oval-face': 25,
                  'oval_face': 25,
                  'big-nose': 7,
                  'big_nose': 7,
                  'big_lips': 6,
                  'mustache': 22,
                  'receding-hairline': 28,
                  'receding_hairline': 28,
                  'bags-under-eyes': 3,
                  'wearing_necktie': 38,
                  'attractive': 2,
                  'male': 20,
                  'young': 39
                 }
    attrs = ['male-oval_face', 'male-big_nose', 'male-big_lips', 'young-big_nose', 'young-mustache', 'young-receding_hairline']
    dense_runs_location = os.path.join("./combined_runs", project['dset'], project['arch']+"", "Dense")
    sparse_runs_location = os.path.join("./combined_runs", project['dset'], project['arch']+"", strategy)
    runs = {}
    for loc in [dense_runs_location, sparse_runs_location]:
        for entry in os.scandir(loc):
            if entry.is_dir():
                attr = entry.name
                if attr not in runs:
                    runs[attr] = []
                for entry in os.scandir(os.path.join(loc, attr, backdoor_type)):
                    sparsity = entry.name
                    for entry in os.scandir(os.path.join(loc, attr, backdoor_type, sparsity)):
                        run_name = entry.name
                        attr_run = attr
                        print("the attr is", attr)
                        if attr=='wearing_necktie':
                            attr_run = attr.split('_')[-1]
                        run = {
                                'sparsity': float(sparsity),
                                'strategy': 'Dense' if float(sparsity) == 0 else strategy,
                                'label': attrs_dict[attr],
                                'name': run_name,
                                'type': sparsity,
                                'arch': project['arch'],
                                'dataset': project['dset'],
                                'combined': True

                                }
                        run['location'] = os.path.join("backdoor_runs", run['dataset'], run['arch']+"", run['strategy'], attr, str(run['sparsity']), run['name'])
                        run['run_dir'] = run['location']
                        runs[attr].append(run)

    return runs

def get_runs_for_project_single(project):
    attrs_dict = {'blond': 9, 
                  'smiling':31,
                  'oval-face': 25,
                  'oval_face': 25,
                  'big-nose': 7,
                  'big_nose': 7,
                  'big_lips': 6,
                  'mustache': 22,
                  'receding-hairline': 28,
                  'receding_hairline': 28,
                  'bags-under-eyes': 3,
                  'wearing_necktie': 38,
                  'attractive': 2,
                  'male': 20,
                  'young': 39
                 }
    single_attrs = ['blond', 'smiling', 'oval-face', 'big-nose', 'mustache', 'receding-hairline', 'bags-under-eyes']
    project = PROJECTS[project]
    dense_runs_location = os.path.join("./runs/single_label_runs", project['dset'], project['arch']+"", "Dense")
    strategy = 'GMP-RI'
    sparse_runs_location = os.path.join("./runs/single_label_runs", project['dset'], project['arch']+"", strategy)
    runs = {}
    for loc in [dense_runs_location, sparse_runs_location]:
        for entry in os.scandir(loc):
            if entry.is_dir():
                attr = entry.name
                if attr not in runs:
                    runs[attr] = []
                for entry in os.scandir(os.path.join(loc, attr)):
                    sparsity = entry.name
                    for entry in os.scandir(os.path.join(loc, attr, sparsity)):
                        run_name = entry.name
                        attr_run = attr
                        attr_run = attr.split('-')
                        run = {
                                'sparsity': float(sparsity),
                                'strategy': 'Dense' if float(sparsity) == 0 else strategy,
                                'label': attrs_dict[attr],
                                'type': f'{sparsity}_{strategy}',
                                'name': run_name,
                                'arch': project['arch'],
                                'dataset': project['dset'],
                                'combined': False
                                }
                        run['location'] = os.path.join(loc, attr, str(int(run['sparsity'])), run['name'])
                        run['run_dir'] = run['location']
                        runs[attr].append(run)

    return runs


def get_thresholds(outputs, labels):
    pos_per = np.sum(labels)
    thresholds = np.ones(pos_per.shape)
    thresholds = \
                np.partition(outputs, -1*round(pos_per))[-1*round(pos_per)]
    return thresholds


def load_run_details(run, test_labels, pos_fracs_df, neg_fracs_df,  best=True, threshold_adjusted=False, use_cache=True):
    print(run['run_dir'])
    if test_labels is None:
        test_labels = get_test_labels()
    if best:
        test_outputs_file = os.path.join(run["run_dir"], "test_outputs_best.txt")
        cached_path = os.path.join(run["run_dir"], "run_stats_best.pkl")
    else:
        test_outputs_file = os.path.join(run["ruu_dir"], "test_outputs_last.txt")
        cached_path = os.path.join(run["run_dir"], "run_stats_best.pkl")
    if use_cache and os.path.exists(cached_path):
        with open (cached_path, 'rb') as f:
            return pkl.load(f)
    elif os.path.exists(test_outputs_file):
        thresholds = np.zeros(1)
        if threshold_adjusted:
            print("threshold adjusting")
            val_labels = get_val_labels()
            if not os.path.exists(run["run_dir"] + "/" + "valid_outputs_best.txt"):
                print("lol here", run["run_dir"])
                return run
            run["val_outputs"] = np.loadtxt(run["run_dir"] + "/" + "valid_outputs_best.txt")
            thresholds = get_thresholds(run["val_outputs"],val_labels[:,run['label']])
        run["thresholds"] = thresholds
        
        run[f"test_outputs"] = np.loadtxt(test_outputs_file)
        run["test_predictions"] = run["test_outputs"] > thresholds
        if 'backdoor_test' in run.keys():
            compute_bbas(run, test_labels)
        elif False and run['combined']:
            compute_cbas(run, test_labels, pos_fracs_df, neg_fracs_df)
        else:
            compute_bas(run, test_labels, pos_fracs_df, neg_fracs_df)
        compute_errors_single(run, test_labels)
        with open (cached_path, 'wb') as f:
            pkl.dump(run, f)
        return run
    else:
        print("!!!!!!!!!!!!!!Not enough artifacts found for run ", run["run_dir"])
    return run


def get_run_summaries(runs, dataset, backdoor, threshold_adjusted=False, use_cache=True):
    test_labels = get_test_labels()
    pos_fracs_df, neg_fracs_df = compute_pos_neg_matrices(dataset)
    for attr, attr_runs in runs.items():
        for i, run in enumerate(attr_runs):
            runs[attr][i] = load_run_details(run, test_labels,
                                             pos_fracs_df, neg_fracs_df,
                                             threshold_adjusted=threshold_adjusted,
                                             use_cache=use_cache)
        runs[attr] = [v for v in runs[attr] if 'test_outputs' in v]
        print("there are this many runs for ", attr,  len(runs[attr]))
    return runs

