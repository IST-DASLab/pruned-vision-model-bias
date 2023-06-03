import os
import importlib
import math

import pandas as pd
import io
import numpy as np
import base64
import pickle as pkl
from .projects import PROJECTS
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression


def celeba_classes():
    return ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]


celeba_identity_labels = [20, 39, 13, 26]

def get_test_labels(dset, val = False):
    split_map = {
        "train": 0,
        "valid": 1,
        "test": 2,
    }
    splits = pd.read_csv("/home/Datasets/celeba/list_eval_partition.txt", delim_whitespace=True, header=None, index_col=0).to_numpy()
    labels = pd.read_csv("/home/Datasets/celeba/list_attr_celeba.txt", delim_whitespace=True, header=1).to_numpy()
    if val:
        labels = labels[ splits.ravel()==1]
    else:
        labels = labels[ splits.ravel()==2]
    return labels>0

def get_val_labels(dset):
    return get_test_labels(dset, val=True)


def get_train_labels(dset):
    split_map = {
        "train": 0,
        "valid": 1,
        "test": 2,
    }
    splits = pd.read_csv("/home/Datasets/celeba/list_eval_partition.txt", delim_whitespace=True, header=None, index_col=0).to_numpy()
    labels = pd.read_csv("/home/Datasets/celeba/list_attr_celeba.txt", delim_whitespace=True, header=1).to_numpy()
    labels = labels[ splits.ravel()==0]
    return labels>0


def compute_bias_amplification(targets, predictions,
        protected_attribute_id, attribute_id, dataset, pos_fracs_df, neg_fracs_df):
    attr_names = celeba_classes()
    if attribute_id == protected_attribute_id:
        return None
    protected_attr = targets[:,protected_attribute_id].ravel()
    protected_pos = np.argwhere(protected_attr == 1)
    if targets[protected_pos, attribute_id].sum() < 10:
        print(f"too few positive examples for attribute {attr_names[attribute_id]}")
        print("predictions", predictions[protected_pos, attribute_id].sum())
        return None
    
    protected_neg = np.argwhere(protected_attr == 0)
    if targets[protected_neg, attribute_id].sum() < 10:
        print(f"too few negative examples for attribute {attr_names[attribute_id]}")
        print("predictions", predictions[protected_neg, attribute_id].sum())
        return None
    protected_pos_predicts = predictions[protected_pos, attribute_id] 
    protected_neg_predicts = predictions[protected_neg, attribute_id]
    protected_pos_targets = targets[protected_pos, attribute_id] 
    protected_neg_targets = targets[protected_neg, attribute_id]
  
    # If the attribute frequency is very close for positive and negative identity
    # attribute, doesn't make much sense to compute BA.
    pos_frac = pos_fracs_df.loc[attribute_id, protected_attribute_id]
    neg_frac = neg_fracs_df.loc[attribute_id, protected_attribute_id]
    if np.abs(pos_frac-neg_frac)/np.minimum(pos_frac, neg_frac) < 0.1:
        print(f"Diff is too small for attribute {attr_names[attribute_id]}")
        return None
    if pos_frac > neg_frac:
        ba = protected_pos_predicts.sum()/predictions[:,attribute_id].sum() - \
             protected_pos_targets.sum()/targets[:, attribute_id].sum()
    else:
        ba = protected_neg_predicts.sum()/(predictions[:,attribute_id]).sum() - \
             protected_neg_targets.sum()/(targets[:, attribute_id]).sum()
    return ba


def compute_bas(run, targets, pos_fracs_df, neg_fracs_df):
    predictions = run["test_predictions"]
    identity_labels = celeba_identity_labels
    attr_names = celeba_classes()
    for identity_label in identity_labels:
        identity_label_name = attr_names[identity_label]
        run[f"{identity_label_name}-bas"] = np.zeros(len(attr_names))
        run[f"{identity_label_name}-bas"][:] = np.nan
        for i in range(len(attr_names)):
            label = attr_names[i]
            if i == identity_label:
                continue
            run[f"{identity_label_name}-bas"][i] = \
                compute_bias_amplification(targets, predictions, identity_label, i, run["dataset"], pos_fracs_df, neg_fracs_df)

            
def compute_errors(run, targets):
    predictions = run["test_predictions"]
    outputs = run["test_outputs"]
    identity_labels = celeba_identity_labels
    attr_names = celeba_classes()
    acc = np.mean(predictions==targets, axis=0)
    auc = np.zeros(acc.shape)
    probs = 1/(1 + np.exp(-outputs))
    for i in range(acc.shape[0]):
        fpr, tpr, thresholds = metrics.roc_curve(targets[:,i], probs[:, i], pos_label=1)
        auc[i] = metrics.auc(fpr, tpr)
    predpos = np.mean(predictions, axis=0)
    uncertainty=np.mean(np.abs(1/(1 + np.exp(-outputs))-0.5) < 0.4, axis=0)
    
    run[f"acc"] = acc
    run[f"auc"] = auc
    run[f"predpos"] = predpos
    run[f"uncertainty"]=uncertainty


def compute_interdependence(run):
    r_values = np.zeros(run['test_predictions'].shape[1])
    for i in range(run["test_predictions"].shape[1]):
        pred_col = run["test_predictions"][:,i]
        feature_columns = run["test_predictions"][:, [j for j in range(40) if j != i]]
        reg = LinearRegression().fit(feature_columns, pred_col)
        r_values[i] = reg.score(feature_columns, pred_col)
    run["interdependence"] = r_values


def compute_uncertainty_calibration(run, test_labels):
  run["test_pred_prob_bucket"] = np.floor(run["test_probabilities"]*10)/10
  run["ece"] = np.zeros(run["test_probabilities"].shape[1])
  for i in range(run["ece"].shape[0]):
    df = pd.DataFrame({"bucket": run["test_pred_prob_bucket"][:,i], "pred":run["test_probabilities"][:, i], "label": test_labels[:,i] })
    df = df.groupby("bucket").agg(["mean", "count"])
    total = df["pred", "count"].sum()
    run["ece"][i] = (np.abs(df["pred", "mean"] -df["label"]["mean"])*df["pred"]["count"]/total).sum()


def compute_rare_val_underpredict(run, test_pos_frac, train_pos_frac):
  run["rare_val_underpredict"] = np.ones_like(test_pos_frac)
  pred_prop = run["test_predictions"].mean(axis=0)
  for i in range(test_pos_frac.shape[0]):
    if train_pos_frac[i] > 0.5:      
      run["rare_val_underpredict"][i] = (1-pred_prop[i])/(1-test_pos_frac[i])
    else:
      run["rare_val_underpredict"][i] = pred_prop[i]/test_pos_frac[i]
  return run


def get_test_image_ids(dset):
    split_map = {
        "train": 0,
        "valid": 1,
        "test": 2,
    }
    splits = pd.read_csv("/home/Datasets/celeba/list_eval_partition.txt", delim_whitespace=True, header=None)
    splits.columns = ["img_ids", "split"]
    img_ids  = splits["img_ids"].to_numpy().ravel()[splits['split'].ravel()==2]
    return img_ids


def compute_pos_neg_matrices(dataset):
    attributes = celeba_classes()
    identity_attributes = celeba_identity_labels
    train_labels = get_train_labels(dataset)
    pos_fracs = np.zeros([train_labels.shape[1], len(identity_attributes)])
    neg_fracs = np.zeros([train_labels.shape[1], len(identity_attributes)])
    for i in range(pos_fracs.shape[0]):
        for j, attr in enumerate(identity_attributes):
            protected_pos_train = train_labels[np.argwhere(train_labels[:, attr] == 1), i]
            pos_fracs[i,j] = protected_pos_train.mean()
            protected_neg_train = train_labels[np.argwhere(train_labels[:, attr] == 0), i]
            neg_fracs[i,j] = protected_neg_train.mean()

    pos_fracs_df = pd.DataFrame(pos_fracs)
    pos_fracs_df.columns = identity_attributes
    neg_fracs_df = pd.DataFrame(neg_fracs)
    neg_fracs_df.columns = identity_attributes

    return pos_fracs_df, neg_fracs_df


def get_thresholds(outputs, labels):
    pos_per = np.sum(labels, axis=0)
    # Note that we do this on the raw outputs and not the sigmoids.
    thresholds = np.ones(pos_per.shape[0])
    for i in range(thresholds.shape[0]):
        thresholds[i] = \
                np.partition(outputs[:,i], -1*round(pos_per[i]))[-1*round(pos_per[i])]
    return thresholds


def get_runs_for_project(project_name, strategy):
    project = PROJECTS[project_name]
    dense_runs_location = os.path.join("./runs/joint_runs", project['dset'], project['arch']+"", "Dense")
    sparse_runs_location = os.path.join("./runs/joint_runs", project['dset'], project['arch']+"", strategy)
    runs = []
    for loc in [dense_runs_location, sparse_runs_location]:
        for entry in os.scandir(loc):
            if entry.is_dir():
                sparsity = entry.name
                for entry in os.scandir(os.path.join(loc, sparsity)):
                    run_name = entry.name
                    run = {
                            'location': os.path.join(loc, sparsity, run_name),
                            'run_dir': os.path.join(loc, sparsity, run_name),
                            'sparsity': float(sparsity),
                            'strategy': 'Dense' if sparsity == '0' else strategy,
                            'name': run_name,
                            'arch': project['arch'],
                            'dataset': project['dset'],
                            'type': f'{sparsity}_{strategy}'

                            }
                    runs.append(run)
    return runs


def load_test_outputs(run):
    run["test_outputs"] = np.loadtxt(run["run_dir"] + "/" + "test_outputs.txt")
    thresholds = np.zeros(run['test_outputs'].shape[0])
    run["test_predictions"] = np.zeros(run["test_outputs"].shape)
    for i in range(run["test_outputs"].shape[1]):
        run["test_predictions"][:, i] = run["test_outputs"][:, i] > thresholds[i]
    run["test_probabilities"] = 1/(1 + np.exp(-run["test_outputs"]))
    return run


def load_run_details(run, pos_fracs_df, neg_fracs_df, threshold_adjusted=False, use_cache=True):
    test_labels = get_test_labels(run["dataset"])
    cached_path = os.path.join(run["run_dir"], "run_stats.pkl")
    if threshold_adjusted:
        cached_path = os.path.join(run["run_dir"], "thresholded_run_stats.pkl")
    if use_cache and os.path.exists(cached_path):
        with open (cached_path, 'rb') as f:
            run = pkl.load(f)
            return run
    elif os.path.exists(run["run_dir"] + "/" + "test_outputs.txt"):
        thresholds = np.zeros(test_labels.shape[1])
        if threshold_adjusted:
            if not os.path.exists(run["run_dir"] + "/" + "valid_outputs.txt"):
                print(f"Validation set outputs missing for run {run_dir}")
                return run
            run["val_outputs"] = np.loadtxt(run["run_dir"] + "/" + "valid_outputs.txt")
            thresholds = get_thresholds(run["val_outputs"], get_val_labels(run["dataset"]))
        run["thresholds"] = thresholds
        run["test_outputs"] = np.loadtxt(run["run_dir"] + "/" + "test_outputs.txt")
        run["test_predictions"] = np.zeros(run["test_outputs"].shape)
        for i in range(test_labels.shape[1]):
            run["test_predictions"][:, i] = run["test_outputs"][:, i] > thresholds[i]

        run["test_probabilities"] = 1/(1 + np.exp(-run["test_outputs"]))
        compute_bas(run, test_labels, pos_fracs_df, neg_fracs_df)
        compute_errors(run, test_labels)
        compute_interdependence(run)
        train_labels = get_train_labels(run["dataset"])
        compute_rare_val_underpredict(run, test_labels.mean(axis=0), train_labels.mean(axis=0))
        compute_uncertainty_calibration(run, test_labels)
        with open (cached_path, 'wb') as f:
            pkl.dump(run, f)
        return run
    else:
        print(f"Test outputs missing for run {run_dir}")
        return run
