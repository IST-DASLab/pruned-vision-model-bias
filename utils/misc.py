import os
import math
import copy
import itertools
import collections
import torch
import torchvision
import numpy as np
import torch.nn as nn
from torch.utils.data import Subset, Dataset

from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix

from scipy.stats import entropy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_tensor_list(tensors):
    tensor = torch.cat([tensor.view(-1) for tensor in tensors], 0)
    return tensor


def get_model_trainable_params(model):
    model_trainable_params = torch.cat([param.view(-1) for param in model.parameters() if param.requires_grad])
    return model_trainable_params


def freeze_model_params(model, reset=False):
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            if hasattr(module, 'weight'):
                module.weight.requires_grad = False
            if hasattr(module, 'bias'):
                if not module.bias is None:
                    module.bias.requires_grad = False
        else:
            module.weight.requires_grad = True
            if not module.bias is None:
                module.bias.requires_grad = True
            if reset:
                module.reset_parameters()

def unfreeze_model_params(model):
    for param in model.parameters():
        param.requires_grad = True


def get_imagenet_pretrained_features(num_labels):
    model = torchvision.models.resnet50(pretrained=True)
    freeze_model_params(model)
    lin_out, lin_in = model.fc.weight.data.shape
    model.fc = nn.Linear(lin_in, num_labels)
    return model

def get_vggfaces_pretrained_features(model, num_labels):
    model.fc8 = nn.Linear(4096, num_labels)
    for param in model.parameters():
        param.requires_grad = False
    model.fc8.weight.requires_grad = True
    model.fc8.bias.requires_grad = True
    return model

# get indices only from a class
def get_class_indices(dataset, class_name, fraction_samples=1.0):
    indices = []
    for i in range(len(dataset)):
        if dataset[i][1] == class_name:
            indices.append(i)
    indices = np.array(indices)
    n_cls_samples = len(indices)
    if fraction_samples > 0.:
        indices = indices[:int(math.ceil(fraction_samples * n_cls_samples))]
    return indices


# get all indices apart from those in a subset
def get_without_class_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset)):
        if dataset[i][1] != class_name:
            indices.append(i)
    return indices


def get_remaining_indices(total_idxs, removed_idxs):
    remaining_idxs = list(set(total_idxs) - set(removed_idxs))
    return remaining_idxs


def add_outer_products_efficiently(matrix, vector, n_splits, scale=1.):
    piece_size = int(math.ceil(len(vector) / n_splits))
    len_vect = len(vector)
    for i in range(n_splits):
        for j in range(n_splits):
            x_start = i * piece_size
            x_end = min((i + 1) * piece_size, len_vect)
            y_start = j * piece_size
            y_end = min((j + 1) * piece_size, len_vect)

            matrix[x_start: x_end, y_start: y_end].add_(torch.ger(vector[x_start: x_end],
                                                                  vector[y_start: y_end]).div_(scale))
    return matrix


def get_grad_norm(model, l2_reg=0):
    model.eval()
    total_norm = 0.
    param_list = [p for p in model.parameters() if p.requires_grad]
    for p in param_list:
        param_grad = p.grad.data
        if l2_reg>0:
            param_grad += l2_reg * p.data
        param_norm = param_grad.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def get_norm_grad_dataset(model, data_loader, device, l2_reg=0, binary=False, n_labels=1):
    """
    Compute norm of the gradient for the whole dataset
    stochastic = True : average the norm of the gradient for all batches  \mathbb{E} [|| \nabla_{batch} f ||^2]
    stochastic = False: compute the norm of the full gradient (on the whole dataset) || \nabla f||^2
    :param model:
    :param data_loader:
    :param use_cuda:
    :return:
    """

    # when reduction='sum', accumulate the gradients over batches, to obtain the full gradient (don't use zero_grad())
    num_samples = len(data_loader.dataset)
    # criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    if binary:
        criterion = nn.BCEWithLogitsLoss(reduction="sum")
    else:
        criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    model.zero_grad()
    for idx, (input, label) in enumerate(data_loader):
        input, label = input.to(device), label.to(device)
        output = model(input)
        if binary:
            label = label.float()
            if n_labels==1:
                output = output.view(-1)
        loss = criterion(output, label) / num_samples
        loss.backward()
    grad_norm = get_grad_norm(model, l2_reg=l2_reg)
    return grad_norm


def get_grad_dataset(model, dataset, device, l2_reg, binary=False, n_labels=1, **kwargs):
    # when reduction='sum', accumulate the gradients over batches, to obtain the full gradient (don't use zero_grad())
    model.eval()
    model.to(device)
    model.zero_grad()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, **kwargs)
    num_samples = len(dataset)
    if binary:
        criterion = nn.BCEWithLogitsLoss(reduction="sum")
    else:
        criterion = nn.CrossEntropyLoss(reduction="sum")

    for idx, (input, label) in enumerate(data_loader):
        input, label = input.to(device), label.to(device)
        output = model(input)
        if binary:
            label = label.float()
            if n_labels==1:
                output = output.view(-1)
        loss = criterion(output, label)
        loss.backward()
    grad_dataset = torch.cat([param.grad.view(-1) for param in model.parameters() if param.requires_grad]).view(-1)
    grad_dataset = grad_dataset / num_samples
    if l2_reg>0:
        grad_dataset += l2_reg * torch.cat([param.view(-1) for param in model.parameters() if param.requires_grad]).view(-1)
    return grad_dataset


def compute_grads_multiple_samples(model, data, device, binary=False, n_labels=1):
    model.eval()
    grads_subset = collections.defaultdict(dict)
    # criterion = torch.nn.functional.cross_entropy
    if binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    params_list = list(model.parameters())
    for idx in range(len(data)):
        sample, target = data[idx]
        sample = sample[None].to(device)
        output = model(sample)
        if binary:
            target = target.float()
            if n_labels==1:
                output = output.view(-1)
        loss = criterion(output.view(1, -1), torch.tensor(target).view(-1).to(device))
        grads = torch.autograd.grad(loss, params_list)
        for i, (name, param) in enumerate(model.named_parameters()):
            grads_subset[name][idx] = grads[i].view(-1).cpu()
    torch.cuda.empty_cache()
    return grads_subset


def compute_loss_gradnorm_single_sample(model, sample, target, device):
    model.eval()
    criterion = torch.nn.functional.cross_entropy
    params_list = list(model.parameters())
    sample = sample[None].to(device)
    output = model(sample)
    loss = criterion(output.view(1, -1), torch.tensor(target).view(-1).to(device))
    params_grads_list = torch.autograd.grad(loss, params_list)
    grad_sample = flatten_tensor_list(params_grads_list).detach().cpu().numpy()
    grad_norm_sample = np.linalg.norm(grad_sample)
    pred = output.argmax(dim=1).item()
    if pred != target:
        misclf = True
    else:
        misclf = False
    return loss.item(), grad_norm_sample, misclf


def get_losses_grads_misclfidxs(model, data, device, model_folder):
    """
    :param model: the pretrained model
    :param data:  dataset
    :param device: gpu / cpu
    :return: losses_samples, grads_norms_samples, misclf_idxs
    """
    model.eval()
    grads_norms_samples = []
    losses_samples = []
    misclf_idxs = []
    num_samples = len(data)
    for idx in range(num_samples):
        trn_sample, trn_label = data[idx]
        # trn_sample = trn_sample[None].to(device)
        loss_sample, grad_norm_sample, misclf = compute_loss_gradnorm_single_sample(model, trn_sample, trn_label, device)
        grads_norms_samples.append(grad_norm_sample)
        losses_samples.append(loss_sample)
        if misclf:
            misclf_idxs.append(idx)
    grads_norms_samples = np.array(grads_norms_samples)
    losses_samples = np.array(losses_samples)
    misclf_idxs = np.array(misclf_idxs)
    filetosave = os.path.join(model_folder, 'samples_losses_grads_misclf.npz')
    np.savez(filetosave, losses=losses_samples, grads=grads_norms_samples, misclf=misclf_idxs)


def get_acc_precision_recall(outputs, targets):
    n_samples = targets.shape[0]
    n_labels = targets.shape[1]
    sig_preds = torch.nn.Sigmoid()(outputs)
    preds = (sig_preds > 0.5).float()
    auc_scores = {}
    prec_at_50 = {}
    prec_at_75 = {}
    prec_at_90 = {}
    prec_at_95 = {}
    per_label_correct = preds.eq(targets.view_as(preds)).sum(dim=0).cpu()
    per_label_target_true = (targets==1).float().sum(dim=0).cpu()
    per_label_target_false = (targets==0).float().sum(dim=0).cpu()
    per_label_preds_true = (preds == 1.).float().sum(dim=0).cpu()
    per_label_true_pos = ((targets==1).float() * (preds==1.).float()).sum(dim=0).cpu()
    per_label_false_pos = ((targets==0).float() * (preds==1.).float()).sum(dim=0).cpu()
    per_label_false_neg = ((targets==1).float() * (preds==0.).float()).sum(dim=0).cpu()
    per_label_acc = per_label_correct / n_samples
    per_label_recall = per_label_true_pos / per_label_target_true
    per_label_precision = per_label_true_pos / per_label_preds_true
    per_label_fpr = per_label_false_pos / per_label_target_false
    per_label_fnr = per_label_false_neg / per_label_target_true
    per_label_f1_score = 2 * per_label_precision * per_label_recall / (per_label_precision + per_label_recall)
    per_label_freqs = per_label_target_true.numpy() / n_samples
    per_label_pred_freqs = per_label_preds_true.numpy() / n_samples
    for label in range(n_labels):
        if (np.sum(targets[:, label].numpy())>0) & (np.sum(targets[:, label].numpy())<n_samples):
            auc_scores[label] = roc_auc_score(targets[:, label].numpy(), sig_preds[:, label].numpy())
            pr_scores = precision_recall_curve(targets[:, label].numpy(), sig_preds[:, label].numpy())
            precs = pr_scores[0]
            recalls = pr_scores[1]
            def prec_at(rec):
                idx = np.argmin(np.abs(recalls - rec))
                if np.abs(recalls[idx]-rec) < 0.05:
                    return precs[idx]
                return np.nan
            prec_at_50[label] = prec_at(0.50)
            prec_at_75[label] = prec_at(0.75)
            prec_at_90[label] = prec_at(0.90)
            prec_at_95[label] = prec_at(0.95)
        else:
            auc_scores[label] = np.nan
            prec_at_50[label] = np.nan
            prec_at_75[label] = np.nan
            prec_at_90[label] = np.nan
            prec_at_95[label] = np.nan
    return per_label_acc.numpy(), \
            per_label_recall.numpy(), per_label_precision.numpy(),\
            per_label_fpr.numpy(), per_label_fnr.numpy(), \
           per_label_f1_score.numpy(), auc_scores, \
           prec_at_50, prec_at_75, prec_at_90, prec_at_95, \
           per_label_freqs, per_label_pred_freqs


def celeba_classes():
    return ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

def awa_classes():
    return ['black', 'white', 'blue', 'brown', 'gray', 'orange', 'red',
'yellow', 'patches', 'spots', 'stripes', 'furry', 'hairless',
'toughskin', 'big', 'small', 'bulbous', 'lean', 'flippers',
'hands', 'hooves', 'pads', 'paws', 'longleg', 'longneck', 'tail',
'chewteeth', 'meatteeth', 'buckteeth', 'strainteeth', 'horns',
'claws', 'tusks', 'smelly', 'flys', 'hops', 'swims', 'tunnels',
'walks', 'fast', 'slow', 'strong', 'weak', 'muscle', 'bipedal',
'quadrapedal', 'active', 'inactive', 'nocturnal', 'hibernate',
'agility', 'fish', 'meat', 'plankton', 'vegetation', 'insects',
'forager', 'grazer', 'hunter', 'scavenger', 'skimmer', 'stalker',
'newworld', 'oldworld', 'arctic', 'coastal', 'desert', 'bush',
'plains', 'forest', 'fields', 'jungle', 'mountains', 'ocean',
'ground', 'water', 'tree', 'cave', 'fierce', 'timid', 'smart',
'group', 'solitary', 'nestspot', 'domestic']


def get_accs_confusion_matrix_entropies(model, loader, device, n_samples, n_classes):
    per_class_correct = torch.zeros(n_classes)
    total_targets = np.zeros(n_samples)
    total_preds = np.zeros(n_samples)
    samples_entropies = np.zeros(n_samples)
    labels_frequencies = np.zeros(n_classes)
    model.eval()
    start_idx = 0
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(device)
            batch_size = samples.shape[0]
            outputs = nn.Softmax(dim=1)(model(samples))
            preds = outputs.argmax(dim=1).detach().cpu()
            preds_ohe = torch.zeros((batch_size, n_classes))
            preds_ohe[np.arange(batch_size), preds] = 1
            targets_ohe = torch.zeros((batch_size, n_classes))
            targets_ohe[np.arange(batch_size), targets] = 1
            per_class_correct += preds_ohe.eq(targets_ohe).sum(dim=0)
            end_idx = start_idx + batch_size
            total_targets[start_idx : end_idx] = targets.numpy()
            total_preds[start_idx : end_idx] = preds.numpy()
            samples_entropies[start_idx : end_idx] = entropy(outputs.detach().cpu().numpy(), axis=1)
            labels_frequencies += np.sum(targets_ohe.numpy(), 0)
            start_idx = end_idx
    per_class_acc = per_class_correct.numpy() / n_samples
    labels_frequencies = labels_frequencies / n_samples
    conf_matrix = confusion_matrix(total_targets, total_preds, labels=np.arange(n_classes))
    return per_class_acc, conf_matrix, samples_entropies, labels_frequencies



def get_attribute_statistics(model, loader, device, attribute):
    total_pos_attribute = 0.
    total_pred_pos_attribute = 0.
    total_true_pos_attribute = 0.
    model.eval()
    with torch.no_grad():
        for samples, targets in loader:
            samples, targets = samples.to(device), targets.to(device)
            outputs = model(samples)
            preds = (torch.nn.Sigmoid()(outputs) > 0.5).float()
            preds_attr = preds[:, attribute]
            targets_attr = targets[:, attribute]
            total_pos_attribute += torch.sum(targets_attr).item()
            total_pred_pos_attribute += torch.sum(preds_attr).item()
            total_true_pos_attribute += ((preds_attr==1.).float() * (targets_attr==1.).float()).sum().item()
    return total_pos_attribute, total_pred_pos_attribute, total_true_pos_attribute



class CustomTensorDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[0][index], self.tensors[1][index].item()

    def __len__(self):
        return self.tensors[0].size(0)



def perturb_labels(samples, n_classes, seed, new_labels=None):
    n_samples = len(samples)
    torch.manual_seed(seed)
    perturbed_samples = None
    perturbed_labels = torch.zeros(n_samples).long()
    if new_labels is not None:
        perturbed_labels = new_labels
    for i in range(n_samples):
        if perturbed_samples is None:
            perturbed_samples = samples[i][0].unsqueeze(0)
        else:
            perturbed_samples = torch.cat((perturbed_samples, samples[i][0].unsqueeze(0)))
        if new_labels is None:
            perturbed_labels[i] = torch.randint(high=n_classes, size=(1,1)).item()
    perturbed_dataset = CustomTensorDataset(perturbed_samples, perturbed_labels)
    return perturbed_dataset
