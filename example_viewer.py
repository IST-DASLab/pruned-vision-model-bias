from flask import *
import numpy as np
import pandas as pd
import bias_utils.runs as runs_joint
import bias_utils.single_label as runs_sl
from bias_utils.projects import PROJECTS
import io
import base64

app = Flask(__name__)

import time
from io import BytesIO
import zipfile
import os


@app.route("/")
def hello():
    return render_template("layout.html")


# Serve a single image from the dataset folder
# Example: http://localhost:8881/image/celeba/047665.jpg
@app.route('/image/<path:dataset>/<path:index>')
def send_image(dataset, index):
    filename=index
    if dataset == "celeba":
        return send_from_directory("/home/Datasets/celeba/img_align_celeba", filename)
    elif dataset == "full_celeba":
        return send_from_directory("/home/Datasets/celeba/img_celeba", filename)
    else:
        raise ValueError(f"don't know how to show images for f{dataset} ")


@app.route('/run_names')
def show_run_names():
    dataset = request.args.get("dataset", default="celeba", type=str)
    arch = request.args.get("arch", default="resnet18", type=str)
    strategy = request.args.get("strategy", default="GMP-RI", type=str)
    short_arch = arch
    if arch == "resnet18":
        short_arch = "rn18"
    project_name = f"{dataset}-all-{short_arch}"
    if project_name not in PROJECTS:
        raise ValueError(f"Project {project_name} doesn't exist.")

    runs = runs_joint.get_runs_for_project(project_name, strategy)
    runs_df = pd.DataFrame(runs)
    return render_template("generic_table.html", title=f"{project_name} Runs", table=runs_df.to_html(table_id="run_names"))

@app.route('/single_run_names')
def show_single_run_names():
    dataset = request.args.get("dataset", default="celeba", type=str)
    arch = request.args.get("arch", default="resnet18", type=str)
    strategy = 'GML-RI'
    backdoor = request.args.get("backdoor", default=False, type=bool)
    short_arch = arch
    if arch == "resnet18":
        short_arch = "rn18"
    if backdoor:
        project_name = f"{dataset}-backdoor-single-{short_arch}"
        runs = runs_sl.get_runs_for_project_backdoor(project_name)
    else:
        project_name = f"{dataset}-single-{short_arch}"
        runs = runs_sl.get_runs_for_project_single(project_name)
    if project_name not in PROJECTS:
        raise ValueError(f"Project {project_name} doesn't exist.")
    
    runs_df = pd.concat([pd.DataFrame(v) for v in runs.values()])
    return render_template("generic_table.html", title=f"{project_name} Runs", table=runs_df.to_html(table_id="run_names"))


@app.route('/classes')
def show_classes():
    dataset = request.args.get("dataset", default="celeba", type=str)
    if 'celeba' in dataset:
        classes = runs_joint.celeba_classes()
    elif 'awa' in dataset:
        classes = runs_joint.awa_classes()
    else:
        classes = []
    numbered_classes = pd.DataFrame([[i,k] for i, k in enumerate(classes)], columns=["number", "class"])
    return render_template("generic_table.html", title=f"{dataset} Classes", table=numbered_classes.to_html(table_id="classes"))



@app.route("/examples")
def show_examples():
    dataset = 'celeba'
    attr = request.args.get("attr", default="Blond_Hair", type=str)
    category = request.args.get("category", default=None, type=str)
    category_value = request.args.get("category_value", default=1, type=int)
    arch = request.args.get("arch", default="resnet18", type=str)
    run_type = request.args.get("run_type", default='joint', type=str)
    sparsity_strategy = request.args.get("sparsity_strategy", default=None, type=str)
    num_images = request.args.get("num_images", default=5, type=int)
    if num_images <= 0:
        num_images = 5
    attr_id = runs_joint.celeba_classes().index(attr)
    all_labels = runs_joint.get_test_labels('celeba')
    labels = all_labels[:, attr_id]
    category_labels = np.ones_like(labels)
    if category and category_value:
        category_labels = all_labels[:, runs_joint.celeba_classes().index(category)]
    if category and not category_value:
        category_labels = 1 - all_labels[:, runs_joint.celeba_classes().index(category)]

    def pick_n(X, n):
        if sum(X) <= n:
            return np.nonzero(X)[0]
        return np.random.choice(np.nonzero(X)[0], n, replace=False)

    image_ids = runs_joint.get_test_image_ids(dataset)

    if not sparsity_strategy:  # If no run requested
        splits = {
            "Positives": labels,
            "Negatives": 1-labels
        }
        title = f"Positive and Negative Examples for Attribute {attr}"

    else:
        short_arch = arch
        if arch == "resnet18":
            short_arch = "rn18"
        project_name = f"{dataset}-all-{short_arch}"
        if project_name not in PROJECTS:
            raise ValueError(f"Project {project_name} doesn't exist.")
        found = False
        if run_type == 'joint':
            strategy = sparsity_strategy.split('_')[1]
            found_runs = runs_joint.get_runs_for_project(project_name, strategy)
            matching_runs = [r for r in found_runs if sparsity_strategy == r['type']]
            run = None
            for mr in matching_runs:
                mr = runs_joint.load_test_outputs(mr)
                if 'test_probabilities' in mr:
                    found = True
                    run = mr
                    break
        elif run_type == 'single':
            found_runs = runs_sl.get_runs_for_project_single(project_name, 'GMP-RI')
            matching_runs = [r for r in found_runs[attr] if sparsity_strategy == r['type']]
            run = None
            for mr in matching_runs:
                mr = runs_joint.load_test_outputs(mr)
                if 'test_probabilities' in mr:
                    found = True
                    run = mr
                    break

        if not found:
            raise ValueError("No matching run was found")

        if run_type == 'joint':
            test_preds = run["test_predictions"][:,attr_id]
            test_pred_probs = run["test_probabilities"][:,attr_id]
        else:
            test_preds = run["test_predictions"]
            test_pred_probs = run["test_probabilities"]

        pred_categories = {
            "True Positives": category_labels*test_preds*labels > 0,
            "False Negatives": category_labels*(1-test_preds)*labels > 0,
            "False Positives": category_labels*test_preds*(1-labels) > 0,
            "True Negatives": category_labels*(1-test_preds)*(1-labels) > 0,
        }

        conf_categories = {
            "High Confidence": np.abs(test_pred_probs - 0.5) >= 0.4,
            "Low Confidence": np.abs(test_pred_probs - 0.5) < 0.4
        }

        splits = {f"{pred_k} - {conf_k}": pred_v*conf_v for pred_k, pred_v in pred_categories.items() for conf_k, conf_v in conf_categories.items() }
        cat_name = category
        if not category_value:
            cat_name = "Not " + category
        title = f"Classification Examples, {attr}/{cat_name}"

    def bool_to_color(b):
        if b == False:
            return "red"
        return "green"

    def index_to_display_tuple(i):
        if sparsity_strategy:
            return {"idx": image_ids[i],
                    "label": f'{image_ids[i]}: {labels[i]} / pred. {test_preds[i]}  {round(test_pred_probs[i], 2)}',
                    "color": bool_to_color(labels[i] == test_preds[i]) 
                    }
        return {"idx": image_ids[i],
                "label": f'{image_ids[i]}: {labels[i]}',
                "color": "white" 
                }
    images = {k: [index_to_display_tuple(i) for i in pick_n(v, num_images)] for k, v in splits.items()}
    return render_template('img.html', dataset=dataset, images = images, page_title = title)



