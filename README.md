This repository contains the code to replicate the analysis  for the paper
 "Bias in Pruned Vision Models: In-Depth Analysis and Countermeasures", as well
as the example_viewer, which can be used to audit the CelebA dataset examples,
as well as false positives and negatives for models trained on this data.


Note: Due to technical issues there is a slight delay with releasing the models will be available by June 6.

This code pre-supposes that the CelebA dataset is downloaded to the server and that the models are in the
 `runs` directory in this folder.

## Replicating the analysis

Please run the code in `compute_run_stats.ipynb` to replicate our analysis for the joint, single-label, and backdoor runs.

Note that in order to replicate the analysis, the experimental results must be downloaded.
They are avalilable [here](https://seafile.ist.ac.at/d/063b8702962c4737adfa/), and the notebook
assumes that they are in a folder called `runs/` at the root level of the code.

Likewise, the CelebA dataset must be downloaded, for example from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
In the code it is assumed to be located in `/home/Datasets/CelebA`.

## Using the example viewer

Please activate the example viewer by running the command below. The individual run
results and the CelebA dataset must be downloaded as described in the section above.

`flask --app example_viewer  run --host=0.0.0.0 --port [PORT]`. Sample URLs are provided in the UI.


