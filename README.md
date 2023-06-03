This repository contains the code to replicate the analysis  for the paper
 "Bias in Pruned Vision Models: In-Depth Analysis and Countermeasures", as well
as the example_viewer, which can be used to audit the CelebA dataset examples,
as well as false positives and negatives for models trained on this data.


Note: Due to technical issues there is a slight delay with releasing the models will be available by June 6.

This code pre-supposes that the CelebA dataset is downloaded to the server and that the models are in the
 `runs` directory in this folder.

## Replicating the experiments

Please run the code in `compute_run_stats.ipynb` to replicate our analysis for the joint, single-label, and backdoor runs.

## Using the example viewer

Please activate the example viewer by running the following command:

`flask --app example_viewer  run --host=0.0.0.0 --port [PORT]`. Sample URLs are provided in the UI.


