# Data Directory

This folder is reserved for the ClinIQLink QA datasets used during evaluation.
The json files within this folder are from the sample data provided by the ClinIQLink team available at github here: 
https://github.com/Brandonio-c/ClinIQLink_Sample-dataset

## Overview

The evaluation script (`submit.py`) expects datasets to be available locally within this `data/` folder. However, participants do **not** need to manually populate this directory.

## How Data Is Loaded

- **Codabench Submissions:**  
  When submitting through Codabench, the dataset will be automatically mounted into the container environment and made available to your submission under the environment variable:
  ```
  CODABENCH_DATASET_DIR
  ```
  The script will look for datasets in this path during runtime.

- **HPC Submissions:**  
  For SLURM-based submissions on an HPC cluster, the dataset will be made available in the appropriate mounted path (e.g., `/data/...`) and similarly accessed via `CODABENCH_DATASET_DIR`.

## Do Not Modify

- Do not manually edit or remove this folder.

## Reference

Dataset filenames expected by `submit.py` (within `CODABENCH_DATASET_DIR`) include:

- `MC.json`
- `TF.json`
- `list.json`
- `short.json`
- `short_inverse.json`
- `multi_hop.json`
- `multi_hop_inverse.json`

These files are automatically loaded and sampled during the evaluation process.

For further guidance, please refer to the official documentation:

[https://brandonio-c.github.io/ClinIQLink-2025/](https://brandonio-c.github.io/ClinIQLink-2025/)