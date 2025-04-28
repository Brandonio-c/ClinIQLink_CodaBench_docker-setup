# ClinIQLink CodaBench Docker Setup

This repository provides the complete evaluation environment for participants in the **ClinIQLink 2025** challenge. 

## Challenge Information

- **Main Website**: https://cliniqlink.org/
- **CodaBench Competition**: https://www.codabench.org/competitions/5117/
- **Official Docker Image**: [`brandonioc/cliniqlink-image`](https://hub.docker.com/r/brandonioc/cliniqlink-image)
- **Sample Dataset**: https://github.com/Brandonio-c/ClinIQLink_Sample-dataset

---

## Submission Instructions 

### 1. (optional) Build the apptainer container

<span style="color:red"><b><u>This is only required if you have made changes to the submit.py script to run your model (e.g. if you needed to add a new participant model load definition, if you have added offline RAG etc.)</u></b></span>

Build your customized submission container with the instructions provided by the build_instructions.md file

- See: [`build_instructions.md`](.build_instructions.md)


### 2. Submit your model 

Submit your model to UMD HPC Zaratan for NIH to run your model on our Benchmark! 

#### A. Request access

You must first be approved to upload your model. Complete the request form at:

[https://docs.google.com/forms/d/e/1FAIpQLSerRZnVHm-Trk9eYp6ebrJHQKTPvSBrI6nBsKPguE8voigrWw/viewform?usp=sharing](https://docs.google.com/forms/d/e/1FAIpQLSerRZnVHm-Trk9eYp6ebrJHQKTPvSBrI6nBsKPguE8voigrWw/viewform?usp=sharing)

After approval, you will receive a Globus upload link.

#### B. Set Up a Globus Account

Follow the instructions here:

- Globus Connect Personal: [https://www.globus.org/globus-connect-personal](https://www.globus.org/globus-connect-personal)  
- UMD HPCC Globus instructions: [https://hpcc.umd.edu/hpcc/help/globus.html#gcp](https://hpcc.umd.edu/hpcc/help/globus.html#gcp)

### C. Upload Your model Files

Once approved, you’ll be emailed a **Globus link** to upload your model files. Upload the files/model through the Globus web interface or the Globus Connect Personal app.

---

## Folder Structure

```
ClinIQLink_CodaBench_docker-setup/
│
├── Dockerfile                      # Defines the evaluation environment used by CodaBench 
├── Singularity.def                 # Apptainer file to containerize the submission for UMD HPC
├── README.md                       # This file
├── LICENSE
│
├── data/                           # Evaluation datasets (mounted at runtime)
│   ├── MC.json
│   ├── TF.json
│   ├── list.json
│   ├── short.json
│   ├── short_inverse.json
│   ├── multi_hop.json
│   ├── multi_hop_inverse.json
│
├── model_submission/                                 # Participant-provided model directory
│   └── YOUR_MODEL_SUBMISSION_FODLER/                 # Example HuggingFace-compatible model
│       ├── e.g. config.json
│       ├── e.g. tokenizer.json
│       ├── e.g. model.safetensors
│       └── e.g. other required files
│
├── submission/                    # Main evaluation logic
   ├── submit.py                   # The submission script automatically run during submission
   ├── evaluate.py                 # The evaluation script automatically run during submission
   ├── entrypoint.sh               # container entrypoint
   ├── requirements.txt            # Python dependencies for submit.py
   ├── README.md                   # Instructions for participants
   └──submission_template/         # Prompt templates used by the evaluation script
         ├── MC_template.prompt
         ├── tf_template.prompt
         ├── list_template.prompt
         ├── short_template.prompt
         ├── short_inverse_template.prompt
         ├── multi_hop_template.prompt
         ├── multi_hop_inverse_template.prompt
         └── README.md
```

---

## Overview

The submit.py script within the submission subfolder is designed to evaluate your model on the ClinIQLink dataset (closed source). The submit.py script is the unified evaluation script which supports:

- Multiple QA types: TF, MC, list, short, short_inverse, multi-hop, multi-hop_inverse (all 7 modalities)
- Prompt injection via templates
- Scoring via F1, semantic similarity, and traditional metrics (BLEU, ROUGE, METEOR)
- HuggingFace, PyTorch, or Python script-based models

You can test your model using the Docker setup or submit directly to CodaBench using the same structure.
All submissions to UMD HPC must be containerized using Singularity apptainer. Instructions on how to are shown within the build_instructions.md file

---

## Sample QA Dataset for Local Testing

You can download a sample dataset of QA pairs for local testing from the following repository:

**Sample QA Dataset Repository**:  
[https://github.com/Brandonio-c/ClinIQLink_Sample-dataset](https://github.com/Brandonio-c/ClinIQLink_Sample-dataset)

The `.json` files (e.g., `MC.json`, `TF.json`, etc.) are pre-loaded directly into the `data/` folder for local evaluation by participants.

---

## Prompt Templates

Prompt templates for each QA type are preloaded to the folder  `submission_template/` and are loaded automatically by `submit.py`. These templates include placeholders (e.g., `{question}`, `{options}`) that are filled dynamically.

- These must **not be modified** by participants.
- See: [`submission/submission_template/README.md`](./submission/submission_template/README.md)

---

## Evaluation Datasets

- The datasets used for evaluation are JSON files located in the `data/` folder.
- Participants do not need to edit or submit this folder; it will be mounted automatically during evaluation on Codabench or UMD HPC. 
- The dataset is closed source and will not be released. This dataset is for testing purposes only and can only be utilised through the NIH. 
- See: [`data/README.md`](./data/README.md)

---


## Additional References

- [Model Submission Guide](./model_submission/README.md)
- [Model Submission Instructions](./submission_instructions.md)
- [Evaluation Script Guide](./submission/README.md)
- [Prompt Template Guide](./submission/submission_template/README.md)

---

For any issues, please refer to the official challenge documentation or email: brandon.colelough@nih.gov.
