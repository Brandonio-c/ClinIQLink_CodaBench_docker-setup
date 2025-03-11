# Sample Submission Guide

This directory provides an example of a correctly formatted submission for **ClinIQLink** evaluations. Participants should use this structure to prepare their submissions.

## Folder Structure

```
sample_submission/
├── submit.py                   # Main submission script (participants modify this)
├── submit_GPT-2_example.py      # Example implementation using GPT-2
├── submit.sh                    # SLURM submission script for HPC environments
├── README.md                    # This document
├── submission_template/          # Blank template for participants
│   ├── MC_template.prompt
│   ├── list_template.prompt
│   ├── multi_hop_template.prompt
│   ├── multi_hop_inverse_template.prompt
│   ├── short_template.prompt
│   ├── short_inverse_template.prompt
│   ├── tf_template.prompt
│   ├── README.md                 # Instructions for using the templates
```

## How to Use

### 1. Modify `submit.py`
Participants should modify `submit.py` to implement their model. The script should:

- Load and initialize the selected LLM.
- Process the provided QA datasets.
- Generate responses following the expected format.
- Output results as a JSON file.

#### Specifically, the following functions must be updated:

- **`load_participant_model(self)`**  
  - Implement loading of the chosen LLM model locally

- **`load_participant_pipeline(self)`**  
  - Initialize the LLM inference pipeline.

- **`YOUR_LLM_PLACEHOLDER(self, prompt)`** *(or rename for another model)*  
  - Update this function to call the loaded model and return generated text.

### 2. Run Locally
Before submitting, test the script locally by running:

```bash
python submit.py
```

### 3. Using GPT-2 (Example)
An example implementation using GPT-2 is provided in `submit_GPT-2_example.py`. This serves as a reference for setting up a model and generating responses.

### 4. Submit to SLURM HPC
If running on an HPC cluster, use the provided SLURM job submission script:

```bash
sbatch submit.sh
```

The `submit.sh` script should be modified to match the specific requirements of the computing environment.

## Submission Template

The `submission_template/` folder contains **prompt templates** for different QA types:

- **MC_template.prompt** – Multiple-choice question prompt
- **list_template.prompt** – List-based question prompt
- **multi_hop_template.prompt** – Multi-hop reasoning question prompt
- **multi_hop_inverse_template.prompt** – Inverse multi-hop question prompt
- **short_template.prompt** – Short-answer question prompt
- **short_inverse_template.prompt** – Inverse short-answer question prompt
- **tf_template.prompt** – True/False question prompt

Each prompt provides a **format example** for how inputs should be structured when interacting with an LLM.

## Submission Format

A valid submission should:

- Follow the structure of the **sample_submission** folder.
- Include all necessary dependencies inside `submit.py`.
- Match the expected **input-output format** of the provided datasets.
- Run without errors in the evaluation environment.

## Notes

- The `submission_template/` folder provides an outline for expected submissions.
- You cannot use an external API for an LLM. 
- More information is available on the ClinIQLink challenge page on using outcall requests to do things like retrieving information on the question for RAG etc. see: https://brandonio-c.github.io/ClinIQLink-2025/
- If running a local model, make sure all dependencies are installed and configured correctly.

For further details, refer to the **main repository README.md**.


