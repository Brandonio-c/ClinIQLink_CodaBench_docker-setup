# Sample Submission Guide

This directory provides an example of a correctly formatted submission for **ClinIQLink** evaluations. Participants should use this structure to prepare their submissions.

## Folder Structure

```
sample_submission/
├── submit.py                   # Main submission script (participants modify this)
├── evaluate.py                 # runs the evaluations 
├── entrypoint.sh               # entrypoint for the container
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
Participants CAN but ARE NOT REQUIRED TO modify `submit.py` to implement their model. The script should:

- Load and initialize the selected LLM.
- Process the provided QA datasets.
- Generate responses following the expected format.
- Output results as a JSON file.

#### Specifically, the following functions can be updated if required:

- **`load_participant_model(self)`**  
  - Implement loading of the chosen LLM model locally

- **`load_participant_pipeline(self)`**  
  - Initialize the LLM inference pipeline.

### 2. Run Locally
Before submitting, test the script locally by running:

```bash
python submit.py --mode local --chunk_size 4 --max_length 200 --num_tf 1 --num_mc 1 --num_list 1 --num_short 1 --num_short_inv 1 --num_multi 1 --num_multi_inv 1
```

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
- More information is available on the ClinIQLink challenge page on using outcall requests to do things like retrieving information on the question for RAG etc. see: [cliniqlink.org](https://cliniqlink.org)
- If running a local model, make sure all dependencies are installed and configured correctly.

For further details, refer to the **main repository README.md**.


