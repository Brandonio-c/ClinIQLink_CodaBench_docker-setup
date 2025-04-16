# ClinIQLink CodaBench Docker Setup

This repository provides the complete evaluation environment for participants in the **ClinIQLink 2025** challenge. 

## Challenge Information

- **Main Website**: https://brandonio-c.github.io/ClinIQLink-2025/
- **CodaBench Competition**: https://www.codabench.org/competitions/5117/
- **Official Docker Image**: [`brandonioc/cliniqlink-image`](https://hub.docker.com/r/brandonioc/cliniqlink-image)
- **Sample Dataset**: https://github.com/Brandonio-c/ClinIQLink_Sample-dataset

---

## Submission Instructions (short) 

1. Request access to UMD HPC via: https://docs.google.com/forms/d/e/1FAIpQLSerRZnVHm-Trk9eYp6ebrJHQKTPvSBrI6nBsKPguE8voigrWw/viewform?usp=sharing
2. Containerize your submission using the provided instructions below
3. Once you have recieved a globus endpoint to upload to:

   a. Upload your model’s codebase (excluding the model weights file / any large files) to the Globus endpoint within a folder named:
      - model_submission
   
   b. Upload a containerized version of your full model architecture to the same Globus endpoint within a folder named:
      - model_container_submission

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
   ├── submit.py                   # The evaluation script automatically run during submission
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
All submissions to UMD HPC must be containerized using Singularity apptainer. Instructions on how to containerize are below. 

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

## Example Output

After running `submit.py`, the final evaluation results will be saved to:

```
/tmp/overall_evaluation_results.json
```

You can mount this path in Docker or SLURM to retrieve the file, or just open the folder with: 

```bash
open /tmp/
```

---

## Submission Instructions (long)

This guide outlines the steps required to prepare, validate, package, and (if needed) convert a model submission for the ClinIQLink evaluation environment. It is designed to be model-agnostic and portable to HPC platforms such as UMD's Zaratan cluster using Apptainer.

- Instructions are also contained within submission_instructions.md file 

---

## Step 1: Clone and Prepare the Project Directory

Clone this Github page:
   ```bash
   Git clone https://github.com/Brandonio-c/ClinIQLink_CodaBench_docker-setup.git
   ```

With the following directory setup: 

```
ClinIQLink_CodaBench_docker-setup/
├── data/                        # Will be loaded externally via bind mount or environment variable
├── model_submissions/          # Place your model files or directory here
│   └── <your_model_directory>/
│       ├── config.json
│       ├── tokenizer.model
│       └── model weights and other artifacts
├── submission/
│   └── submit.py               # Main submission script (provided)
├── Dockerfile                  # Dockerfile to containerize the submission
├── Singularity.def             # Apptainer file to containerize the submission
├── README.md
```

Place your model within the model_submissions subfolder. 
Make sure your model is one of the following:
   - Hugging Face-compatible
   - PyTorch compatible
   - Python script-based models WITH the required logic in `submit.py` added to load and run inference on your model.

---

## Step 2: Test the Submission Script Locally (optional to ensure it works)

1. Create a Python virtual environment:
   ```bash
   python3 -m venv ClinIQLink_submission_test_env
   source ClinIQLink_submission_test_env/bin/activate
   pip install -r requirements.txt  # Ensure all dependencies are satisfied
   ```

2. Inside the `submission/` folder, run:
   ```bash
   python submit.py --mode local --max_length 200 --num_tf 1 --num_mc 1 --num_list 1 --num_short 1 --num_short_inv 1 --num_multi 1 --num_multi_inv 1
   ```

   This will execute the evaluation logic, using your model under `model_submissions/`.

---

## Step 3: Build a Docker Image for Linux x86_64 (Required)

There are three ways to build the containerized submission:

## Option 1 – Build and Run Directly with Apptainer

This method builds the container directly using the provided `Singularity.def` file.  
> **Note:** Ensure you are building on (or emulating) a Linux x86_64 environment.  
> • On Linux, you can run the command directly.  
> • On macOS, use a Lima VM to emulate linux/amd64.  
> • On Windows, use WSL2 with Ubuntu.

1. **Ensure the `Singularity.def` file is in your project root.**

2. **Build your container directly:**

   ```bash
   apptainer build cliniqlink_submission.sif Singularity.def
   ```

   This command produces a container image file named `cliniqlink_submission.sif` built for the host’s architecture. Ensure your host (or VM/WSL2) is configured as Linux x86_64.

3. **Run the container locally:**

   To execute the evaluation script inside the container, run:
   
   ```bash
   apptainer exec cliniqlink_submission.sif python /app/submit.py --mode container --max_length 1028 --num_tf 1 --num_mc 1 --num_list 1 --num_short 1 --num_short_inv 1 --num_multi 1 --num_multi_inv 1
   ```

   Alternatively, if you prefer using the default run command specified in the `%runscript` section, simply run:
   
   ```bash
   apptainer run cliniqlink_submission.sif
   ```

---

## Option 2 – Build the Apptainer SIF Directly from the Dockerfile

This method builds the Docker image directly using the Dockerfile, ensuring the image is for linux/amd64, then converts it directly to an Apptainer SIF without creating a tar archive.  
> **Note:**  
> • On Linux, run these commands directly.  
> • On macOS, ensure you run Docker with the `--platform linux/amd64` flag or use Docker Desktop configured for amd64 builds.  
> • On Windows, use Docker Desktop (with WSL2) and specify the platform.

1. **Build the Docker image locally using the Dockerfile:**  
   In your project directory (where your `Dockerfile` resides), run:

   ```bash
   docker build --platform linux/amd64 -t cliniqlink_submission_image:latest .
   ```

2. **Convert the Docker image from the local Docker daemon directly into an Apptainer SIF:**

   ```bash
   apptainer build cliniqlink_submission.sif docker-daemon://cliniqlink_submission_image:latest
   ```

   This command directly accesses the Docker image (built for linux/amd64) from your local Docker daemon and converts it into an Apptainer SIF file.

---

### Platform-Specific Notes

- **macOS Users:**  
  • For Option 1, if your macOS system is not Linux, use a Lima VM configured for x86_64. For example:
  
  ```bash
  brew install lima qemu
  limactl create --name=apptainer-x86_64 --arch=x86_64 template://apptainer
  limactl start apptainer-x86_64
  limactl shell apptainer-x86_64
  ```
  
  Then run the Option 1 commands from inside the VM.  
  
  • For Option 2, ensure Docker Desktop is set to build images with the linux/amd64 platform (via the `--platform linux/amd64` flag).

- **Windows Users (via WSL2):**  
  • Use Docker Desktop with WSL2 integration and run the commands from a WSL2 shell (e.g., Ubuntu).  
  • For Option 1, you can install Apptainer directly in Ubuntu on WSL2.  
  • For Option 2, ensure Docker builds with the `--platform linux/amd64` flag as shown above.

---


## Option 3 – Build from Docker and Convert to Apptainer SIF (Using a Tar Archive)


### I. Build the docker image (ensuring it is amd64)

Docker on macOS (especially on Apple Silicon) defaults to ARM64 builds, which are incompatible with the target HPC environment. You must explicitly build for `linux/amd64`.

```bash
docker buildx create --use   # One-time setup if not already done
docker buildx build --platform linux/amd64 -t cliniqlink_submission_image --load .
```

### II. save it as a `.tar` file:
   ```bash
   docker save cliniqlink_submission_image:latest -o cliniqlink_submission.tar
   ```

---

## III: Convert Docker Image to Apptainer SIF (Platform-Specific Options)

After you've built your Docker image in **Step 3**, you'll need to convert it into an Apptainer `.sif` file to run it on the ClinIQLink evaluation infrastructure (e.g., UMD's Zaratan HPC).

Choose your platform below to proceed:

---

### **Option A – On macOS (via Lima VM + Apptainer)**

macOS cannot run Apptainer natively. You must use a Linux VM with Apptainer preinstalled.

1. **Install Lima (one-time):**

```bash
brew install lima
brew install qemu
limactl create --name=apptainer-x86_64 --arch=x86_64 template://apptainer
```

2. **Open a shell inside the Lima VM:**

```bash
limactl start apptainer-x86_64
limactl shell apptainer-x86_64
```

And now, inside the vm check: 
```bash
uname -m  # should return x86_64
```

2a. **Or, to start an already made instance, use:**

```bash
limactl start apptainer-x86_64
limactl shell apptainer-x86_64
```

2b. **Or, to restart or delete the VM instance, use:**

```bash
limactl stop apptainer-x86_64
limactl delete apptainer-x86_64
```

3. **Inside the VM: create a build directory**

```bash
mkdir -p ~/apptainer_build
```

4. **Within the VM, copy the Docker `.tar` into the VM**

```bash
cp /path/to/cliniqlink_submission.tar ~/apptainer_build/
```

4a. **Or, on macOS, copy the Docker `.tar` into the VM**

```bash
cp /path/to/cliniqlink_submission.tar ~/.lima/apptainer-x86_64/rootfs/home/YOUR_VM_USERNAME/apptainer_build/
```

5. **Inside the VM, build the `.sif`**

```bash
cd ~/apptainer_build
apptainer build cliniqlink_submission.sif docker-archive://cliniqlink_submission.tar
```

6. **Copy it back for upload**

```bash
cp ~/apptainer_build/cliniqlink_submission.sif /tmp/lima/
```

Then back on the macOS host:

```bash
cp ~/.lima/apptainer-x86_64/rootfs/tmp/lima/cliniqlink_submission.sif /Users/yourusername/Downloads/
```
You can also open the folder with 

```bash
open /tmp/lima
```

And shut down the VM

```bash
limactl stop apptainer-x86_64
```

---

### **Option B – On Linux (Native Apptainer)**

1. **Install Apptainer (if not already)**

```bash
sudo apt update && sudo apt install -y apptainer
```

Or follow the latest instructions: https://apptainer.org/docs/user/main/installation.html

2. **Build directly from Docker (if Docker daemon is accessible)**

```bash
apptainer build cliniqlink_submission.sif docker-daemon://cliniqlink_submission_image:latest
```

3. **Or build from the `.tar` (safer and portable)**

```bash
apptainer build cliniqlink_submission.sif docker-archive://cliniqlink_submission.tar
```

---

### **Option C – On Windows (via WSL2 + Ubuntu + Apptainer)**

1. **Enable WSL2 and install Ubuntu (Windows 10/11)**  
   Follow Microsoft’s official WSL guide: https://docs.microsoft.com/en-us/windows/wsl/install

2. **Inside Ubuntu (via WSL), install Apptainer**

```bash
sudo apt update
sudo apt install -y apptainer
```

Or follow: https://apptainer.org/docs/user/main/installation.html

3. **Move your Docker `.tar` to the WSL home directory**

```bash
cp /mnt/c/Users/YOUR_NAME/Downloads/cliniqlink_submission.tar ~/
```

4. **Build the `.sif`**

```bash
apptainer build cliniqlink_submission.sif docker-archive://cliniqlink_submission.tar
```

5. **Access the `.sif` from Windows**

Your built file will be located under:

```
\\wsl$\Ubuntu\home\yourusername\cliniqlink_submission.sif
```

---

## Step 4: Submit the Container to UMD Zaratan

### 1. Request Submission Access

You must first be approved to upload your container. Complete the request form at:

[https://docs.google.com/forms/d/e/1FAIpQLSerRZnVHm-Trk9eYp6ebrJHQKTPvSBrI6nBsKPguE8voigrWw/viewform?usp=sharing](https://docs.google.com/forms/d/e/1FAIpQLSerRZnVHm-Trk9eYp6ebrJHQKTPvSBrI6nBsKPguE8voigrWw/viewform?usp=sharing)

After approval, you will receive a Globus upload link.

### 2. Set Up a Globus Account

Follow the instructions here:

- Globus Connect Personal: [https://www.globus.org/globus-connect-personal](https://www.globus.org/globus-connect-personal)  
- UMD HPCC Globus instructions: [https://hpcc.umd.edu/hpcc/help/globus.html#gcp](https://hpcc.umd.edu/hpcc/help/globus.html#gcp)

### 3. Upload Your `.sif` File

Once approved, you’ll be emailed a **Globus link** to upload your containerized Apptainer `.sif` file as well as your models code. Upload the files/model through the Globus web interface or the Globus Connect Personal app.

---

## Additional References

- [Model Submission Guide](./model_submission/README.md)
- [Model Submission Instructions](./submission_instructions.md)
- [Evaluation Script Guide](./submission/README.md)
- [Prompt Template Guide](./submission/submission_template/README.md)

---

For any issues, please refer to the official challenge documentation or email: brandon.colelough@nih.gov.
