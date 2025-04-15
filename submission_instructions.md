
# ClinIQLink Model Submission: Full Build and Deployment Guide

This guide outlines the steps required to prepare, validate, package, and convert a model submission for the ClinIQLink evaluation environment. It is designed to be model-agnostic and portable to HPC platforms such as UMD's Zaratan cluster using Apptainer.

---

## Step 1: Prepare the Project Directory

Create a directory structure like the following:

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
├── README.md
```

> Make sure your model is Hugging Face-compatible, or you provide the required logic in `submit.py` to load and run inference.

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

### 1. Build the docker image (ensuring it is amd64)

Docker on macOS (especially on Apple Silicon) defaults to ARM64 builds, which are incompatible with the target HPC environment. You must explicitly build for `linux/amd64`.

```bash
docker buildx create --use   # One-time setup if not already done
docker buildx build --platform linux/amd64 -t cliniqlink_submission_image --load .
```

### 2. save it as a `.tar` file:
   ```bash
   docker save cliniqlink_submission_image:latest -o cliniqlink_submission.tar
   ```

---

## Step 4: Convert Docker Image to Apptainer SIF (Platform-Specific Options)

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

## Step 5: Submit the Container to UMD Zaratan

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