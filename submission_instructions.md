
# ClinIQLink Model Submission

## 1. (optional) Build the apptainer container

<span style="color:red"><b><u>This is only required if you have made changes to the submit.py script to run your model (e.g. if you needed to add a new participant model load definition, if you have added offline RAG etc.)</u></b></span>

Build your customized submission container with the instructions provided by the build_instructions.md file

- See: [`build_instructions.md`](.build_instructions.md)


## 2. Submit your model 

Submit your model to UMD HPC Zaratan for NIH to run your model on our Benchmark! 

### A. Request access

You must first be approved to upload your model. Complete the request form at:

[https://docs.google.com/forms/d/e/1FAIpQLSerRZnVHm-Trk9eYp6ebrJHQKTPvSBrI6nBsKPguE8voigrWw/viewform?usp=sharing](https://docs.google.com/forms/d/e/1FAIpQLSerRZnVHm-Trk9eYp6ebrJHQKTPvSBrI6nBsKPguE8voigrWw/viewform?usp=sharing)

After approval, you will receive a Globus upload link.

### B. Set Up a Globus Account

Follow the instructions here:

- Globus Connect Personal: [https://www.globus.org/globus-connect-personal](https://www.globus.org/globus-connect-personal)  
- UMD HPCC Globus instructions: [https://hpcc.umd.edu/hpcc/help/globus.html#gcp](https://hpcc.umd.edu/hpcc/help/globus.html#gcp)

### C. Upload Your model Files

Once approved, you’ll be emailed a **Globus link** to upload your model files. Upload the files/model through the Globus web interface or the Globus Connect Personal app.

---