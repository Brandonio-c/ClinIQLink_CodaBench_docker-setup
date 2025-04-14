# Model Submission Directory

This directory is intended for participants to place their local language model files for evaluation within the ClinIQLink framework.

## Directory Structure

The `model_submission/` folder must contain a subdirectory with your model files. A typical structure is as follows:

```
model_submission/
├── your_model_directory/
│   ├── config.json
│   ├── tokenizer.model or tokenizer.json
│   ├── model weights (e.g., .bin, .safetensors, or .pth files)
│   └── other required files to load the model
```

## Requirements

- The model must be stored locally. Remote or API-based models are not permitted.
- The directory must include all files required to load the model using either:
  - Hugging Face Transformers (`AutoModelForCausalLM` and `AutoTokenizer`), or
  - A PyTorch checkpoint (`.pt` or `.pth` file), or
  - A Python script that defines a `model` object (callable or with a `.generate()` method).
- The `submit.py` script will automatically search for and load your model from this directory.
- All dependencies must be compatible with the evaluation environment and pre-installed in your setup.

## Notes

- Avoid using absolute paths in your scripts. All model paths should be relative to `model_submission/`.
- Do not modify the folder name `model_submission/`, as the evaluation environment depends on this name.
- Ensure your model directory contains all necessary files to initialize both the model and tokenizer without external downloads.

## Reference

For additional instructions and examples, see the main ClinIQLink submission documentation at:

[https://brandonio-c.github.io/ClinIQLink-2025/](https://brandonio-c.github.io/ClinIQLink-2025/)
