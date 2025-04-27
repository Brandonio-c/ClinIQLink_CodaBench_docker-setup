import json
import os
import random
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
from transformers.pipelines import TextGenerationPipeline
import re

# Explicitly set HuggingFace & Torch cache paths for consistency and safety inside container
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/transformers"
os.environ["TORCH_HOME"] = "/app/.cache/torch"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

class ClinIQLinkSampleDatasetSubmit:
    def __init__(self, run_mode="container", max_length=1028, sample_sizes=None, random_sample=False, chunk_size=2,
                    do_sample=False, temperature=None, top_p=None, top_k=None):
        self.run_mode = run_mode.lower()
        self.max_length = max_length
        self.sample_sizes = sample_sizes or {}
        self.random_sample = random_sample
        self.chunk_size = chunk_size
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if self.do_sample and (self.temperature is None or self.temperature <= 0):
            print("[WARN] --do_sample was set but temperature <=0; "
                "setting temperature=0.7 for safety.", flush=True)
            self.temperature = 0.7
            
        # Base directories and setup depending on run mode
        if run_mode == "container":
            print("Running in container mode.", flush=True)
            self.base_dir = "/app"
        else:
            print("Running in local mode.", flush=True)
            self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # Dataset and template directory setup
        self.dataset_dir = os.getenv("DATA_DIR", os.path.join(self.base_dir, "../data"))
        self.template_dir = os.path.join(self.base_dir, "submission_template")

        # Placeholder: load the participant's LLM model and inference pipeline.
        self.model = self.load_participant_model()
        self.pipeline = self.load_participant_pipeline()
        # Load and sample the dataset
        self.sampled_qa_pairs = self.load_and_sample_dataset()
        self.SYSTEM_MSG = (
            "You are a highly knowledgeable medical expert. "
            "Reply **only** with the requested answer format. "
            "Do not repeat the question or add explanations."
        )
        self.CAP_LETTERS_RE = re.compile(r"\b[A-Z]\b")

    def _strip_noise(self, text: str) -> str:
        """Remove leading blank lines and stray 'assistant' artefacts."""
        return re.sub(r"\bassistant\b", "", text, flags=re.I).strip()
    
    
    def _batched_inference(self, prompts, qa_type):
        """Run self.participant_model() on small chunks to avoid GPU OOM."""
        responses = []
        for i in range(0, len(prompts), self.chunk_size):
            chunk = prompts[i : i + self.chunk_size]
            out = self.participant_model(chunk if len(chunk) > 1 else chunk[0],
                                        qa_type=qa_type)
            # participant_model returns str for single prompt, list for many
            if isinstance(out, list):
                responses.extend(out)
            else:
                responses.append(out)
        return responses
    
    def _bundle(self, inputs, responses, prompts=None):
        """
        Return a structure like:
            {
            "inputs":    [... QA dicts each augmented with 'response' (and 'prompt')],
            "responses": [... model outputs ...],
            "prompts":   [... optional prompts ...]
            }
        So the evaluator can access clean splits, but the QA dicts still carry the outputs.
        """
        bundled_inputs = []
        for i, qa in enumerate(inputs):
            item = qa.copy()                  # Copy QA fields
            item["response"] = responses[i]   # Insert model output into QA dict
            if prompts:
                item["prompt"] = prompts[i]    # Insert prompt if given
            bundled_inputs.append(item)

        result = {
            "inputs": bundled_inputs,
            "responses": responses
        }
        if prompts:
            result["prompts"] = prompts

        return result

    def load_participant_model(self):
        """
        Dynamically loads the participant's LLM model from the 'model_submissions' directory.
        Supports multiple submission types: pre-trained Hugging Face models, raw weights, or model scripts.
        """
        print("Searching for participant's LLM model in 'model_submissions'...", flush=True)
        
        if self.run_mode == "local":
            model_submissions_dir = os.path.join(self.base_dir, "../model_submission")
        else:
            model_dir_env = os.getenv("USE_INTERNAL_MODEL", "1").strip().lower()
            if model_dir_env in ["1", "true", "yes"]:
                print(self.base_dir)
                print(os.path.join(self.base_dir, "model_submission"))
                model_submissions_dir = os.path.join(self.base_dir, "model_submission/snapshots")
            else:
                model_submissions_dir = os.path.join(self.base_dir, "model_submission/snapshots")
        
        if not os.path.exists(model_submissions_dir):
            print(f"Error: 'model_submissions' folder not found at {model_submissions_dir}", flush=True)
            return None

        # Search for potential models in the 'model_submissions' folder
        for entry in os.listdir(model_submissions_dir):
            entry_path = os.path.join(model_submissions_dir, entry)

            # Case 1: Hugging Face Pretrained Model Directory
            if os.path.isdir(entry_path) and "config.json" in os.listdir(entry_path):
                print(f"Loading Hugging Face model from: {entry_path}", flush=True)
                try:
                    # Dynamically select torch_dtype for compatibility
                    if torch.cuda.is_available():
                        torch_dtype = torch.bfloat16
                        device_map = "auto"
                    elif torch.backends.mps.is_available():
                        torch_dtype = torch.float32  # bfloat16 not supported on MPS
                        device_map = {"": torch.device("mps")}
                    else:
                        torch_dtype = torch.float32
                        device_map = "auto"  # fallback to CPU

                    model = AutoModelForCausalLM.from_pretrained(
                        entry_path,
                        trust_remote_code=True,
                        use_safetensors=True,
                        device_map=device_map,
                        torch_dtype=torch_dtype
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(entry_path, trust_remote_code=True, padding=True, padding_side="left")
                    print("Participant's Hugging Face model loaded successfully.", flush=True)

                    if self.tokenizer.chat_template is None:
                        print("Setting manual chat template for LLaMA tokenizer...", flush=True)
                        self.tokenizer.chat_template = (
                            "{% for message in messages %}"
                            "{% if message['role'] == 'system' %}"
                            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                            "{{ message['content'] }}<|eot_id|>"
                            "{% elif message['role'] == 'user' %}"
                            "<|start_header_id|>user<|end_header_id|>\n"
                            "{{ message['content'] }}<|eot_id|>"
                            "{% elif message['role'] == 'assistant' %}"
                            "<|start_header_id|>assistant<|end_header_id|>\n"
                            "{{ message['content'] }}<|eot_id|>"
                            "{% endif %}"
                            "{% endfor %}"
                            "<|start_header_id|>assistant<|end_header_id|>\n"
                        )

                    for module, device in model.hf_device_map.items():
                        print(f"Module '{module or 'root'}' loaded on device: {device}")


                    num_gpus = torch.cuda.device_count()
                    for i in range(num_gpus):
                        print(f"GPU {i} - {torch.cuda.get_device_name(i)}")
                        allocated = torch.cuda.memory_allocated(i) / 1024**3  # Convert bytes to GB
                        reserved = torch.cuda.memory_reserved(i) / 1024**3    # Convert bytes to GB
                        print(f"  Allocated: {allocated:.2f} GB")
                        print(f"  Reserved:  {reserved:.2f} GB")

                    return model
                except Exception as e:
                    print(f"Failed to load Hugging Face model: {e}", flush=True)

            # Case 2: Model Checkpoint (PyTorch)
            elif entry.endswith(".pt") or entry.endswith(".pth"):
                print(f"Loading PyTorch model checkpoint: {entry_path}", flush=True)
                try:
                    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = torch.load(entry_path, map_location=map_location)
                    print("Participant's PyTorch model loaded successfully.", flush=True)
                    # Set fallback tokenizer
                    if not hasattr(self, "tokenizer") or self.tokenizer is None:
                        fallback_tokenizer_path = os.path.join(os.path.dirname(entry_path), "tokenizer")
                        if os.path.isdir(fallback_tokenizer_path):
                            try:
                                self.tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer_path, padding_side="left")
                                print(f"Tokenizer loaded from fallback path: {fallback_tokenizer_path}", flush=True)
                            except Exception as e:
                                print(f"Failed to load tokenizer from fallback path: {e}", flush=True)
                        else:
                            print("Warning: No tokenizer found. You must manually set `self.tokenizer` for raw model inference.", flush=True)
                    return model
                except Exception as e:
                    print(f"Failed to load PyTorch model checkpoint: {e}", flush=True)

            # Case 3: Python Model Script
            elif entry.endswith(".py"):
                print(f"Attempting to execute model script: {entry_path}", flush=True)
                try:
                    model_namespace = {}
                    with open(entry_path, "r") as f:
                        exec(f.read(), model_namespace)
                    model = model_namespace.get("model", None)
                    # Set fallback tokenizer if not already set
                    if not hasattr(self, "tokenizer") or self.tokenizer is None:
                        # Try loading tokenizer from a default path or adjacent tokenizer folder
                        fallback_tokenizer_path = os.path.join(os.path.dirname(entry_path), "tokenizer")
                        if os.path.isdir(fallback_tokenizer_path):
                            try:
                                self.tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer_path, padding_side="left")
                                print(f"Tokenizer loaded from fallback path: {fallback_tokenizer_path}", flush=True)
                            except Exception as e:
                                print(f"Failed to load tokenizer from fallback path: {e}", flush=True)
                        else:
                            print("Warning: No tokenizer found. You must manually set `self.tokenizer` for raw model inference.", flush=True)
                    if model:
                        print("Participant's Python-based model loaded successfully.", flush=True)
                        return model
                    else:
                        print(f"No 'model' object found in {entry_path}.", flush=True)
                except Exception as e:
                    print(f"Failed to execute model script: {e}", flush=True)

        print("Error: No valid model found in 'model_submissions'.", flush=True)
        return None

    def load_participant_pipeline(self):
        """
        Dynamically loads the participant's LLM inference pipeline from 'model_submissions'.
        Supports multiple types of models, including:
        - Hugging Face models (transformers)
        - PyTorch models (saved as .pt or .pth files)
        - Custom Python scripts defining a model
        """
        print("Searching for participant's LLM pipeline in 'model_submissions'...", flush=True)

        if self.run_mode == "local":
            model_submissions_dir = os.path.join(self.base_dir, "../model_submission")
        else:
            model_dir_env = os.getenv("USE_INTERNAL_MODEL", "1").strip().lower()
            if model_dir_env in ["1", "true", "yes"]:
                model_submissions_dir = os.path.join(self.base_dir, "model_submission/snapshots")
            else:
                model_submissions_dir = os.path.join(self.base_dir, "model_submission/snapshots")
        
        if not os.path.exists(model_submissions_dir):
            print(f"Error: 'model_submissions' folder not found at {model_submissions_dir}", flush=True)
            return None

        for entry in os.listdir(model_submissions_dir):
            entry_path = os.path.join(model_submissions_dir, entry)

            # Case 1: Hugging Face Transformer Model
            if os.path.isdir(entry_path) and "config.json" in os.listdir(entry_path):
                print(f"Loading Hugging Face model from: {entry_path}", flush=True)
                try:
                    _pipeline = pipeline(
                        "text-generation",
                        model      = self.model,
                        tokenizer  = self.tokenizer,
                        batch_size = self.chunk_size,
                        max_length = self.max_length,
                        truncation = True,
                        do_sample  = self.do_sample,
                        temperature= self.temperature,
                        top_p      = self.top_p,
                        top_k      = self.top_k,
                    )

                    # Safely set pad_token if missing
                    if _pipeline.tokenizer.pad_token is None:
                        print("Tokenizer missing pad_token; setting it to eos_token", flush=True)
                        _pipeline.tokenizer.pad_token = _pipeline.tokenizer.eos_token
                        _pipeline.tokenizer.pad_token_id = _pipeline.tokenizer.eos_token_id

                    return _pipeline
                except Exception as e:
                    print(f"Failed to load Hugging Face pipeline: {e}", flush=True)

            # Case 2: PyTorch Model Checkpoint
            elif entry.endswith(".pt") or entry.endswith(".pth"):
                print(f"Loading PyTorch model checkpoint: {entry_path}", flush=True)
                try:
                    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = torch.load(entry_path, map_location=map_location)
                    print("PyTorch model loaded successfully.", flush=True)
                    return model  # Returning model directly, user must implement inference separately
                except Exception as e:
                    print(f"Failed to load PyTorch model checkpoint: {e}", flush=True)

            # Case 3: Python Script-based Model
            elif entry.endswith(".py"):
                print(f"Attempting to execute model script: {entry_path}", flush=True)
                try:
                    model_namespace = {}
                    with open(entry_path, "r") as f:
                        exec(f.read(), model_namespace)
                    model = model_namespace.get("model", None)
                    if model:
                        print("Python-based model loaded successfully.", flush=True)
                        return model
                    else:
                        print(f"No 'model' object found in {entry_path}.", flush=True)
                except Exception as e:
                    print(f"Failed to execute model script: {e}", flush=True)

        print("Error: No valid model found in 'model_submissions'.", flush=True)
        return None

    def load_json(self, filepath):
        """
        Load JSON data from the specified file.
        """
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {filepath}: {e}", flush=True)
            return None

    def load_template(self, filename):
        """
        Load the content of the template file.
        """
        filepath = os.path.join(self.template_dir, filename)
        try:
            with open(filepath, "r") as f:
                return f.read()
        except Exception as e:
            print(f"Error loading template {filename} from {filepath}: {e}", flush=True)
            return None   
    
    def load_and_sample_dataset(self):
        """
        Load and randomly sample QA pairs with predefined sample sizes per type.
        If --random is passed, randomly sample. Else, take the first N.
        """

        # Define QA types and corresponding filenames
        qa_types = {
            "multiple_choice": ("MC.json", self.sample_sizes.get("num_mc", 200)),
            "true_false": ("TF.json", self.sample_sizes.get("num_tf", 200)),
            "list": ("list.json", self.sample_sizes.get("num_list", 200)),
            "short": ("short.json", self.sample_sizes.get("num_short", 200)),
            "short_inverse": ("short_inverse.json", self.sample_sizes.get("num_short_inv", 200)),
            "multi_hop": ("multi_hop.json", self.sample_sizes.get("num_multi", 200)),
            "multi_hop_inverse": ("multi_hop_inverse.json", self.sample_sizes.get("num_multi_inv", 200)),
        }

        sampled_qa = {}  # Store sampled QA pairs by type

        for qa_type, (filename, sample_size) in qa_types.items():
            filepath = os.path.join(self.dataset_dir, filename)
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Ensure we do not exceed the available number of QA pairs
                    flat_data = [item for sublist in data for item in (sublist if isinstance(sublist, list) else [sublist])]
                    if self.random_sample:
                        sampled_data = random.sample(flat_data, min(sample_size, len(flat_data)))
                    else:
                        sampled_data = flat_data[:sample_size]
                    sampled_qa[qa_type] = sampled_data

            except Exception as e:
                print(f"Error loading {filename}: {e}", flush=True)
                sampled_qa[qa_type] = []  # Store an empty list if loading fails

        print(f"Successfully sampled {sum(len(v) for v in sampled_qa.values())} QA pairs.", flush=True)
        return sampled_qa


    def generate_prompt(self, template, qa, qa_type):
        """
        Generates a prompt for a single QA dictionary using the specified template.
        
        Args:
            template (str): The prompt template string with placeholders.
            qa (dict or list): A dictionary representing the QA pair, or a list of such dicts.
            qa_type (str): The QA type (e.g., "true_false", "multiple_choice", "list", etc.).

        Returns:
            str: The formatted prompt or an error message.
        """
        try:
            if not isinstance(qa, dict):
                print(f"Error: QA is not a dictionary after unpacking. Type: {type(qa)}")
                return "Invalid QA input."

            # Extract common fields
            question = qa.get("question", "Unknown Question")
            answer = qa.get("answer", "")
            options = qa.get("options", {})
            reasoning = qa.get("reasoning", "")
            false_answer = qa.get("false_answer", "")

            # Format based on QA type
            if qa_type == "true_false":
                return template.format(question=question)

            elif qa_type == "multiple_choice":
                if isinstance(options, dict):
                    letter_map = options  # Already mapped A-D
                elif isinstance(options, list):
                    letter_map = {chr(65 + i): opt for i, opt in enumerate(options)}
                else:
                    raise ValueError("Multiple choice options must be a list or dict.")

                return template.format(
                    question=question,
                    options_A=letter_map.get("A", "Option A missing"),
                    options_B=letter_map.get("B", "Option B missing"),
                    options_C=letter_map.get("C", "Option C missing"),
                    options_D=letter_map.get("D", "Option D missing"),
                )



            elif qa_type == "list":
                # Assign letters (A, B, C, ...) to each option
                if isinstance(options, list):
                    options_dict = {chr(65 + i): opt for i, opt in enumerate(options)}
                elif isinstance(options, dict):
                    options_dict = options
                else:
                    options_dict = {"A": str(options)}

                # Join options with letter prefixes (e.g., A: ..., B: ...)
                options_joined = "\n".join(f"{k}: {v}" for k, v in options_dict.items())
                return template.format(question=question, options_joined=options_joined)

            elif qa_type == "multi_hop":
                return template.format(question=question)

            elif qa_type == "multi_hop_inverse":
                return template.format(question=question, answer=answer, reasoning=reasoning)

            elif qa_type == "short":
                return template.format(question=question)

            elif qa_type == "short_inverse":
                return template.format(question=question, false_answer=false_answer)

            else:
                print(f"Warning: Unknown QA type '{qa_type}'")
                return f"Unsupported QA type: {qa_type}"

        except KeyError as ke:
            print(f"KeyError during prompt generation: {ke}")
            return f"Missing key in QA object: {ke}"

        except Exception as e:
            print(f"Exception in generate_prompt: {e}")
            print("QA Type:", qa_type)
            print("QA Object Dump:", json.dumps(qa, indent=2))
            return "Error generating prompt."




    def participant_model(self, prompt, qa_type=None):
        """
        Uses the participant's loaded model to generate a response based on the given prompt.
        Supports Hugging Face chat models (LLaMA-3), PyTorch models, and script-based models.
        """
        if not self.model:
            print("No participant model loaded. Returning default response.", flush=True)
            return "NO LLM IMPLEMENTED"

        try:
            # Handle Hugging Face chat models
            if isinstance(self.pipeline, TextGenerationPipeline):
                if isinstance(prompt, list):
                    conversations = [
                        [{"role": "system", "content": self.SYSTEM_MSG},
                        {"role": "user", "content": p}]
                        for p in prompt
                    ]
                    inputs = self.tokenizer.apply_chat_template(
                        conversations,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.model.device)
                else:
                    conversations = [
                        {"role": "system", "content": self.SYSTEM_MSG},
                        {"role": "user", "content": prompt}
                    ]
                    inputs = self.tokenizer.apply_chat_template(
                        conversations,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(self.model.device)

                eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                gen_cfg = GenerationConfig(
                        max_new_tokens = {
                            "multiple_choice": 1024,
                            "list": 1024,
                            "true_false": 1024,
                            "short": 2048,
                            "short_inverse": 2048,
                            "multi_hop": 2048,
                            "multi_hop_inverse": 2048,
                        }.get(qa_type, 32),
                        do_sample      = self.do_sample,
                        temperature    = self.temperature or 1.0 if self.do_sample else None,
                        top_p          = self.top_p,
                        top_k          = self.top_k,
                        top_n_tokens   = None,        # keep defaults
                        eos_token_id   = eot_id,
                        pad_token_id   = self.tokenizer.eos_token_id,
                    )

                output_ids = self.model.generate(inputs, generation_config=gen_cfg)
                response = self.tokenizer.batch_decode(
                    output_ids[:, inputs.shape[-1]:], skip_special_tokens=True
                )

                response = response if isinstance(prompt, list) else response[0]

            # Handle PyTorch models directly
            elif hasattr(self.model, "generate"):
                if isinstance(prompt, list):
                    input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)["input_ids"]
                else:
                    input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
                input_ids = input_ids.to(self.model.device)
                with torch.no_grad():
                    output_ids = self.model.generate(input_ids, max_length=self.max_length)
                response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                response = response if isinstance(prompt, list) else response[0]

            # Script-based model
            elif callable(self.model):
                response = self.model(prompt)

            else:
                print("Unknown model type. Returning default response.", flush=True)
                response = "NO LLM IMPLEMENTED"

        except Exception as e:
            print(f"Error during inference: {e}", flush=True)
            response = "ERROR DURING INFERENCE"

        # === Post-processing (preserve full model response) ===
        if isinstance(response, str):
            response = self._strip_noise(response)
        elif isinstance(response, list):
            response = [self._strip_noise(r) if isinstance(r, str) else r for r in response]

        return response


    def submit_true_false_questions(self):
        """
        Run inference on all True/False questions.
        Returns a dictionary containing:
        - responses: list of model-generated responses
        - inputs: list of original QA dicts
        """
        try:
            tf_data = self.sampled_qa_pairs.get("true_false", [])
            if not tf_data:
                print("No True/False data loaded.", flush=True)
                return {"responses": [], "inputs": []}

            template = self.load_template("tf_template.prompt")
            prompts = []

            for qa in tf_data:
                try:
                    prompts.append(self.generate_prompt(template, qa, "true_false"))
                except Exception as e:
                    print(f"Error generating prompt for TF QA: {e}", flush=True)
                    prompts.append("")

            try:
                responses = self._batched_inference(prompts, qa_type="true_false")
            except Exception as e:
                print(f"Error during model inference for TF QA: {e}", flush=True)
                responses = ["ERROR"] * len(prompts)

            return self._bundle(tf_data, responses, prompts)

        except Exception as e:
            print(f"Error submitting True/False questions: {e}", flush=True)
            return {"responses": [], "inputs": [], "prompts": []}


    def submit_multiple_choice_questions(self):
        """
        Run inference on all Multiple Choice questions.
        Returns a dictionary containing:
        - responses: list of model-generated responses
        - inputs: list of original QA dicts
        - prompts: list of generated prompts
        """
        try:
            mc_data = self.sampled_qa_pairs.get("multiple_choice", [])
            if not mc_data:
                print("No Multiple Choice data loaded.", flush=True)
                return {"responses": [], "inputs": [], "prompts": []}

            template = self.load_template("MC_template.prompt")
            prompts = []

            for qa in mc_data:
                try:
                    prompts.append(self.generate_prompt(template, qa, "multiple_choice"))
                except Exception as e:
                    print(f"Error generating prompt for MC QA: {e}", flush=True)
                    prompts.append("")

            try:
                responses = self._batched_inference(prompts, qa_type="multiple_choice")
            except Exception as e:
                print(f"Error during model inference for MC QA: {e}", flush=True)
                responses = ["ERROR"] * len(prompts)

            return self._bundle(mc_data, responses, prompts)

        except Exception as e:
            print(f"Error submitting Multiple Choice questions: {e}", flush=True)
            return {"responses": [], "inputs": [], "prompts": []}


    def submit_list_questions(self):
        """
        Run inference on all List questions.
        Returns a dictionary containing:
        - responses: list of model-generated responses
        - inputs: list of original QA dicts
        - prompts: list of generated prompts
        """
        try:
            list_data = self.sampled_qa_pairs.get("list", [])
            if not list_data:
                print("No List data loaded.", flush=True)
                return {"responses": [], "inputs": [], "prompts": []}

            template = self.load_template("list_template.prompt")
            prompts = []

            for qa in list_data:
                try:
                    prompts.append(self.generate_prompt(template, qa, "list"))
                except Exception as e:
                    print(f"Error generating prompt for List QA: {e}", flush=True)
                    prompts.append("")

            try:
                responses = self._batched_inference(prompts, qa_type="list")
            except Exception as e:
                print(f"Error during model inference for List QA: {e}", flush=True)
                responses = ["ERROR"] * len(prompts)

            return self._bundle(list_data, responses, prompts)

        except Exception as e:
            print(f"Error submitting List questions: {e}", flush=True)
            return {"responses": [], "inputs": [], "prompts": []}


    def submit_short_questions(self):
        """
        Run inference on all Short Answer questions.
        Returns a dictionary containing:
        - responses: list of model-generated responses
        - inputs: list of original QA dicts
        - prompts: list of generated prompts
        """
        try:
            short_data = self.sampled_qa_pairs.get("short", [])
            if not short_data:
                print("No Short Answer data loaded.", flush=True)
                return {"responses": [], "inputs": [], "prompts": []}

            template = self.load_template("short_template.prompt")
            prompts = []

            for qa in short_data:
                try:
                    prompts.append(self.generate_prompt(template, qa, "short"))
                except Exception as e:
                    print(f"Error generating prompt for Short QA: {e}", flush=True)
                    prompts.append("")

            try:
                responses = self._batched_inference(prompts, qa_type="short")
            except Exception as e:
                print(f"Error during model inference for Short QA: {e}", flush=True)
                responses = ["ERROR"] * len(prompts)

            return self._bundle(short_data, responses, prompts)

        except Exception as e:
            print(f"Error submitting Short Answer questions: {e}", flush=True)
            return {"responses": [], "inputs": [], "prompts": []}


    
    def submit_short_inverse_questions(self):
        """
        Run inference on all Short Inverse questions.
        Returns a dictionary containing:
        - responses: list of model-generated responses
        - inputs: list of original QA dicts
        - prompts: list of generated prompts
        """
        try:
            short_inverse_data = self.sampled_qa_pairs.get("short_inverse", [])
            if not short_inverse_data:
                print("No Short Inverse data loaded.", flush=True)
                return {"responses": [], "inputs": [], "prompts": []}

            template = self.load_template("short_inverse_template.prompt")
            prompts = []

            for qa in short_inverse_data:
                try:
                    prompts.append(self.generate_prompt(template, qa, "short_inverse"))
                except Exception as e:
                    print(f"Error generating prompt for Short Inverse QA: {e}", flush=True)
                    prompts.append("")

            try:
                responses = self._batched_inference(prompts, qa_type="short_inverse")
            except Exception as e:
                print(f"Error during model inference for Short Inverse QA: {e}", flush=True)
                responses = ["ERROR"] * len(prompts)

            return self._bundle(short_inverse_data, responses, prompts)

        except Exception as e:
            print(f"Error submitting Short Inverse questions: {e}", flush=True)
            return {"responses": [], "inputs": [], "prompts": []}


    def submit_multi_hop_questions(self):
        """
        Run inference on all Multi-hop questions.
        Returns a dictionary containing:
        - responses: list of model-generated responses
        - inputs: list of original QA dicts
        - prompts: list of generated prompts
        """
        try:
            mh_data = self.sampled_qa_pairs.get("multi_hop", [])
            if not mh_data:
                print("No Multi-hop data loaded.", flush=True)
                return {"responses": [], "inputs": [], "prompts": []}

            template = self.load_template("multi_hop_template.prompt")
            prompts = []

            for qa in mh_data:
                try:
                    prompts.append(self.generate_prompt(template, qa, "multi_hop"))
                except Exception as e:
                    print(f"Error generating prompt for Multi-hop QA: {e}", flush=True)
                    prompts.append("")

            try:
                responses = self._batched_inference(prompts, qa_type="multi_hop")
            except Exception as e:
                print(f"Error during model inference for Multi-hop QA: {e}", flush=True)
                responses = ["ERROR"] * len(prompts)

            return self._bundle(mh_data, responses, prompts)

        except Exception as e:
            print(f"Error submitting Multi-hop questions: {e}", flush=True)
            return {"responses": [], "inputs": [], "prompts": []}


    def submit_multi_hop_inverse_questions(self):
        """
        Run inference on all Multi-hop Inverse questions.
        Returns a dictionary containing:
        - responses: list of model-generated responses
        - inputs: list of original QA dicts
        - prompts: list of generated prompts
        """
        try:
            mh_inverse_data = self.sampled_qa_pairs.get("multi_hop_inverse", [])
            if not mh_inverse_data:
                print("No Multi-hop Inverse data loaded.", flush=True)
                return {"responses": [], "inputs": [], "prompts": []}

            template = self.load_template("multi_hop_inverse_template.prompt")
            prompts = []

            for qa in mh_inverse_data:
                try:
                    prompts.append(self.generate_prompt(template, qa, "multi_hop_inverse"))
                except Exception as e:
                    print(f"Error generating prompt for Multi-hop Inverse QA: {e}", flush=True)
                    prompts.append("")

            try:
                responses = self._batched_inference(prompts, qa_type="multi_hop_inverse")
            except Exception as e:
                print(f"Error during model inference for Multi-hop Inverse QA: {e}", flush=True)
                responses = ["ERROR"] * len(prompts)

            return self._bundle(mh_inverse_data, responses, prompts)

        except Exception as e:
            print(f"Error submitting Multi-hop Inverse questions: {e}", flush=True)
            return {"responses": [], "inputs": [], "prompts": []}

    def run_all_submissions(self):
        """
        Run inference for all QA types and save the generated responses to individual JSON files.
        Ensures the output directory exists.
        """
        try:
            output_dir = os.path.join(self.base_dir, "submission_output")
            os.makedirs(output_dir, exist_ok=True)

            qa_types = {
                "true_false": self.submit_true_false_questions,
                "multiple_choice": self.submit_multiple_choice_questions,
                "list": self.submit_list_questions,
                "short": self.submit_short_questions,
                "multi_hop": self.submit_multi_hop_questions,
                "short_inverse": self.submit_short_inverse_questions,
                "multi_hop_inverse": self.submit_multi_hop_inverse_questions,
            }

            for qa_type, submit_fn in qa_types.items():
                print(f"Running inference for: {qa_type}", flush=True)
                result = submit_fn()
                output_path = os.path.join(output_dir, f"{qa_type}.json")
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=4)
                print(f"Saved {qa_type} results to {output_path}", flush=True)

            print(f"All inference outputs saved to separate JSON files in {output_dir}", flush=True)

        except Exception as e:
            print(f"Error running all submissions: {e}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="ClinIQLink submission Script")
    parser.add_argument(
        "--mode",
        choices=["local", "container"],
        default="container",
        help="Run mode: 'local' for local dev, 'container' for inside Docker/Apptainer (default: container)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1028,
        help="Maximum token length for generated responses (default: 1028)"
    )

    # Add arguments for each QA type's sample size
    parser.add_argument("--num_tf", type=int, default=200, help="Number of True/False questions to evaluate")
    parser.add_argument("--num_mc", type=int, default=200, help="Number of Multiple Choice questions to evaluate")
    parser.add_argument("--num_list", type=int, default=200, help="Number of List questions to evaluate")
    parser.add_argument("--num_short", type=int, default=200, help="Number of Short Answer questions to evaluate")
    parser.add_argument("--num_short_inv", type=int, default=200, help="Number of Short Inverse questions to evaluate")
    parser.add_argument("--num_multi", type=int, default=200, help="Number of Multi-hop questions to evaluate")
    parser.add_argument("--num_multi_inv", type=int, default=200, help="Number of Multi-hop Inverse questions to evaluate")
    parser.add_argument("--random", action="store_true", help="If set, sample QA pairs randomly. Otherwise, take first N.")
    parser.add_argument("--chunk_size", type=int, default=2, help="Chunk size for batching prompts during inference (default: 2)")
    # ------------------------------------------------------------------
    # generation-control flags   (all optional; sensible defaults below)
    # ------------------------------------------------------------------
    parser.add_argument("--do_sample",   action="store_true",
                        help="Enable stochastic sampling (default off → greedy)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (>0). Ignored if --do_sample is not set")
    parser.add_argument("--top_p",       type=float, default=None,
                        help="Nucleus-sampling top-p (0‒1)")
    parser.add_argument("--top_k",       type=int,   default=None,
                        help="Top-k sampling (integer)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sample_sizes = {
        "num_tf": args.num_tf,
        "num_mc": args.num_mc,
        "num_list": args.num_list,
        "num_short": args.num_short,
        "num_short_inv": args.num_short_inv,
        "num_multi": args.num_multi,
        "num_multi_inv": args.num_multi_inv,
    }

    submit = ClinIQLinkSampleDatasetSubmit(
        run_mode      = args.mode,
        max_length    = args.max_length,
        sample_sizes  = sample_sizes,
        random_sample = args.random,
        chunk_size    = args.chunk_size,
        do_sample   = args.do_sample,
        temperature = args.temperature,
        top_p       = args.top_p,
        top_k       = args.top_k,
    )
    submit.run_all_submissions()