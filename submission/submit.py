import json
import os
import random
import argparse
import torch
import torch.distributed as dist
from contextlib import suppress
from transformers.pipelines import TextGenerationPipeline
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    pipeline,
    GenerationConfig,
)
import re
try:
    import flash_attn
    flash_installed = True
except ImportError:
    flash_installed = False

# Explicitly set HuggingFace & Torch cache paths for consistency and safety inside container
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/transformers"
os.environ["TORCH_HOME"] = "/app/.cache/torch"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def is_main_process() -> bool:
    return (not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0)

class ClinIQLinkSampleDatasetSubmit:
    def __init__(self, run_mode="container", max_length=1028, sample_sizes=None, random_sample=False, chunk_size=2,
                    do_sample=False, temperature=None, top_p=None, top_k=None, distributed=False):
        self.run_mode = run_mode.lower()
        self.max_length = max_length
        self.sample_sizes = sample_sizes or {}
        self.random_sample = random_sample
        self.chunk_size = chunk_size
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.distributed = distributed
        self.eos_id = None
        self.model_type = None
        self._pipeline_task = ""
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

    def _ensure_pad_token(self, tokenizer):
        """Make sure tokenizer has a valid pad-token (needed before pipeline is built)."""
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)


    def infer_model_type_from_config(self, cfg) -> str:
        """
        Normalize and map AutoConfig.model_type to a high-level model type string.
        """
        model_type = cfg.model_type.lower()

        # Map known model types to standardized categories
        if model_type.startswith("llama"):
            return "llama"
        elif model_type.startswith("mistral"):
            return "mistral"
        elif model_type.startswith("mixtral"):
            return "mixtral"
        elif model_type.startswith("falcon"):
            return "falcon"
        elif model_type.startswith("qwen"):
            return "qwen"
        elif model_type.startswith("phi"):
            return "phi"
        elif model_type in {"vicuna", "koala", "wizard", "openchat"}:
            return "vicuna"
        elif model_type in {"gptj", "gpt_neox", "gpt2"}:
            return "gpt"
        elif model_type.startswith("t5") or "flan" in model_type or "ul2" in model_type:
            return "t5"
        elif model_type.startswith("bert") or "roberta" in model_type or "deberta" in model_type:
            return "bert"
        elif model_type.startswith("albert"):
            return "albert"
        elif "mpt" in model_type:
            return "mpt"
        elif "bloom" in model_type:
            return "bloom"
        elif "chatglm" in model_type:
            return "chatglm"
        elif "claude" in model_type or "anthropic" in model_type:
            return "claude"
        elif "clinicalmosaic" in model_type:
            return "clinicalmosaic"

        return "unknown"

    
    def set_eos_id(self, model_type: str):
        """
        Given a model_type key, pick the correct end‑of‑sequence token,
        ensure the tokenizer knows about it, and save its ID as self.eos_id.
        """
        # 1) Define the family → EOS-token map
        eos_map = {
            "llama":   "<|eot_id|>",
            "vicuna":  "</s>",
            "falcon":  "</s>",
            "mistral": "</s>",
            "mixtral": "</s>",
            "qwen":    "<|endoftext|>",
            "gpt":     "",            # GPT‑2 uses default eos_token
            "phi":     "</s>",
            "claude":  "",            # default eos_token
            # extend as you add new model families…
        }

        # 2) Pick your token string (fall back to tokenizer.eos_token)
        token_str = eos_map.get(model_type.lower(), None) or self.tokenizer.eos_token
        if token_str is None:
            raise ValueError(f"No EOS token known for model_type={model_type}")

        # 3) If it's not already in vocab, register it as the eof special token
        if token_str not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"eos_token": token_str})
            # if your model needs resized embeddings:
            try:
                self.model.resize_token_embeddings(len(self.tokenizer))
            except AttributeError:
                pass

        # 4) Lookup and stash the ID
        self.eos_id = self.tokenizer.convert_tokens_to_ids(token_str)
        # sync the tokenizer attributes
        self.tokenizer.eos_token = token_str
        self.tokenizer.eos_token_id = self.eos_id

    
    def get_chat_template_string(self, model_path_or_id: str) -> str:
        model_id = model_path_or_id.lower()

        # === Meta LLaMA 2 / 3 / 4 ===
        if "llama" in model_id:
            return (
                # "<|begin_of_text|>"
                # "{% for message in messages %}"
                # "{% if message['role'] == 'system' %}"
                # "<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
                # "{% elif message['role'] == 'user' %}"
                # "<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
                # "{% elif message['role'] == 'assistant' %}"
                # "<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
                # "{% endif %}"
                # "{% endfor %}"
                # "<|start_header_id|>assistant<|end_header_id|>\n"

                "{% for message in messages %}"
                "{% if message['role']=='system' %}"
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                "{{ message['content'] }}<|eot_id|>"
                "{% elif message['role']=='user' %}"
                "<|start_header_id|>user<|end_header_id|>\n"
                "{{ message['content'] }}<|eot_id|>"
                "{% elif message['role']=='assistant' %}"
                "<|start_header_id|>assistant<|end_header_id|>\n"
                "{{ message['content'] }}<|eot_id|>"
                "{% endif %}{% endfor %}"
                "<|start_header_id|>assistant<|end_header_id|>\n"   
            )

        # === Mistral / Mixtral ===
        elif "mistral" in model_id or "mixtral" in model_id:
            return (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "<s>[INST] <<SYS>>\n{{ message['content'] }}\n<</SYS>>\n"
                "{% elif message['role'] == 'user' %}"
                "{{ message['content'] }} [/INST] "
                "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] }}</s>\n"
                "{% endif %}"
                "{% endfor %}"
                "<s>[INST] "
            )

        # === Qwen-style ===
        elif "qwen" in model_id:
            return (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'user' %}"
                "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'assistant' %}"
                "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
                "{% endif %}"
                "{% endfor %}"
                "<|im_start|>assistant\n"
            )

        # === Claude-style ===
        elif "claude" in model_id or "anthropic" in model_id:
            return (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "\\n\\nHuman: {{ message['content'] }}"
                "{% elif message['role'] == 'assistant' %}"
                "\\n\\nAssistant: {{ message['content'] }}"
                "{% endif %}"
                "{% endfor %}"
                "\\n\\nAssistant:"
            )

        # === Vicuna / GPT-style ===
        elif any(k in model_id for k in ["vicuna", "gpt", "koala", "openchat"]):
            return (
                "{% for message in messages %}"
                "{{ message['role'].capitalize() }}: {{ message['content'] }}\n"
                "{% endfor %}"
                "Assistant: "
            )

        # === Falcon-style fallback ===
        elif "falcon" in model_id or "deepseek" in model_id or "phi" in model_id:
            return (
                "{% for message in messages %}"
                "[{{ message['role'].upper() }}]: {{ message['content'] }}\n"
                "{% endfor %}"
                "[ASSISTANT]: "
            )

        # === Default fallback ===
        else:
            return (
                "{% for message in messages %}"
                "{{ message['role'] }}: {{ message['content'] }}\n"
                "{% endfor %}"
                "assistant: "
            )

    
    def apply_chat_template(self, tokenizer, messages, model_type=None, fallback_role="assistant", add_generation_prompt=True, max_length=2048):
        """
        Applies an appropriate chat-style prompt format based on the model's name or type.

        Args:
            tokenizer: HuggingFace tokenizer.
            messages: List of message dicts (each with 'role' and 'content').
            model_type: Optional model family name (inferred if not set).
            fallback_role: Role to use for generation prompt if needed.
            add_generation_prompt: Whether to add an empty assistant turn at the end.
            max_length: Max length for tokenizer truncation.

        Returns:
            input_ids suitable for model.generate()
        """

        if not messages:
            raise ValueError("`messages` must be a non-empty list of dicts.")

        model_id = (model_type or getattr(tokenizer, "name_or_path", "")).lower()

        # === Meta LLaMA 2 / 3 / 4 ===
        if "llama" in model_id:
            template = ("<|begin_of_text|>"
                        "{% for message in messages %}"
                        "{% if message['role'] == 'system' %}"
                        "<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
                        "{% elif message['role'] == 'user' %}"
                        "<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
                        "{% elif message['role'] == 'assistant' %}"
                        "<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
                        "{% endif %}"
                        "{% endfor %}")
            if add_generation_prompt:
                template += "<|start_header_id|>assistant<|end_header_id|>\n"
            # Render manually since we don't have Jinja2
            rendered = ""
            for m in messages:
                role = m["role"]
                content = m["content"].strip()
                rendered += f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>\n"
            if add_generation_prompt:
                rendered += f"<|start_header_id|>{fallback_role}<|end_header_id|>\n"
            return tokenizer(rendered.strip(), return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]

        # === Mistral / Mixtral ===
        elif "mistral" in model_id or "mixtral" in model_id:
            template = ""
            for m in messages:
                role = m["role"]
                content = m["content"].strip()
                if role == "system":
                    template += f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n"
                elif role == "user":
                    template += f"{content} [/INST] "
                elif role == "assistant":
                    template += f"{content}</s>\n"
            if add_generation_prompt:
                template += "<s>[INST] "
            return tokenizer(template.strip(), return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]

        # === Falcon / Phi / Qwen / DeepSeek (shared format) ===
        elif any(k in model_id for k in ["falcon", "phi", "qwen", "deepseek"]):
            template = ""
            for m in messages:
                role = m["role"].upper()
                content = m["content"].strip()
                template += f"[{role}]: {content}\n"
            if add_generation_prompt:
                template += "[ASSISTANT]: "
            return tokenizer(template.strip(), return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]

        # === Vicuna ===
        elif "vicuna" in model_id:
            template = ""
            for m in messages:
                role = m["role"]
                content = m["content"].strip()
                template += f"### {role.capitalize()}:\n{content}\n"
            if add_generation_prompt:
                template += "### Assistant:\n"
            return tokenizer(template.strip(), return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]

        # === Claude-like ===
        elif any(k in model_id for k in ["claude", "anthropic"]):
            template = ""
            for m in messages:
                if m["role"] == "user":
                    template += f"\n\nHuman: {m['content'].strip()}"
                elif m["role"] == "assistant":
                    template += f"\n\nAssistant: {m['content'].strip()}"
            if add_generation_prompt:
                template += "\n\nAssistant:"
            return tokenizer(template.strip(), return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]

        # === Koala / GPT-J / GPT-NeoX fallback ===
        elif any(k in model_id for k in ["gpt", "openchat", "koala", "wizard"]):
            template = ""
            for m in messages:
                template += f"{m['role'].capitalize()}: {m['content'].strip()}\n"
            if add_generation_prompt:
                template += "Assistant: "
            return tokenizer(template.strip(), return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]

        # === Default fallback ===
        else:
            template = "\n\n".join(m["content"].strip() for m in messages if m["role"] in {"system", "user"})
            if add_generation_prompt:
                template += f"\n\n{fallback_role.capitalize()}:"
            return tokenizer(template.strip(), return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]
    
    def load_participant_tokenizer(self, padding_side):
        """Load tokenizer from a directory with fallback for non-fast versions."""
        if self.run_mode == "local":
            model_submissions_dir = os.path.join(self.base_dir, "../model_submission")
        else:
            model_dir_env = os.getenv("USE_INTERNAL_MODEL", "1").strip().lower()
            if model_dir_env in ["1", "true", "yes"]:
                model_submissions_dir = os.path.join(self.base_dir, "model_submission/snapshots")
            else:
                model_submissions_dir = os.path.join(self.base_dir, "model_submission/snapshots")
        
        if not os.path.exists(model_submissions_dir):
            raise FileNotFoundError(f"[Tokenizer] No 'model_submissions' folder at {model_submissions_dir}")


        for entry in os.listdir(model_submissions_dir):
            entry_path = os.path.join(model_submissions_dir, entry)
            if os.path.isdir(entry_path) and "config.json" in os.listdir(entry_path):
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        entry_path,
                        trust_remote_code=True,
                        padding_side=padding_side,
                        use_fast=True,
                    )
                    self._ensure_pad_token(self.tokenizer)
                    print(f"[Tokenizer] Loaded from {entry_path}", flush=True)
                    return  # Success
                except Exception as e:
                    print(f"[WARN] fast tokenizer failed: {e} – retrying with use_fast=False", flush=True)
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        entry_path,
                        trust_remote_code=True,
                        padding_side=padding_side,
                        use_fast=False,
                    )
                    self._ensure_pad_token(self.tokenizer)
                    print(f"[Tokenizer] Loaded (slow) from {entry_path}", flush=True)
                    return
                
                except Exception as e2:
                    print(f"[Tokenizer] Failed loading tokenizer from {entry_path}: {e2}", flush=True)

        raise RuntimeError("[Tokenizer] No valid tokenizer found in model_submissions.")
    
    
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

                    # ---------------------------------------------------------------------
                    # Pick the right HF head + tell the rest of the code which pipeline to
                    # build.  Order matters: we exit on the *first* rule that matches.
                    # ---------------------------------------------------------------------
                    cfg = AutoConfig.from_pretrained(
                            entry_path,
                            trust_remote_code=True,
                            local_files_only=True
                        )


                    auto_map = getattr(cfg, "auto_map", {}) or {}      # present in many custom repos

                    LLAMA_FAMILY = {"llama", "llama2", "llama3", "llama4"}

                    # 0) LLaMA and friends (all decoder‑only) as well as custom ones for some trouble architectures 
                    if cfg.model_type in LLAMA_FAMILY:
                        ModelClass       = AutoModelForCausalLM
                        self._pipeline_task = "text-generation"
                    
                    elif cfg.model_type in {"albert", "albert_xlarge"}:
                        ModelClass       = AutoModelForMaskedLM
                        self._pipeline_task = "fill-mask"

                    elif cfg.model_type.startswith("t5") or "flan" in cfg.model_type:
                        ModelClass       = AutoModelForSeq2SeqLM
                        self._pipeline_task = "text2text-generation"

                    elif cfg.model_type == "clinical_mosaic" or "clinicalmosaic" in entry_path.lower():
                        ModelClass       = AutoModelForSequenceClassification
                        self._pipeline_task = "text-classification"

                    # ─── handle Mistral-3 / Mixtral custom configs ───
                    elif cfg.model_type == "mistral3":
                        from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
                        ModelClass       = Mistral3ForConditionalGeneration
                        self._pipeline_task = "text-generation"
                        
                    # ─── handle Mixtral and earlier Mistral families ───
                    elif cfg.model_type in {"mixtral", "mistral"}:
                        ModelClass       = AutoModelForCausalLM
                        self._pipeline_task = "text-generation"

                    # 1) Custom repositories that ship their own heads
                    #    (ClinicalMosaic, future Gemma‑Flash, etc.)
                    elif "AutoModelForSequenceClassification" in auto_map:
                        ModelClass       = AutoModelForSequenceClassification
                        self._pipeline_task = "text-classification"

                    elif "AutoModelForMaskedLM" in auto_map:
                        ModelClass       = AutoModelForMaskedLM
                        self._pipeline_task = "fill-mask"

                    # 2) Encoder‑decoder models (T5/BART/UL2/FLAN, …)
                    elif getattr(cfg, "is_encoder_decoder", False):
                        ModelClass       = AutoModelForSeq2SeqLM
                        self._pipeline_task = "text2text-generation"

                    # 3) Architectures list tells us exactly which head to use
                    elif any(a.endswith("ForMaskedLM")           for a in getattr(cfg, "architectures", []) or []):
                        ModelClass       = AutoModelForMaskedLM
                        self._pipeline_task = "fill-mask"

                    elif any(a.endswith("ForQuestionAnswering")  for a in getattr(cfg, "architectures", []) or []):
                        ModelClass       = AutoModelForQuestionAnswering
                        self._pipeline_task = "question-answering"

                    elif any(a.endswith("ForSequenceClassification") for a in getattr(cfg, "architectures", []) or []):
                        ModelClass       = AutoModelForSequenceClassification
                        self._pipeline_task = "text-classification"


                    # 4) Large decoder‑only families that don’t set is_decoder=True
                    #    (Falcon‑3, Mistral‑24B, Mixtral, Qwen‑2, etc.)
                    else:
                        DECODER_ONLY = {
                            # GPT style
                            "gpt2", "gptj", "gpt_neo", "gpt_neox",
                            # Meta / Mistral AI
                            "llama", "llama2", "llama3", "llama4",
                            "mistral", "mistral3",
                            "mixtral",
                            # Others
                            "bloom", "gemma",
                            "falcon", "falcon2", "falcon3",
                            "qwen", "qwen2",
                            "mpt"
                        }
                        if getattr(cfg, "is_decoder", False) or cfg.model_type in DECODER_ONLY:
                            ModelClass       = AutoModelForCausalLM
                            self._pipeline_task = "text-generation"
                        else:
                            print(f"[WARN] Unknown model_type '{cfg.model_type}', falling back to causal‑LM.", flush=True)
                            ModelClass       = AutoModelForCausalLM
                            self._pipeline_task = "text-generation"

                    # Final guard – if *anything* above forgot to set a task
                    if not self._pipeline_task:
                        self._pipeline_task = "text-generation"
                    
                    # === Load model, preferring safetensors but falling back to PyTorch .bin ===

                    #added in as mistral / mixtral was slow 
                    extra_model_kwargs = {}
                    if flash_installed and cfg.model_type in {
                        "llama", "llama2", "llama3", "llama4",
                        "mistral", "mixtral", "mistral3",
                        "granite", "dbrx",
                        "falcon", "gemma",
                        "phi", "phi2", "phi3",
                        "qwen2", "qwen2moe",
                        "stablelm", "starcoder2"
                    }:
                        extra_model_kwargs["attn_implementation"] = "flash_attention_2"
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.set_float32_matmul_precision("high")
                        
                    if "attn_implementation" in extra_model_kwargs:
                        print("[INFO] Using FlashAttention2 for model acceleration.", flush=True)
                    else:
                        print("[INFO] FlashAttention2 not available – using default attention mechanism.", flush=True)

                    try:
                        model = ModelClass.from_pretrained(
                            entry_path,
                            trust_remote_code=True,
                            use_safetensors=True,
                            device_map=device_map,
                            torch_dtype=torch_dtype,
                            local_files_only=True,
                            **extra_model_kwargs, 
                        )
                        print(f"[INFO] Loaded model from {entry_path} with safetensors.", flush=True)

                    except Exception as e:
                        err_str = str(e).lower()

                        # Detect the common “no safetensors found” failure
                        if "safetensors" in err_str:
                            print(f"[WARN] No safetensors weights at {entry_path}; retrying with use_safetensors=False", flush=True)

                            # Retry with standard PyTorch weights
                            model = ModelClass.from_pretrained(
                                entry_path,
                                trust_remote_code=True,
                                use_safetensors=False,
                                device_map=device_map,
                                torch_dtype=torch_dtype,
                                local_files_only=True
                            )
                            print(f"[INFO] Loaded model from {entry_path} with PyTorch .bin weights.", flush=True)

                        else:
                            # Some other error: propagate with context
                            print(f"Unrecognised / unsupported model type '{cfg.model_type}'. "
                                    "Add an appropriate head mapping in load_participant_model().")
                            raise RuntimeError(f"Failed to load model at {entry_path}: {e}") from e
                    
                    # try:
                    #     self.tokenizer = AutoTokenizer.from_pretrained(
                    #         entry_path,
                    #         trust_remote_code=True,
                    #         padding_side="left",
                    #         use_fast=True,
                    #     )
                    # except Exception as e:
                    #     print(f"[WARN] fast tokenizer failed: {e} – retrying with use_fast=False", flush=True)
                    #     self.tokenizer = AutoTokenizer.from_pretrained(
                    #         entry_path,
                    #         trust_remote_code=True,
                    #         padding_side="left",
                    #         use_fast=False,
                    #     )

                    is_enc_dec = getattr(cfg, "is_encoder_decoder", False)
                    padding_side = "right" if is_enc_dec else "left"

                    self.load_participant_tokenizer(padding_side)
                    if self.tokenizer is None:
                        raise RuntimeError("Tokenizer failed to load. Cannot continue.")
                    
                    # make sure pad_token is defined
                    self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
                    self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
                    self._ensure_pad_token(self.tokenizer)
                    model.config.pad_token_id = self.tokenizer.pad_token_id
                    if hasattr(model, "generation_config"):
                        model.generation_config.pad_token_id = self.tokenizer.pad_token_id
                    
                    print("Participant's Hugging Face model loaded successfully.", flush=True)
                    print(f"Loaded {cfg.model_type} as {ModelClass.__name__} + task='{self._pipeline_task}'", flush=True)

                    # # optional chat_template for LLaMA-like chat models
                    # if getattr(self.tokenizer, "chat_template", None) is None and "llama" in cfg.model_type:
                    #     print("Setting manual chat template for LLaMA tokenizer...", flush=True)
                    #     self.tokenizer.chat_template = (
                    #         "{% for message in messages %}"
                    #         "{% if message['role']=='system' %}"
                    #         "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                    #         "{{ message['content'] }}<|eot_id|>"
                    #         "{% elif message['role']=='user' %}"
                    #         "<|start_header_id|>user<|end_header_id|>\n"
                    #         "{{ message['content'] }}<|eot_id|>"
                    #         "{% elif message['role']=='assistant' %}"
                    #         "<|start_header_id|>assistant<|end_header_id|>\n"
                    #         "{{ message['content'] }}<|eot_id|>"
                    #         "{% endif %}{% endfor %}"
                    #         "<|start_header_id|>assistant<|end_header_id|>\n"
                    #     )

                    if getattr(self.tokenizer, "chat_template", None) is None:
                        self.model_type = self.infer_model_type_from_config(cfg)
                        print(f"[INFO] Inferred Model Type: {self.model_type}", flush=True)
                        print("Setting manual chat template for tokenizer...", flush=True)
                        self.tokenizer.chat_template = self.get_chat_template_string(self.model_type)
                        print("Setting EOS_ID for tokenizer...", flush=True)
                        self.set_eos_id(self.model_type)


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
                                self._ensure_pad_token(self.tokenizer)
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
                                self._ensure_pad_token(self.tokenizer)
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

                # build the kwargs for whichever pipeline we're about to create
                pipeline_kwargs = {
                    "model":      self.model,
                    "tokenizer":  self.tokenizer,
                    "batch_size": self.chunk_size,
                }

                # generation pipelines get full control over sampling / length
                if self._pipeline_task in ("text-generation", "text2text-generation"):
                    pipeline_kwargs.update({
                        "max_length": self.max_length,
                        "truncation": True,
                        "do_sample":  self.do_sample,
                        "temperature": self.temperature,
                        "top_p":       self.top_p,
                        "top_k":       self.top_k,
                    })

                # fill-mask only supports mask-specific args (e.g. top_k)
                elif self._pipeline_task == "fill-mask":
                    if self.top_k is not None:
                        pipeline_kwargs["top_k"] = self.top_k

                # classification & QA just need truncation
                else:
                    pipeline_kwargs["truncation"] = True

                try:
                    _pipeline = pipeline(self._pipeline_task, **pipeline_kwargs)
                    self._ensure_pad_token(_pipeline.tokenizer)
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

    def participant_model(self, prompt, qa_type=None):
        if not self.pipeline:
            return "ERROR: No pipeline loaded"

        try:
            # HuggingFace pipelines auto-handle list vs str input
            outputs = self.pipeline(prompt)
            if isinstance(outputs, list):
                if self._pipeline_task in ["text-generation", "text2text-generation"]:
                    return [o["generated_text"] if "generated_text" in o else o["text"] for o in outputs]
                elif self._pipeline_task == "fill-mask":
                    return [o[0]["token_str"] for o in outputs]  # top prediction
                elif self._pipeline_task in ["text-classification", "question-answering"]:
                    return [o["label"] if "label" in o else o["answer"] for o in outputs]
                else:
                    return outputs
            elif isinstance(outputs, dict):
                return outputs.get("answer") or outputs.get("label") or outputs.get("text") or str(outputs)
            else:
                return str(outputs)

        except Exception as e:
            print(f"Error in participant_model(): {e}", flush=True)
            return "ERROR DURING INFERENCE"

    
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
                    batch_convs = [
                        [
                            {"role": "system", "content": self.SYSTEM_MSG},
                            {"role": "user",   "content": p}
                        ]
                        for p in prompt
                    ]
                else:
                    batch_convs = [[
                        {"role": "system", "content": self.SYSTEM_MSG},
                        {"role": "user",   "content": prompt}
                    ]]
                    
                # Process prompts (single or batch)
                all_input_ids = []
                all_attention_masks = []
                
                for i, conv in enumerate(batch_convs):
                    # print(f"[DEBUG] Processing conversation {i+1}/{len(batch_convs)}", flush=True)
                    
                    try:
                        # Apply chat template 
                        result = self.tokenizer.apply_chat_template(
                            conv,
                            add_generation_prompt=True,
                            return_tensors="pt",
                            truncation=True,
                            max_length=self.max_length,
                        )
                        
                        # Handle both dictionary and direct tensor return types
                        if isinstance(result, dict) and "input_ids" in result:
                            input_ids = result["input_ids"]
                            attention_mask = result.get("attention_mask", torch.ones_like(input_ids))
                        elif isinstance(result, torch.Tensor):
                            input_ids = result
                            attention_mask = torch.ones_like(input_ids)
                        else:
                            raise ValueError(f"Unexpected result type: {type(result)}")
                            
                        # Ensure proper shape
                        if len(input_ids.shape) == 1:
                            input_ids = input_ids.unsqueeze(0)
                            attention_mask = attention_mask.unsqueeze(0)
                            
                        all_input_ids.append(input_ids)
                        all_attention_masks.append(attention_mask)
                            
                    except Exception as e:
                        print(f"[ERROR] Failed chat template: {e}. Using fallback.", flush=True)
                        # Fallback to basic encoding
                        fallback_input = self.tokenizer.encode(
                            str(conv),  # Convert conversation to string
                            return_tensors="pt",
                            truncation=True,
                            max_length=self.max_length,
                        )
                        all_input_ids.append(fallback_input)
                        all_attention_masks.append(torch.ones_like(fallback_input))
                
                # Process collected tensors

                # # Find maximum sequence length for padding
                # max_len = max(ids.shape[1] for ids in all_input_ids)
                
                # # Pad sequences as needed
                # padded_ids = []
                # padded_masks = []
                
                # pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
                
                # for ids, mask in zip(all_input_ids, all_attention_masks):
                #     pad_len = max_len - ids.shape[1]
                #     if pad_len > 0:
                #         # Add padding
                #         padded_id = torch.nn.functional.pad(ids, (0, pad_len), value=pad_token_id)
                #         padded_mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
                #     else:
                #         padded_id = ids
                #         padded_mask = mask
                    
                #     padded_ids.append(padded_id)
                #     padded_masks.append(padded_mask)
                
                # # Combine all tensors and move to device
                # input_ids = torch.cat(padded_ids, dim=0).to(self.model.device)
                # attention_mask = torch.cat(padded_masks, dim=0).to(self.model.device)
                
                # # re‐encode the batch of conversations (as strings) with left‐padding
                
                # texts = [
                #     self.tokenizer.decode(ids.squeeze(), skip_special_tokens=True)
                #     for ids in all_input_ids
                # ]

                # batch = self.tokenizer(
                #         texts,
                #         return_tensors="pt",
                #         padding=True,
                #         truncation=True,
                #         # max_length=self.max_length,
                #     )

                # input_ids      = batch["input_ids"].to(self.model.device)
                # attention_mask = batch["attention_mask"].to(self.model.device)

                # assume all_input_ids and all_attention_masks are Python lists of lists or tensors
                # convert each tensor to a plain Python list of ints
                batch_encoding = {
                    "input_ids":      [ids.squeeze(0).tolist() for ids in all_input_ids],
                    "attention_mask": [mask.squeeze(0).tolist() for mask in all_attention_masks],
                }

                # this pads on the longest sequence, using tokenizer.pad_token_id/padding_side
                batch = self.tokenizer.pad(
                    batch_encoding,
                    padding=True,
                    return_tensors="pt",
                )

                input_ids      = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)

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
                        eos_token_id   = self.eos_id,
                        pad_token_id   = self.tokenizer.eos_token_id,
                    )

                # Recalculate attention mask as needed
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=gen_cfg,
                    #max_new_tokens=self.max_length,
                    max_new_tokens=gen_cfg.max_new_tokens,  #added as mistral / mixtral was extremely slow 
                    pad_token_id=self.tokenizer.pad_token_id
                )

                response = self.tokenizer.batch_decode(
                    output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
                )
                response = response if isinstance(prompt, list) else response[0]

            # Handle PyTorch models directly
            elif hasattr(self.model, "generate"):
                if isinstance(prompt, list):
                    input_ids = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,  
                    )["input_ids"]
                else:
                    input_ids = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,  
                    )["input_ids"]

                input_ids = input_ids.to(self.model.device)
                with torch.no_grad():
                    attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        # max_new_tokens=self.max_length,
                        max_new_tokens=gen_cfg.max_new_tokens,  #added as mistral / mixtral was extremely slow 
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # print(f"[DEBUG] input_ids shape: {input_ids.shape}", flush=True)
                # print(f"[DEBUG] output_ids shape: {output_ids.shape}", flush=True)
                
                response = self.tokenizer.batch_decode(
                    output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
                )
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
                if self.distributed:
                    if is_main_process():
                        with open(output_path, "w") as f:
                            json.dump(result, f, indent=4)
                        print(f"Saved {qa_type} results to {output_path}", flush=True)

                    if is_main_process():
                        print(f"All inference outputs saved to separate JSON files in {output_dir}", flush=True)

                else:
                    with open(output_path, "w") as f:
                        json.dump(result, f, indent=4)
                    print(f"Saved {qa_type} results to {output_path}", flush=True)

            if self.distributed:
                if dist.is_initialized():
                        dist.barrier()

            if self.distributed:
                if is_main_process():
                    print(f"[DONE] All outputs in {output_dir}", flush=True)
            else:
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
    
    parser.add_argument(
        "--local_rank", type=int, default=os.environ.get("LOCAL_RANK", 0),
        help="Local rank passed by torchrun"
    )

    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable torch.distributed (multi-GPU / multi-node via torchrun or SLURM) (launch with torchrun or srun –ntasks>1"
    )

    
    return parser.parse_args()


def init_distributed(local_rank: int):
    """
    Initialise torch.distributed if launched with --distributed.
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")
    if dist.is_initialized():
        print("[WARN] torch.distributed already initialized.", flush=True)
        return
    if "RANK" not in os.environ:
        raise RuntimeError("Distributed mode requested but RANK env var not found. Are you using torchrun or SLURM with srun?")
    
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    print(f"[Distributed] Initialized rank {dist.get_rank()}/{dist.get_world_size()} on device {torch.cuda.current_device()}", flush=True)


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


    if args.distributed:
        init_distributed(args.local_rank)


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
        distributed = args.distributed,
    )
    submit.run_all_submissions()