import json
import os
import random
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines import TextGenerationPipeline
import re


# Explicitly set HuggingFace & Torch cache paths for consistency and safety inside container
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/transformers"
os.environ["TORCH_HOME"] = "/app/.cache/torch"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import nltk

# Ensure NLTK uses the global path set via Docker ENV
nltk.data.path.append(os.environ.get("NLTK_DATA", "/usr/local/nltk_data"))

# Sanity check: ensure 'punkt' is found (optional, should already be present)
try:
    nltk.data.find("tokenizers/punkt")
    print("NLTK 'punkt' tokenizer available.", flush=True)
except LookupError:
    print("Warning: 'punkt' tokenizer not found. This should not happen in a prebuilt container.", flush=True)
    nltk.download('punkt', quiet=True)
    print("Downloading 'punkt' tokenizer...")


class ClinIQLinkSampleDatasetSubmit:
    def __init__(self, run_mode="container", max_length=1028, sample_sizes=None):
        self.run_mode = run_mode.lower()
        self.max_length = max_length
        self.sample_sizes = sample_sizes or {}
        # Base directories and setup depending on run mode
        if run_mode == "container":
            print("Running in container mode.", flush=True)
            self.base_dir = "/app"
            self.st_model = SentenceTransformer('/app/models/sentence-transformers_all-MiniLM-L6-v2', local_files_only=True)
        else:
            print("Running in local mode.", flush=True)
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
            self.st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # HF cache used

        # Dataset and template directory setup
        self.dataset_dir = os.getenv("DATA_DIR", os.path.join(self.base_dir, "..", "data"))
        self.template_dir = os.path.join(self.base_dir, "submission_template")

        # NLTK local data directory
        self.nltk_data_dir = os.path.join(self.base_dir, "nltk_data")
        nltk.data.path.append(self.nltk_data_dir)

        # Ensure both 'punkt' and 'punkt_tab' are downloaded
        for resource in ['punkt', 'punkt_tab']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
                print(f"NLTK '{resource}' tokenizer found.", flush=True)
            except LookupError:
                print(f"NLTK '{resource}' tokenizer not found. Downloading to local directory...", flush=True)
                nltk.download(resource, download_dir=self.nltk_data_dir)
                nltk.data.path.append(self.nltk_data_dir)


        # Placeholder: load the participant's LLM model and inference pipeline.
        self.model = self.load_participant_model()
        self.pipeline = self.load_participant_pipeline()
        # Load and sample the dataset
        self.sampled_qa_pairs = self.load_and_sample_dataset()

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
                model_submissions_dir = os.path.join(self.base_dir, "model_submission")
            else:
                model_submissions_dir = os.path.join(self.base_dir, "../model_submission")
        
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
                    model = AutoModelForCausalLM.from_pretrained(entry_path, trust_remote_code=True, use_safetensors=True)
                    self.tokenizer = AutoTokenizer.from_pretrained(entry_path, trust_remote_code=True)
                    print("Participant's Hugging Face model loaded successfully.", flush=True)
                    return model
                except Exception as e:
                    print(f"Failed to load Hugging Face model: {e}", flush=True)

            # Case 2: Model Checkpoint (PyTorch)
            elif entry.endswith(".pt") or entry.endswith(".pth"):
                print(f"Loading PyTorch model checkpoint: {entry_path}", flush=True)
                try:
                    model = torch.load(entry_path, map_location=torch.device("cpu"))
                    print("Participant's PyTorch model loaded successfully.", flush=True)
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
                model_submissions_dir = os.path.join(self.base_dir, "model_submission")
            else:
                model_submissions_dir = os.path.join(self.base_dir, "../model_submission")
        
        if not os.path.exists(model_submissions_dir):
            print(f"Error: 'model_submissions' folder not found at {model_submissions_dir}", flush=True)
            return None

        for entry in os.listdir(model_submissions_dir):
            entry_path = os.path.join(model_submissions_dir, entry)

            # Case 1: Hugging Face Transformer Model
            if os.path.isdir(entry_path) and "config.json" in os.listdir(entry_path):
                print(f"Loading Hugging Face model from: {entry_path}", flush=True)
                try:
                    model = AutoModelForCausalLM.from_pretrained(entry_path, trust_remote_code=True, use_safetensors=True)
                    self.tokenizer = AutoTokenizer.from_pretrained(entry_path, trust_remote_code=True)
                    print("Hugging Face model loaded successfully.", flush=True)
                    return pipeline("text-generation", model=model, tokenizer=self.tokenizer)
                except Exception as e:
                    print(f"Failed to load Hugging Face model: {e}", flush=True)

            # Case 2: PyTorch Model Checkpoint
            elif entry.endswith(".pt") or entry.endswith(".pth"):
                print(f"Loading PyTorch model checkpoint: {entry_path}", flush=True)
                try:
                    model = torch.load(entry_path, map_location=torch.device("cpu"))
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
        """ Load and randomly sample QA pairs with predefined sample sizes per type. """

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
                    sampled_data = random.sample(flat_data, min(sample_size, len(flat_data)))


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
                # Ensure options is a list and map to A–D
                if isinstance(options, list):
                    letter_map = {chr(65 + i): opt for i, opt in enumerate(options)}
                else:
                    raise ValueError("Multiple choice options should be a list.")

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



    def compute_f1_score(self, true_list, pred_list):
        """
        Compute precision, recall, and F1 score for list-type answers.
        """
        try:
            true_set = set(item.strip().lower() for item in true_list)
            pred_set = set(item.strip().lower() for item in pred_list)
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            return precision, recall, f1
        except Exception as e:
            print(f"Error computing F1 score: {e}", flush=True)
            return 0.0, 0.0, 0.0


    def evaluate_true_false(self, expected, prediction):
        """
        Evaluate True/False questions: returns 1 if answers match, else 0.
        """
        try:
            return 1.0 if expected.strip().lower() == prediction.strip().lower() else 0.0
        except Exception as e:
            print(f"Error evaluating True/False question: {e}", flush=True)
            return 0.0

    def evaluate_multiple_choice(self, expected, prediction):
        """
        Evaluate Multiple Choice questions: returns 1 if the selected option matches the expected answer.
        """
        try:
            return 1.0 if expected.strip().lower() == prediction.strip().lower() else 0.0
        except Exception as e:
            print(f"Error evaluating Multiple Choice question: {e}", flush=True)
            return 0.0

    def evaluate_list(self, expected, prediction, options=None):
        """
        Evaluate List questions using the F1 score.
        `expected`: list of correct answer strings.
        `prediction`: string of comma-separated option letters (e.g., "A, C, E").
        `options`: original list of options shown to the model (e.g., ["A", "B", "C", ...]).
        """
        try:
            if isinstance(prediction, list):
                prediction = ", ".join(str(p) for p in prediction)

            pred_letters = [item.strip().upper() for item in prediction.split(",") if item.strip()]
            
            if not options or not isinstance(options, list):
                print("Warning: Options missing or not a list for list-type QA.")
                return 0.0

            # Build mapping of letter -> option value
            letter_to_option = {chr(65 + idx): val.strip().lower() for idx, val in enumerate(options)}
            # Now reverse: option value -> letter
            option_to_letter = {v: k for k, v in letter_to_option.items()}

            # Convert expected list of values to expected list of letters
            expected_letters = [option_to_letter.get(ans.strip().lower()) for ans in expected]
            expected_letters = [e for e in expected_letters if e]  # Remove None

            _, _, f1 = self.compute_f1_score(expected_letters, pred_letters)
            return f1
        except Exception as e:
            print(f"Error evaluating List question: {e}", flush=True)
            return 0.0




    def compute_word_level_similarity(self, expected_text, prediction_text):
        """
        Compute a word-level similarity score using token embeddings.
        For each word in expected_text, find the maximum cosine similarity with any word in prediction_text,
        and vice versa, then compute the harmonic mean of the averaged precision and recall.
        Returns a float score between 0 and 1.
        """
        try:
            expected_words = expected_text.split()
            prediction_words = prediction_text.split()
            if not expected_words or not prediction_words:
                return 0.0
            expected_embeds = self.st_model.encode(expected_words, convert_to_tensor=True).cpu().numpy()
            prediction_embeds = self.st_model.encode(prediction_words, convert_to_tensor=True).cpu().numpy()
            
            sims_expected = [np.max(cosine_similarity([embed], prediction_embeds)) for embed in expected_embeds]
            sims_prediction = [np.max(cosine_similarity([embed], expected_embeds)) for embed in prediction_embeds]
            
            recall = np.mean(sims_expected)
            precision = np.mean(sims_prediction)
            if (precision + recall) == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)
        except Exception as e:
            print(f"Error computing word-level similarity: {e}", flush=True)
            return 0.0


    def compute_sentence_level_similarity(self, expected_text, prediction_text):
        """
        Compute sentence-level similarity by splitting texts into sentences,
        encoding them, and averaging the maximum cosine similarity for each expected sentence.
        Returns a float score between 0 and 1.
        """
        try:
            expected_sentences = nltk.sent_tokenize(expected_text)
            prediction_sentences = nltk.sent_tokenize(prediction_text)
            if not expected_sentences or not prediction_sentences:
                return 0.0
            expected_embeds = self.st_model.encode(expected_sentences, convert_to_tensor=True).cpu().numpy()
            prediction_embeds = self.st_model.encode(prediction_sentences, convert_to_tensor=True).cpu().numpy()
            sims = [np.max(cosine_similarity([embed], prediction_embeds)) for embed in expected_embeds]
            return np.mean(sims)
        except Exception as e:
            print(f"Error computing sentence-level similarity: {e}", flush=True)
            return 0.0


    def compute_paragraph_level_similarity(self, expected_text, prediction_text):
        """
        Compute paragraph-level similarity using embeddings for the full texts.
        Returns a similarity score between 0 and 1.
        """
        try:
            expected_embed = self.st_model.encode(expected_text, convert_to_tensor=True).cpu().numpy()
            prediction_embed = self.st_model.encode(prediction_text, convert_to_tensor=True).cpu().numpy()
            sim = cosine_similarity([expected_embed], [prediction_embed])[0][0]
            return sim
        except Exception as e:
            print(f"Error computing paragraph-level similarity: {e}", flush=True)
            return 0.0


    def evaluate_open_ended(self, expected, prediction):
        """
        Evaluate open-ended questions by first checking for an exact match.
        If the response exactly matches the expected answer, return 1.0.
        Otherwise, compute a weighted semantic similarity using:
            - Word-level similarity (weight 0.3)
            - Sentence-level similarity (weight 0.3)
            - Paragraph-level similarity (weight 0.4)
        Full points are given if the final semantic score is >= 0.9,
        0 points if below 0.4, and linear interpolation is used between.
        """
        try:
            # Ensure inputs are strings (not lists)
            if isinstance(expected, list):
                expected = " ".join(expected)
            if isinstance(prediction, list):
                prediction = " ".join(prediction)

            if expected.strip().lower() == prediction.strip().lower():
                return 1.0

            word_sim = self.compute_word_level_similarity(expected, prediction)
            sentence_sim = self.compute_sentence_level_similarity(expected, prediction)
            paragraph_sim = self.compute_paragraph_level_similarity(expected, prediction)

            # Weights that sum to 1
            w_word = 0.3
            w_sentence = 0.3
            w_paragraph = 0.4
            semantic_score = w_word * word_sim + w_sentence * sentence_sim + w_paragraph * paragraph_sim

            if semantic_score >= 0.9:
                return 1.0
            elif semantic_score < 0.4:
                return 0.0
            else:
                return (semantic_score - 0.4) / 0.5
        except Exception as e:
            print(f"Error evaluating open-ended question: {e}", flush=True)
            return 0.0

        
    def evaluate_open_ended_metrics(self, expected, prediction):
        """
        Calculate BLEU, ROUGE, and METEOR scores for the given expected and predicted answers.
        Returns a dictionary with the scores.
        """
        try:
            smoothing_function = SmoothingFunction().method1
            expected_tokens = nltk.word_tokenize(expected) if isinstance(expected, str) else expected
            predicted_tokens = nltk.word_tokenize(prediction) if isinstance(prediction, str) else prediction

            bleu = sentence_bleu([expected_tokens], predicted_tokens, smoothing_function=smoothing_function)
            meteor = meteor_score([' '.join(expected_tokens)], ' '.join(predicted_tokens))

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(expected, prediction)
            rouge_avg = (rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure) / 2.0
            return {"bleu": bleu, "meteor": meteor, "rouge": rouge_avg}
        except Exception as e:
            print(f"Error evaluating open-ended metrics: {e}", flush=True)
            return {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}



    def participant_model(self, prompt, qa_type=None):
        """
        Uses the participant's loaded model to generate a response based on the given prompt.
        If a Hugging Face model is detected, it will use a text-generation pipeline.
        If a PyTorch or script-based model is detected, it assumes a `generate()` method exists.
        """
        if not self.model:
            print("No participant model loaded. Returning default response.", flush=True)
            return "NO LLM IMPLEMENTED"

        try:
            # If using a Hugging Face model
            if isinstance(self.pipeline, TextGenerationPipeline):
                response = self.pipeline(prompt, max_length=self.max_length, do_sample=True)[0]['generated_text']
            
            # If using a PyTorch model with a `generate` method
            elif hasattr(self.model, "generate"):
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                output_ids = self.model.generate(input_ids, max_length=self.max_length)
                response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # If using a script-based model with a callable function
            elif callable(self.model):
                response = self.model(prompt)

            else:
                print("Unknown model type. Returning default response.", flush=True)
                response = "NO LLM IMPLEMENTED"
        
        except Exception as e:
            print(f"Error during inference: {e}", flush=True)
            response = "ERROR DURING INFERENCE"

        # Post-process to extract clean answer based on QA type
        if isinstance(response, str):
            response_clean = response.strip().lower()

            if qa_type == "true_false":
                if "true" in response_clean:
                    response = "true"
                elif "false" in response_clean:
                    response = "false"

            elif qa_type == "multiple_choice":
                # Extract single letter A–D
                match = re.search(r"\b[a-d]\b", response_clean)
                response = match.group() if match else response_clean

            elif qa_type == "list":
                # Return comma-separated list or split lines
                response = ", ".join(re.findall(r"[a-zA-Z0-9\s\-]+", response_clean))

            elif qa_type in {"short", "multi_hop", "short_inverse", "multi_hop_inverse"}:
                # For open-ended questions, return the trimmed response directly
                response = response_clean

        return response


    def evaluate_true_false_questions(self):
        """
        Evaluate all True/False questions and compute the overall F1 score.
        Returns a dictionary containing the average score and a mapping of paragraph_id to individual QA scores.
        """
        try:
            tf_data = self.sampled_qa_pairs.get("true_false", [])
            if tf_data is None:
                print("No True/False data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("tf_template.prompt")
            results = {}
            scores = []
            for qa in tf_data:
                try:
                    prompt = self.generate_prompt(template, qa, "true_false")
                    response = self.participant_model(prompt, qa_type="true_false")  # or "multiple_choice", etc.
                    expected = qa.get("answer", "").strip().lower()
                    predicted = response.strip().lower()
                    score = self.evaluate_true_false(expected, predicted)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": qa.get("question", ""),
                        "expected": expected,
                        "predicted": predicted,
                        "score": score
                    }
                    scores.append(score)
                except Exception as e:
                    print(f"Error processing True/False QA: {e}", flush=True)
            overall_f1 = sum(scores) / len(scores) if scores else 0.0
            print(f"Overall True/False F1 Score: {overall_f1:.2f}", flush=True)
            return {"average": overall_f1, "scores": results}
        except Exception as e:
            print(f"Error evaluating True/False questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}


    def evaluate_multiple_choice_questions(self):
        """
        Evaluate all Multiple Choice questions and compute the overall F1 score.
        Returns a dictionary containing the average score and a mapping of paragraph_id to individual QA scores.
        """
        try:
            mc_data = self.sampled_qa_pairs.get("multiple_choice", [])  # Multiple Choice
            if mc_data is None:
                print("No Multiple Choice data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("MC_template.prompt")
            results = {}
            scores = []
            for qa in mc_data:
                try:
                    prompt = self.generate_prompt(template, qa, "multiple_choice")
                    response = self.participant_model(prompt, qa_type="multiple_choice") 
                    expected_value = qa.get("correct_answer", "").strip().lower()
                    options = qa.get("options", [])
                    letter_map = {chr(65 + i): opt.strip().lower() for i, opt in enumerate(options)}
                    value_to_letter = {v: k for k, v in letter_map.items()}
                    expected_letter = value_to_letter.get(expected_value)

                    # Compare to model's predicted letter
                    predicted = response.strip().upper()
                    score = 1.0 if predicted == expected_letter else 0.0

                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": qa.get("question", ""),
                        "expected": expected_letter,
                        "predicted": predicted,
                        "score": score
                    }

                    scores.append(score)
                except Exception as inner_e:
                    print(f"Error processing Multiple Choice QA: {inner_e}", flush=True)
            overall_f1 = sum(scores) / len(scores) if scores else 0.0
            print(f"Overall Multiple Choice F1 Score: {overall_f1:.2f}", flush=True)
            return {"average": overall_f1, "scores": results}
        except Exception as e:
            print(f"Error evaluating Multiple Choice questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}


    def evaluate_list_questions(self):
        """
        Evaluate all List questions using the F1 score based on letter options (e.g., A, B, C).
        """
        try:
            list_data = self.sampled_qa_pairs.get("list", [])
            if list_data is None:
                print("No List data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("list_template.prompt")
            results = {}
            scores = []
            for qa in list_data:
                try:
                    prompt = self.generate_prompt(template, qa, "list")
                    response = self.participant_model(prompt, qa_type="list")
                    
                    options = qa.get("options", [])
                    # Build letter → option and reverse
                    letter_to_option = {chr(65 + i): opt.strip().lower() for i, opt in enumerate(options)}
                    option_to_letter = {v: k for k, v in letter_to_option.items()}
                    
                    expected_values = [ans.strip().lower() for ans in qa.get("answer", [])]
                    expected_letters = [option_to_letter.get(v) for v in expected_values if option_to_letter.get(v)]
                    
                    # Clean and deduplicate predictions
                    predicted_letters = list(set(
                        [x.strip().upper() for x in response.split(",") if x.strip().upper() in letter_to_option]
                    ))

                    # Clean and deduplicate expected answers
                    expected_values = [ans.strip().lower() for ans in qa.get("answer", [])]
                    expected_letters = list(set(
                        [option_to_letter.get(v) for v in expected_values if option_to_letter.get(v)]
                    ))
                    
                    predicted_letters.sort()
                    expected_letters.sort()

                    _, _, f1 = self.compute_f1_score(expected_letters, predicted_letters)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")

                    results[para_id] = {
                        "question": qa.get("question", ""),
                        "expected": expected_letters,
                        "predicted": predicted_letters,
                        "score": f1
                    }
                    scores.append(f1)
                except Exception as inner_e:
                    print(f"Error processing List QA: {inner_e}", flush=True)
            overall_f1 = sum(scores) / len(scores) if scores else 0.0
            print(f"Overall List Question F1 Score: {overall_f1:.2f}", flush=True)
            return {"average": overall_f1, "scores": results}
        except Exception as e:
            print(f"Error evaluating List questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}



    def evaluate_short_questions(self):
        """
        Evaluate all Short Answer questions using semantic similarity metrics.
        Returns a dictionary containing the average F1 score and a mapping (by paragraph_id) of individual QA scores.
        """
        try:
            short_data = self.sampled_qa_pairs.get("short", [])         # Short Answer
            if short_data is None:
                print("No Short Answer data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("short_template.prompt")
            results = {}
            scores = []
            for qa in short_data:
                try:
                    prompt = self.generate_prompt(template, qa, "short")
                    response = self.participant_model(prompt, qa_type="short") 
                    expected = qa.get("answer", "")
                    f1_score = self.evaluate_open_ended(expected, response)
                    metrics = self.evaluate_open_ended_metrics(expected, response)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": qa.get("question", ""),
                        "expected": expected,
                        "predicted": response,
                        "f1_score": f1_score,
                        "metrics": metrics
                    }
                    scores.append(f1_score)
                except Exception as inner_e:
                    print(f"Error processing Short Answer QA: {inner_e}", flush=True)
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Short Answer F1 Score: {avg:.2f}", flush=True)
            return {"average": avg, "scores": results}
        except Exception as e:
            print(f"Error evaluating Short Answer questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}

    
    def evaluate_short_inverse_questions(self):
        """
        Evaluate Short Inverse questions by comparing the LLM's response to the provided incorrect explanation.
        Returns a dictionary containing the average F1 score and a mapping (by paragraph_id) of individual QA scores.
        """
        try:
            short_inverse_data = self.sampled_qa_pairs.get("short_inverse", [])  # Short Inverse
            if short_inverse_data is None:
                print("No Short Inverse data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("short_inverse_template.prompt")
            results = {}
            scores = []
            for qa in short_inverse_data:
                try:
                    prompt = self.generate_prompt(template, qa, "short_inverse")
                    response = self.participant_model(prompt, qa_type="short_inverse") 
                    # print("Short Inverse Response:", response, flush=True)
                    # Use the provided incorrect explanation as the expected text.
                    expected = qa.get("incorrect_explanation", "")
                    f1_score = self.evaluate_open_ended(expected, response)
                    metrics = self.evaluate_open_ended_metrics(expected, response)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": qa.get("question", ""),
                        "expected": expected,
                        "predicted": response,
                        "f1_score": f1_score,
                        "metrics": metrics
                    }
                    scores.append(f1_score)
                except Exception as inner_e:
                    print(f"Error processing Short Inverse QA: {inner_e}", flush=True)
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Short Inverse F1 Score: {avg:.2f}", flush=True)
            return {"average": avg, "scores": results}
        except Exception as e:
            print(f"Error evaluating Short Inverse questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}


    def evaluate_multi_hop_questions(self):
        """
        Evaluate all Multi-hop questions using semantic similarity metrics.
        Returns a dictionary containing the average F1 score and a mapping (by paragraph_id)
        of individual QA scores.
        """
        try:
            mh_data = self.sampled_qa_pairs.get("multi_hop", [])       # Multi-hop
            if mh_data is None:
                print("No Multi-hop data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("multi_hop_template.prompt")
            results = {}
            scores = []
            for qa in mh_data:
                try:
                    prompt = self.generate_prompt(template, qa, "multi_hop")
                    response = self.participant_model(prompt, qa_type="multi_hop") 
                    expected = qa.get("answer", "")
                    f1_score = self.evaluate_open_ended(expected, response)
                    metrics = self.evaluate_open_ended_metrics(expected, response)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": qa.get("question", ""),
                        "expected": expected,
                        "predicted": response,
                        "f1_score": f1_score,
                        "metrics": metrics
                    }
                    scores.append(f1_score)
                except Exception as inner_e:
                    print(f"Error processing Multi-hop QA: {inner_e}", flush=True)
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Multi-hop F1 Score: {avg:.2f}", flush=True)
            return {"average": avg, "scores": results}
        except Exception as e:
            print(f"Error evaluating Multi-hop questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}


    def evaluate_multi_hop_inverse_questions(self):
        """
        Evaluate all Multi-hop Inverse questions by comparing the LLM's response with the provided
        incorrect reasoning step. Returns a dictionary containing the average F1 score and a mapping
        (by paragraph_id) of individual QA scores.
        """
        try:
            mh_inverse_data = self.sampled_qa_pairs.get("multi_hop_inverse", [])  # Multi-hop Inverse
            if mh_inverse_data is None:
                print("No Multi-hop Inverse data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("multi_hop_inverse_template.prompt")
            results = {}
            scores = []
            for qa in mh_inverse_data:
                try:
                    prompt = self.generate_prompt(template, qa, "multi_hop_inverse")
                    response = self.participant_model(prompt, qa_type="multi_hop_inverse") 
                    # print("Multi-hop Inverse Response:", response, flush=True)
                    # Use the provided incorrect reasoning step as the expected text.
                    expected = qa.get("incorrect_reasoning_step", "")
                    f1_score = self.evaluate_open_ended(expected, response)
                    metrics = self.evaluate_open_ended_metrics(expected, response)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": qa.get("question", ""),
                        "expected": expected,
                        "predicted": response,
                        "f1_score": f1_score,
                        "metrics": metrics
                    }
                    scores.append(f1_score)
                except Exception as inner_e:
                    print(f"Error processing Multi-hop Inverse QA: {inner_e}", flush=True)
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Multi-hop Inverse F1 Score: {avg:.2f}", flush=True)
            return {"average": avg, "scores": results}
        except Exception as e:
            print(f"Error evaluating Multi-hop Inverse questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}


    def run_all_evaluations(self):
        """
        Run evaluations for all QA types and save the overall results to a JSON file.
        """
        try:
            overall_results = {}
            overall_results["true_false"] = self.evaluate_true_false_questions()
            overall_results["multiple_choice"] = self.evaluate_multiple_choice_questions()
            overall_results["list"] = self.evaluate_list_questions()
            overall_results["short"] = self.evaluate_short_questions()
            overall_results["multi_hop"] = self.evaluate_multi_hop_questions()
            overall_results["short_inverse"] = self.evaluate_short_inverse_questions()
            overall_results["multi_hop_inverse"] = self.evaluate_multi_hop_inverse_questions()
            
            overall_json = json.dumps(overall_results, indent=2)
            # print("Overall Evaluation Results:", overall_json, flush=True)
            
            output_file = os.environ.get("RESULTS_PATH", "/tmp/overall_evaluation_results.json")
            with open(output_file, "w") as f:
                json.dump(overall_results, f, indent=2)
            print(f"Saved overall evaluation results to {output_file}", flush=True)
        except Exception as e:
            print(f"Error running overall evaluations: {e}", flush=True)

def parse_args():
    parser = argparse.ArgumentParser(description="ClinIQLink Evaluation Script")
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
    evaluator = ClinIQLinkSampleDatasetSubmit(run_mode=args.mode, max_length=args.max_length, sample_sizes=sample_sizes)
    evaluator.run_all_evaluations()