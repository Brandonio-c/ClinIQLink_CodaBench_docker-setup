import json
import os
import numpy as np
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


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

class ClinIQLinkSampleDatasetEvaluate:
    def __init__(self, run_mode="container", results_dir="submission_output"):
        self.run_mode = run_mode.lower()
        self.results_dir = results_dir
        # Base directories and setup depending on run mode
        if run_mode == "container":
            print("Running in container mode.", flush=True)
            self.base_dir = "/app"
            self.st_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=os.path.join(self.base_dir, ".cache"),
                trust_remote_code=True
            )
        else:
            print("Running in local mode.", flush=True)
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
            self.st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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
        # Ensure wordnet is available for METEOR scoring
        try:
            nltk.data.find("corpora/wordnet")
            print("NLTK 'wordnet' resource found.", flush=True)
        except LookupError:
            print("Downloading NLTK 'wordnet'...", flush=True)
            nltk.download("wordnet", download_dir=self.nltk_data_dir)

                
        # -------------------------------------------------------------------
        # Where each task’s generation file lives (relative to --results_dir)
        # -------------------------------------------------------------------
        self.QA_FILES = {
            "true_false":        "true_false.json",
            "multiple_choice":   "multiple_choice.json",
            "list":              "list.json",
            "short":             "short.json",
            "multi_hop":         "multi_hop.json",
            "short_inverse":     "short_inverse.json",
            "multi_hop_inverse": "multi_hop_inverse.json",
        }

        self.output_data = self._load_outputs()

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
        

    def _load_outputs(self):
        data = {}
        for k, fname in self.QA_FILES.items():
            fpath = os.path.join(self.results_dir, fname)
            if not os.path.exists(fpath):
                print(f"[WARN] {fpath} not found – skipping {k}", flush=True)
                data[k] = None
                continue
            data[k] = self.load_json(fpath)
        return data
    
    def _average(self, lst):
        """Safe mean – returns 0.0 for empty lists."""
        return sum(lst) / len(lst) if lst else 0.0

    def _summarise_metrics(self, metric_lists):
        """
        metric_lists = {"f1": [...], "bleu": [...], "meteor": [...], "rouge": [...]}
        Returns {"avg_f1":x, "avg_bleu":y, ...}
        """
        return {
            "avg_f1":     self._average(metric_lists["f1"]),
            "avg_bleu":   self._average(metric_lists.get("bleu", [])),
            "avg_meteor": self._average(metric_lists.get("meteor", [])),
            "avg_rouge":  self._average(metric_lists.get("rouge", [])),
        }
    
    def _to_text(self, x):
        """
        Recursively unwrap HF pipeline outputs (lists / dicts) to a clean string.
        """
        if isinstance(x, (list, tuple)):
            # flatten nested lists of dicts
            return " ".join(self._to_text(i) for i in x).strip()

        if isinstance(x, dict):
            # HuggingFace pipelines use 'generated_text' (text-gen) or 'text' (chat)
            return str(x.get("generated_text") or x.get("text") or "").strip()

        return str(x).strip()


    def compute_classification_metrics(self, y_true, y_pred, average="binary", labels=None):
        try:
            precision = precision_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
            recall = recall_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
            return precision, recall, f1
        except Exception as e:
            print(f"Error computing classification metrics: {e}", flush=True)
            return 0.0, 0.0, 0.0

    
    def compute_overlap_f1(self, true_list, pred_list):
        """
        Compute precision, recall, and F1 score for list-type answers.
        """
        try:
            true_set = set(map(str, true_list))
            pred_set = set(map(str, pred_list))
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

            _, _, f1 = self.compute_overlap_f1(expected_letters, pred_letters)
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
        Evaluate open-ended questions using semantic similarity:
        - 1.0 if exact match
        - Weighted similarity of:
            - word-level (0.3)
            - sentence-level (0.3)
            - paragraph-level (0.4)
        - Score thresholds:
            - >= 0.9 → 1.0
            - <= 0.4 → 0.0
            - Linear interpolation in-between
        """
        try:
            # Convert lists to strings, remove whitespace
            if isinstance(expected, list):
                expected = " ".join(expected)
            if isinstance(prediction, list):
                prediction = " ".join(prediction)

            expected = expected.strip()
            prediction = prediction.strip()

            # Exact match
            if expected.lower() == prediction.lower():
                return 1.0

            # Compute semantic similarity at three levels
            w_word, w_sent, w_para = 0.3, 0.3, 0.4

            word_sim      = self.compute_word_level_similarity(expected, prediction)
            sentence_sim  = self.compute_sentence_level_similarity(expected, prediction)
            paragraph_sim = self.compute_paragraph_level_similarity(expected, prediction)

            semantic_score = w_word * word_sim + w_sent * sentence_sim + w_para * paragraph_sim

            # Map similarity to final score
            if semantic_score >= 0.9:
                return 1.0
            elif semantic_score <= 0.4:
                return 0.0
            else:
                return (semantic_score - 0.4) / (0.9 - 0.4)  # linear interpolation
        except Exception as e:
            print(f"Error evaluating open-ended question: {e}", flush=True)
            return 0.0

        
    def evaluate_open_ended_metrics(self, expected, prediction):
        """
        Calculate BLEU, ROUGE, and METEOR scores for the given expected and predicted answers.
        Returns a dictionary with the scores.
        NOTE: METEOR (NLTK >= 3.8) requires pre-tokenized input (lists of tokens).
        """
        try:
            # Canonicalize inputs to plain lowercase strings
            if isinstance(expected, list):
                expected = " ".join(expected)
            if isinstance(prediction, list):
                prediction = " ".join(prediction)

            expected = expected.strip().lower()
            prediction = prediction.strip().lower()

            # Tokenize
            ref_tokens = nltk.word_tokenize(expected)
            hyp_tokens = nltk.word_tokenize(prediction)

            # BLEU
            bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method1)

            # METEOR (tokens expected for both reference and hypothesis)
            meteor = meteor_score([ref_tokens], hyp_tokens)

            # ROUGE (works on strings)
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(expected, prediction)
            rouge_avg = (rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure) / 2.0

            return {"bleu": bleu, "meteor": meteor, "rouge": rouge_avg}

        except Exception as e:
            print(f"Error evaluating open-ended metrics: {e}", flush=True)
            return {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}



    def evaluate_true_false_questions(self):
        """
        Evaluate all True/False questions using precision, recall, and F1.
        Accepts only 'true' or 'false' predictions.
        Returns all metrics along with per-example scores.
        """
        try:
            blob = self.output_data.get("true_false")
            if not blob:
                print("No True/False output data found.", flush=True)
                return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}

            inputs = blob.get("inputs", [])
            predictions = blob.get("responses", [])

            if len(inputs) != len(predictions):
                print("Mismatch in number of inputs and predictions.", flush=True)
                return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}

            results = {}
            all_expected = []
            all_predicted = []

            for gold, pred in zip(inputs, predictions):
                predicted = self._to_text(pred).strip().lower()
                expected = str(gold.get("answer", "")).strip().lower()

                # Only accept valid binary classification values
                if predicted in {"true", "false"}:
                    all_expected.append(expected)
                    all_predicted.append(predicted)
                    score = 1.0 if expected == predicted else 0.0
                else:
                    all_expected.append(expected)
                    all_predicted.append("invalid")  # force a false match
                    score = 0.0

                para_id = gold.get("source", {}).get("paragraph_id", "unknown")
                results[para_id] = {
                    "question": gold.get("question", ""),
                    "expected": expected,
                    "predicted": predicted,
                    "score": score,
                    "source": gold.get("source", {})
                }

            # Compute precision, recall, and F1 using your existing method
            # Convert true/false to binary labels: true=1, false=0
            binary_map = {"true": 1, "false": 0}
            try:
                true_labels = [binary_map.get(e, 0) for e in all_expected]
                pred_labels = [binary_map.get(p, 0) for p in all_predicted]
            except Exception as label_error:
                print(f"Label conversion error: {label_error}", flush=True)
                return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": results}

            precision, recall, f1 = self.compute_classification_metrics(true_labels, pred_labels)
            avg_score = sum(results[pid]["score"] for pid in results) / len(results) if results else 0.0

            print(f"True/False Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}", flush=True)

            return {
                "average": avg_score,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "scores": results
            }

        except Exception as e:
            print(f"Error evaluating True/False questions: {e}", flush=True)
            return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}


    def evaluate_multiple_choice_questions(self):
        """
        Evaluate all Multiple Choice questions using precision, recall, and F1.
        Maps correct answer value to letter, compares against predicted letter.
        Returns average accuracy, precision, recall, and F1, plus per-QA scores.
        """
        try:
            blob = self.output_data.get("multiple_choice")
            if not blob:
                print("No Multiple Choice output data found.", flush=True)
                return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}

            inputs = blob.get("inputs", [])
            predictions = blob.get("responses", [])

            if len(inputs) != len(predictions):
                print("Mismatch in number of inputs and predictions.", flush=True)
                return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}

            results = {}
            all_expected = []
            all_predicted = []
            raw_scores = []

            for gold, pred in zip(inputs, predictions):
                predicted = self._to_text(pred).strip().upper()
                options = gold.get("options", [])
                expected_value = str(gold.get("correct_answer", "")).strip().lower()
                
                # Map option values to letters
                letter_map = {chr(65 + i): opt.strip().lower() for i, opt in enumerate(options)}
                value_to_letter = {v: k for k, v in letter_map.items()}
                expected_letter = value_to_letter.get(expected_value, None)

                score = 1.0 if predicted == expected_letter and predicted in letter_map else 0.0
                raw_scores.append(score)

                # F1 calculation inputs
                all_expected.append(expected_letter if expected_letter else "invalid")
                all_predicted.append(predicted if predicted in letter_map else "invalid")

                para_id = gold.get("source", {}).get("paragraph_id", "unknown")
                results[para_id] = {
                    "question": gold.get("question", ""),
                    "expected": expected_letter,
                    "predicted": predicted,
                    "score": score,
                    "source": gold.get("source", {})
                }

            # Compute macro precision/recall/f1 using label matching
            label_list = list(letter_map.keys()) + ["invalid"]
            precision, recall, f1 = self.compute_classification_metrics(
                all_expected, all_predicted,
                average="macro", labels=label_list
            )
            avg_score = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0

            print(f"Multiple Choice Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}", flush=True)

            return {
                "average": avg_score,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "scores": results
            }

        except Exception as e:
            print(f"Error evaluating Multiple Choice questions: {e}", flush=True)
            return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}



    def evaluate_list_questions(self):
        """
        Evaluate all List questions.
        - Extracts option letters (A, B, ...) from prediction using regex.
        - Maps expected values to their corresponding letters.
        - Computes precision, recall, and F1 score using predicted vs expected letters.
        - Returns per-item results and overall averages.
        """
        try:
            blob = self.output_data.get("list")
            if not blob:
                print("No List output data found.", flush=True)
                return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}

            inputs = blob.get("inputs", [])
            predictions = blob.get("responses", [])

            if len(inputs) != len(predictions):
                print("Mismatch in number of inputs and predictions.", flush=True)
                return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}

            results = {}
            precision_list = []
            recall_list = []
            f1_list = []
            global_tp, global_fp, global_fn = 0, 0, 0


            for gold, raw_pred in zip(inputs, predictions):
                try:
                    options = gold.get("options", [])
                    letter_to_option = {chr(65 + i): opt.strip().lower() for i, opt in enumerate(options)}
                    option_to_letter = {v: k for k, v in letter_to_option.items()}
                    valid_letters_set = set(letter_to_option.keys())

                    expected_values = [v.strip().lower() for v in gold.get("answer", [])]
                    expected_letters = sorted(
                        {option_to_letter[v] for v in expected_values if v in option_to_letter}
                    )

                    pred_text = self._to_text(raw_pred).strip().upper()
                    predicted_letters = []

                    if len(pred_text.split()) <= 10 and len(pred_text) <= 50:
                        import re
                        regex = r'\b[' + ''.join(valid_letters_set) + r']\b'
                        predicted_letters = sorted(set(re.findall(regex, pred_text)))

                    precision, recall, f1 = self.compute_overlap_f1(expected_letters, predicted_letters)

                    tp = len(set(expected_letters) & set(predicted_letters))
                    fp = len(set(predicted_letters) - set(expected_letters))
                    fn = len(set(expected_letters) - set(predicted_letters))
                    global_tp += tp
                    global_fp += fp
                    global_fn += fn

                    para_id = gold.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": gold.get("question", ""),
                        "expected": expected_letters,
                        "predicted": predicted_letters if predicted_letters else pred_text,
                        "score": f1,
                        "precision": precision,
                        "recall": recall,
                        "source": gold.get("source", {})
                    }
                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_list.append(f1)

                except Exception as inner_e:
                    print(f"Error processing List QA: {inner_e}", flush=True)

            avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0.0
            avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
            avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

            micro_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
            micro_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
            micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

            print(f"List Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, F1: {avg_f1:.2f}", flush=True)

            return {
                "average": avg_f1,
                "macro_precision": avg_precision,
                "macro_recall": avg_recall,
                "macro_f1_score": avg_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1_score": micro_f1,
                "scores": results
            }


        except Exception as e:
            print(f"Error evaluating List questions: {e}", flush=True)
            return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}


    def evaluate_short_questions(self):
        """
        Evaluate all Short Answer questions using semantic similarity metrics.
        Returns:
            - Average semantic similarity score as `semantic_match_score`
            - Per-item scores and metrics
        """
        try:
            blob = self.output_data.get("short")
            if not blob:
                print("No Short Answer output data found.", flush=True)
                return {"semantic_match_score": 0.0, "scores": {}}

            inputs = blob.get("inputs", [])
            predictions = blob.get("responses", [])

            if len(inputs) != len(predictions):
                print("Mismatch in number of inputs and predictions.", flush=True)
                return {"semantic_match_score": 0.0, "scores": {}}

            results = {}
            semantic_scores = []

            for gold, pred in zip(inputs, predictions):
                try:
                    predicted = self._to_text(pred).strip().lower()
                    question  = str(gold.get("question", "")).strip().lower()
                    expected  = str(gold.get("answer", "")).strip()

                    if predicted == question:
                        sim_score = 0.0
                        metrics = {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}
                    else:
                        sim_score = self.evaluate_open_ended(expected, predicted)
                        metrics = self.evaluate_open_ended_metrics(expected, predicted)

                    para_id = gold.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": gold.get("question", ""),
                        "expected": expected,
                        "predicted": predicted,
                        "semantic_match_score": sim_score,
                        "metrics": metrics,
                        "source": gold.get("source", {})
                    }
                    semantic_scores.append(sim_score)

                except Exception as inner_e:
                    print(f"Error processing Short Answer QA: {inner_e}", flush=True)

            avg_sim_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
            print(f"Average Short Answer Semantic Similarity Score: {avg_sim_score:.2f}", flush=True)

            return {
                "semantic_match_score": avg_sim_score,
                "scores": results
            }

        except Exception as e:
            print(f"Error evaluating Short Answer questions: {e}", flush=True)
            return {"semantic_match_score": 0.0, "scores": {}}
    
    def evaluate_short_inverse_questions(self):
        """
        Evaluate Short Inverse questions using semantic similarity.
        The model must identify what's wrong with a given incorrect explanation.
        If the response is identical to the question, returns 0.
        Returns:
            - Average semantic similarity score as 'semantic_match_score'
            - Per-item breakdowns with BLEU, METEOR, ROUGE
        """
        try:
            blob = self.output_data.get("short_inverse")
            if not blob:
                print("No Short Inverse output data found.", flush=True)
                return {"semantic_match_score": 0.0, "scores": {}}

            inputs = blob.get("inputs", [])
            predictions = blob.get("responses", [])

            if len(inputs) != len(predictions):
                print("Mismatch in number of inputs and predictions.", flush=True)
                return {"semantic_match_score": 0.0, "scores": {}}

            results = {}
            semantic_scores = []

            for gold, pred in zip(inputs, predictions):
                try:
                    predicted = self._to_text(pred).strip().lower()
                    question = str(gold.get("question", "")).strip().lower()
                    expected = str(gold.get("incorrect_explanation", "")).strip()

                    if predicted == question:
                        sim_score = 0.0
                        metrics = {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}
                    else:
                        sim_score = self.evaluate_open_ended(expected, predicted)
                        metrics = self.evaluate_open_ended_metrics(expected, predicted)

                    para_id = gold.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": gold.get("question", ""),
                        "expected": expected,
                        "predicted": predicted,
                        "semantic_match_score": sim_score,
                        "metrics": metrics,
                        "source": gold.get("source", {})
                    }
                    semantic_scores.append(sim_score)

                except Exception as inner_e:
                    print(f"Error evaluating Short Inverse QA: {inner_e}", flush=True)

            avg_sim_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
            print(f"Average Short Inverse Semantic Similarity Score: {avg_sim_score:.2f}", flush=True)

            return {
                "semantic_match_score": avg_sim_score,
                "scores": results
            }

        except Exception as e:
            print(f"Error evaluating Short Inverse questions: {e}", flush=True)
            return {"semantic_match_score": 0.0, "scores": {}}


    def evaluate_multi_hop_questions(self):
        """
        Evaluate Multi-hop questions using semantic similarity.
        - Uses weighted combination of word/sentence/paragraph level cosine similarity
        - Returns:
            - Average semantic similarity score (`semantic_match_score`)
            - Per-question breakdowns with BLEU, METEOR, ROUGE
        """
        try:
            blob = self.output_data.get("multi_hop")
            if not blob:
                print("No Multi-hop output data found.", flush=True)
                return {"semantic_match_score": 0.0, "scores": {}}

            inputs = blob.get("inputs", [])
            predictions = blob.get("responses", [])

            if len(inputs) != len(predictions):
                print("Mismatch in number of inputs and predictions.", flush=True)
                return {"semantic_match_score": 0.0, "scores": {}}

            results = {}
            semantic_scores = []

            for gold, pred in zip(inputs, predictions):
                try:
                    predicted = self._to_text(pred).strip().lower()
                    question = str(gold.get("question", "")).strip().lower()
                    expected = str(gold.get("answer", "")).strip()

                    if predicted == question:
                        sim_score = 0.0
                        metrics = {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}
                    else:
                        sim_score = self.evaluate_open_ended(expected, predicted)
                        metrics = self.evaluate_open_ended_metrics(expected, predicted)

                    para_id = gold.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": gold.get("question", ""),
                        "expected": expected,
                        "predicted": predicted,
                        "semantic_match_score": sim_score,
                        "metrics": metrics,
                        "source": gold.get("source", {})
                    }
                    semantic_scores.append(sim_score)

                except Exception as inner_e:
                    print(f"Error evaluating Multi-hop QA: {inner_e}", flush=True)

            avg_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
            print(f"Average Multi-hop Semantic Similarity Score: {avg_score:.2f}", flush=True)

            return {
                "semantic_match_score": avg_score,
                "scores": results
            }

        except Exception as e:
            print(f"Error evaluating Multi-hop questions: {e}", flush=True)
            return {"semantic_match_score": 0.0, "scores": {}}



    def evaluate_multi_hop_inverse_questions(self):
        """
        Evaluate Multi-hop Inverse questions using semantic similarity metrics.
        - Uses weighted word/sentence/paragraph embedding similarity
        - Returns:
            - Average semantic similarity as `semantic_match_score`
            - Per-question breakdowns with BLEU, METEOR, and ROUGE
        """
        try:
            blob = self.output_data.get("multi_hop_inverse")
            if not blob:
                print("No Multi-hop Inverse output data found.", flush=True)
                return {"semantic_match_score": 0.0, "scores": {}}

            inputs = blob.get("inputs", [])
            predictions = blob.get("responses", [])

            if len(inputs) != len(predictions):
                print("Mismatch in number of inputs and predictions.", flush=True)
                return {"semantic_match_score": 0.0, "scores": {}}

            results = {}
            scores = []

            for gold, pred in zip(inputs, predictions):
                try:
                    predicted = self._to_text(pred).strip().lower()
                    question = str(gold.get("question", "")).strip().lower()
                    expected = str(gold.get("incorrect_reasoning_step", "")).strip()

                    if predicted == question:
                        sim_score = 0.0
                        metrics = {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}
                    else:
                        sim_score = self.evaluate_open_ended(expected, predicted)
                        metrics = self.evaluate_open_ended_metrics(expected, predicted)

                    para_id = gold.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": gold.get("question", ""),
                        "expected": expected,
                        "predicted": predicted,
                        "semantic_match_score": sim_score,
                        "metrics": metrics,
                        "source": gold.get("source", {})
                    }
                    scores.append(sim_score)

                except Exception as inner_e:
                    print(f"Error evaluating Multi-hop Inverse QA: {inner_e}", flush=True)

            avg_score = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Multi-hop Inverse Semantic Similarity Score: {avg_score:.2f}", flush=True)

            return {
                "semantic_match_score": avg_score,
                "scores": results
            }

        except Exception as e:
            print(f"Error evaluating Multi-hop Inverse questions: {e}", flush=True)
            return {"semantic_match_score": 0.0, "scores": {}}



    def run_all_evaluations(self):
        """
        Run evaluations for all QA types and save results to five category files and one overall summary file.
        """
        try:
            output_dir = os.path.join(self.base_dir, "evaluation_output")
            os.makedirs(output_dir, exist_ok=True)

            overall_results = {}

            # Evaluate each QA type individually
            tf_results = self.evaluate_true_false_questions()
            mc_results = self.evaluate_multiple_choice_questions()
            list_results = self.evaluate_list_questions()
            short_results = self.evaluate_short_questions()
            short_inverse_results = self.evaluate_short_inverse_questions()
            multi_hop_results = self.evaluate_multi_hop_questions()
            multi_hop_inverse_results = self.evaluate_multi_hop_inverse_questions()

            # Organize and save grouped results
            grouped_outputs = {
                "true_false_results.json": {
                    "true_false": tf_results
                },
                "multiple_choice_results.json": {
                    "multiple_choice": mc_results
                },
                "list_results.json": {
                    "list": list_results
                },
                "short_results.json": {
                    "short": short_results,
                },
                "multi_hop_results.json": {
                    "multi_hop": multi_hop_results,
                },
                "short_inverse_results.json": {
                    "short_inverse": short_inverse_results
                },
                "multi_hop_inverse_results.json": {
                    "multi_hop_inverse": multi_hop_inverse_results
                }
            }

            # Save each category file
            for filename, data in grouped_outputs.items():
                file_path = os.path.join(output_dir, filename)
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"Saved {filename}", flush=True)

            # Merge everything for overall results
            overall_results["true_false"]        = tf_results
            overall_results["multiple_choice"]   = mc_results
            overall_results["list"]              = list_results
            overall_results["short"]             = short_results
            overall_results["short_inverse"]     = short_inverse_results
            overall_results["multi_hop"]         = multi_hop_results
            overall_results["multi_hop_inverse"] = multi_hop_inverse_results


            # Save overall summary
            with open(os.path.join(output_dir, "overall_evaluation_results.json"), "w") as f:
                json.dump(overall_results, f, indent=4)
            print("Saved overall_evaluation_results.json", flush=True)

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

    parser.add_argument("--results_dir",
                    default="submission_output",
                    help="Folder that already contains the seven *.json files")


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluator = ClinIQLinkSampleDatasetEvaluate(run_mode=args.mode, results_dir=args.results_dir)
    evaluator.run_all_evaluations()