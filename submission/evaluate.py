"""
================================================================================
    ClinIQLink - Comprehensive Evaluation Script
================================================================================
This module evaluates generated answers for **seven distinct clinical QA tasks**
and produces both per-item results and aggregated analytics (JSON + plots).

--------------------------------------------
1.  Supported QA task files  (in `results_dir`)
--------------------------------------------
│  true_false.json          → yes / no judgement
│  multiple_choice.json     → single-best option (A, B, …)
│  list.json                → multi-select option list
│  short.json               → short free-text answers
│  multi_hop.json           → multi-sentence reasoning answers
│  short_inverse.json       → explain *why* a short answer is wrong
│  multi_hop_inverse.json   → find + explain an incorrect reasoning **step**
└───────────────────────────────────────────────────────────────────────────────

Each file must contain two top-level keys:  
* `"inputs"`- list of ground-truth items  
* `"responses"` - list of model outputs (same order & length)

--------------------------------------------
2.  Categorical tasks and their metrics
--------------------------------------------
• **True / False** - *Accuracy*  
    `accuracy = #correct / N`

• **Multiple-Choice** - *Accuracy* (after normalising punctuation/case)

• **List (multi-select)** - token-level  **Macro & Micro F1**  
    - Macro: mean F1 across questions  
    - Micro: pool all TP/FP/FN, then F1

--------------------------------------------
3.  Open-ended tasks: custom semantic similarity
--------------------------------------------
Returned key: `"semantic_match_score"`  ∈ [0, 1]

Step-by-step computation
~~~~~~~~~~~~~~~~~~~~~~~~
1. **Pre-cleaning** - strip stop-words & punctuation for all similarity layers.  
2. **Exact-match guard** - if the raw (uncleaned) strings match ⇒ score 1.0.  
3. **Three-layer cosine similarity**
   * **Word-level** (weight 0.4)  
        • WordPiece-tokenise both texts.  
        • Look up an *IDF* for every token, built **once** from all reference
        answers:  
        `idf(tok) = log((N+1)/(df+1)) + 1`  
        • Embed each token with SBERT and perform *greedy max alignment*  
        to obtain an IDF-weighted precision/recall → F1.
   * **Sentence-level** (weight 0.4)  
        • Encode each full text with SBERT (`all-MiniLM-L6-v2`)  
        and take the CLS vector.  
        • `cosine(exp_vec, pred_vec)`.
   * **Paragraph-level** (weight 0.2)  
     • Same as sentence-level but on the **raw** strings.  
        • Adjust for SBERT's baseline bias: a background sample of 100 random,
       *unrelated* reference pairs is averaged once at start-up to yield
        `para_baseline` (≈ 0.25-0.35).  
        The raw cosine is mapped:  
        `adj = (raw - baseline) / (1 - baseline)`  → clipped to ⩾0.
4. **Weighted sum** → `semantic_score`.  
5. **Bias clamp** - subtract the SBERT “unrelated” floor (~0.25).  
6. **Cap** - if `semantic_score ≥ 0.95` set to **1.0**.

*For **multi-hop-inverse** items the score is additionally penalised
by how far the predicted “incorrect step #” is from the true one
(distance 0→x1, 1→x0.7, 2→x0.3, ≥3 halves each further step).*


--------------------------------------------
4.  N-gram reference metrics (open-ended)
--------------------------------------------
For every open-ended answer we also compute:
* **BLEU-1…4** (smooth-1) - reported as the cumulative BLEU (weights 0.25⁴).  
* **METEOR** (NLTK ≥3.8, stemming + synonym matching via WordNet).  
* **ROUGE-1/L** average F-measure.

Aggregations:
* Per-task **avg_bleu / avg_meteor / avg_rouge**  
* Global average across the four open-ended tasks.

--------------------------------------------
5.  Visual analytics
--------------------------------------------
All plots are saved as **SVG** under `<evaluation_output>/plots/`:

* Per-metric box, jitter & histogram (semantic, BLEU, METEOR, ROUGE).  
* A combined dashboard (`all_metrics_dashboard.svg`) with one row
    per metric x three columns (box | jitter | hist).

--------------------------------------------
6.  Command-line usage
--------------------------------------------
python evaluate.py \
        --mode container \          # or 'local'
        --results_dir submission_output \
        --bin_width 0.05            # histogram bin width

Dependencies

Python ≥3.10, numpy <2, scikit-learn, sentence-transformers, torch,
nltk, rouge-score, matplotlib (SVG backend), scipy, transformers,
plus the usual tokenizer/sentencepiece extras.

"""

import json
import os
import re
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
plt.rcParams['savefig.format'] = 'svg'
import scipy.stats as stats
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer


# Explicitly set HuggingFace & Torch cache paths for consistency and safety inside container
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/transformers"
os.environ["TORCH_HOME"] = "/app/.cache/torch"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import nltk

from collections import Counter
from transformers import AutoTokenizer
from nltk.corpus import stopwords
import string


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
    def __init__(self, run_mode="container", results_dir="submission_output", bin_width=0.05):
        self.run_mode = run_mode.lower()
        self.results_dir = results_dir
        self.bin_width  = bin_width
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

        # — stop-word / punctuation setup —
        try:
            self._STOP_WORDS = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            self._STOP_WORDS = set(stopwords.words("english"))
        self._PUNCT = set(string.punctuation)

        all_refs = []
        for blob in self.output_data.values():
            if blob and "inputs" in blob:
                for inp in blob["inputs"]:
                    ans = inp.get("answer", "")
                    if isinstance(ans, list):
                        ans = " ".join(ans)
                    if ans:
                        all_refs.append(ans)
        
        
        # ── build IDF from SBERT’s own WordPiece tokenizer ──
        # collect all the raw reference texts first
        refs = all_refs.copy()

        # initialize tokenizer & empty DF counts
        self.tokenizer   = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self._doc_freq   = Counter()
        self._total_docs = 0

        # count document frequency per WordPiece token
        for ref in refs:
            toks = set(self.tokenizer.tokenize(ref))
            self._doc_freq.update(toks)
            self._total_docs += 1

        # precompute IDF: idf = log((N+1)/(df+1)) + 1
        self._idf_scores = {
            tok: np.log((self._total_docs + 1) / (df + 1)) + 1.0
            for tok, df in self._doc_freq.items()
        }

        # ── estimate SBERT “unrelated” paragraph baseline ──
        if len(refs) >= 100:
            sample = random.sample(refs, 100)
            sims = []
            for a, b in zip(sample[:50], sample[50:]):
                ea = self.st_model.encode(a, convert_to_tensor=True).cpu().numpy()
                eb = self.st_model.encode(b, convert_to_tensor=True).cpu().numpy()
                sims.append(cosine_similarity([ea], [eb])[0,0])
            self.para_baseline = float(sum(sims) / len(sims))
        else:
            # fallback if too few refs
            self.para_baseline = 0.3

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
    
    def _safe_binary_metrics(self, y_true, y_pred):
        """
        Precision/recall/F1 that behave sensibly when only one class is present.
        If the two vectors are identical -> all metrics = 1.
        If they differ -> all metrics = 0.
        Otherwise fall back to sklearn.
        """
        if y_true == y_pred:
            # exact match, can be all positives or all negatives
            return 1.0, 1.0, 1.0
        if set(y_true) == {0} and set(y_pred) == {0}:
            # both all-negative but not identical (length mismatch case)
            return 0.0, 0.0, 0.0

        # normal two-class case
        return self.compute_classification_metrics(
            y_true, y_pred, average="binary", labels=[0,1]
        )
    
    def _extract_numbers(self, text: str):
        """Find all numeric literals in a string."""
        return re.findall(r"\d+(?:\.\d+)?", text)

    def _has_negation(self, text: str):
        """Detect simple negation words."""
        return any(neg in text for neg in [" not ", " without ", " instead "])

    def _compute_word_overlap(self, ref: str, hyp: str):
        """
        Simple word-level F1: 
        2·|ref∩hyp| / (2·|ref∩hyp| + |hyp−ref| + |ref−hyp|)
        """
        ref_set = set(ref.split())
        hyp_set = set(hyp.split())
        tp = len(ref_set & hyp_set)
        fp = len(hyp_set - ref_set)
        fn = len(ref_set - hyp_set)
        if tp == 0:
            return 0.0
        return 2 * tp / (2*tp + fp + fn)

    def _content_only(self, text: str) -> str:
        toks = [t for t in text.split()
                if t.lower() not in self._STOP_WORDS
                and not all(ch in self._PUNCT for ch in t)]
        return " ".join(toks)
    

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
        IDF-weighted greedy word alignment using precomputed WordPiece IDF.
        """
        try:
            # WordPiece tokenization (only alphanumeric pieces)
            exp_tokens = [tok for tok in self.tokenizer.tokenize(expected_text) if tok.isalnum()]
            pred_tokens = [tok for tok in self.tokenizer.tokenize(prediction_text) if tok.isalnum()]
            if not exp_tokens or not pred_tokens:
                return 0.0

            # Lookup IDF for each token (default to 1.0 if unseen)
            exp_weights = np.array([self._idf_scores.get(tok, 1.0) for tok in exp_tokens])
            pred_weights = np.array([self._idf_scores.get(tok, 1.0) for tok in pred_tokens])

            # Embed all tokens in one batch
            exp_emb = self.st_model.encode(exp_tokens, convert_to_tensor=True).cpu().numpy()
            pred_emb = self.st_model.encode(pred_tokens, convert_to_tensor=True).cpu().numpy()

            # Compute pairwise cosine and do greedy max‑matching
            sims = cosine_similarity(pred_emb, exp_emb)  # shape (|pred|, |exp|)
            best_for_pred = sims.max(axis=1)  # precision analogue
            best_for_exp  = sims.max(axis=0)  # recall analogue

            # Weighted precision & recall → F1
            precision = (best_for_pred * pred_weights).sum() / pred_weights.sum()
            recall    = (best_for_exp  * exp_weights).sum()  / exp_weights.sum()
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)

        except Exception as e:
            print(f"[word-sim] {e}", flush=True)
            return 0.0


    def compute_sentence_level_similarity(self, exp: str, pred: str) -> float:
        """
        Sentence‐level similarity using single‐vector (CLS) embeddings.
        """
        try:
            # encode each full sentence as one vector
            exp_vec  = self.st_model.encode([exp],  convert_to_tensor=True).cpu().numpy()
            pred_vec = self.st_model.encode([pred], convert_to_tensor=True).cpu().numpy()
            # single cosine between the two vectors
            return float(cosine_similarity(exp_vec, pred_vec)[0][0])
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
            raw = cosine_similarity([expected_embed], [prediction_embed])[0][0]
            adj = (raw - self.para_baseline) / (1 - self.para_baseline)
            return max(0.0, adj)
        except Exception as e:
            print(f"Error computing paragraph-level similarity: {e}", flush=True)
            return 0.0

    
    def evaluate_open_ended(self, expected, prediction):
        """
        Evaluate open-ended questions using semantic similarity:
            - exact match → 1.0
            - IDF-/stop-word-cleaned similarities:
                word (0.4), sentence (0.4), paragraph (0.2)
            - clamp out the ~0.25 SBERT bias
            - cap at 1.0
        """
        try:
            # prepare raw strings
            if isinstance(expected, list):
                expected = " ".join(expected)
            if isinstance(prediction, list):
                prediction = " ".join(prediction)
            expected_raw   = expected.strip()
            prediction_raw = prediction.strip()

            # strip stop-words + punctuation for all sims
            expected = self._content_only(expected_raw)
            prediction = self._content_only(prediction_raw)

            # exact‐match shortcut
            if expected_raw.lower() == prediction_raw.lower():
                return 1.0

            # compute sims with new weights
            w_word, w_sent, w_para = 0.4, 0.4, 0.2
            word_sim      = self.compute_word_level_similarity(expected, prediction)
            sentence_sim  = self.compute_sentence_level_similarity(expected, prediction)
            paragraph_sim = self.compute_paragraph_level_similarity(expected, prediction)

            semantic_score = (w_word * word_sim
                            + w_sent * sentence_sim
                            + w_para * paragraph_sim)

            # clamp out base SBERT bias (~0.25 for unrelated)
            semantic_score = max(0.0, semantic_score - 0.25)

            # threshold to 1.0
            return 1.0 if semantic_score >= 0.95 else semantic_score

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



    # def evaluate_true_false_questions(self):
    #     """
    #     Evaluate all True/False questions using precision, recall, and F1.
    #     Accepts only 'true' or 'false' predictions.
    #     Returns all metrics along with per-example scores.
    #     """
    #     try:
    #         blob = self.output_data.get("true_false")
    #         if not blob:
    #             print("No True/False output data found.", flush=True)
    #             return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}

    #         inputs = blob.get("inputs", [])
    #         predictions = blob.get("responses", [])

    #         if len(inputs) != len(predictions):
    #             print("Mismatch in number of inputs and predictions.", flush=True)
    #             return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}

    #         results = {}
    #         all_expected = []
    #         all_predicted = []

    #         for gold, pred in zip(inputs, predictions):
    #             predicted = self._to_text(pred).strip().lower()
    #             expected = str(gold.get("answer", "")).strip().lower()

    #             # Accept *only* the exact words "true" or "false"
    #             if predicted == "true" or predicted == "false":
    #                 all_expected.append(expected)
    #                 all_predicted.append(predicted)
    #                 score = 1.0 if predicted == expected else 0.0
    #             else:
    #                 all_expected.append(expected)
    #                 all_predicted.append("invalid")  # force a false match
    #                 score = 0.0

    #             para_id = gold.get("source", {}).get("paragraph_id", "unknown")
    #             results[para_id] = {
    #                 "question": gold.get("question", ""),
    #                 "expected": expected,
    #                 "predicted": predicted,
    #                 "score": score,
    #                 "source": gold.get("source", {})
    #             }

    #         # Compute precision, recall, and F1 using your existing method
    #         # Convert true/false to binary labels: true=1, false=0
    #         binary_map = {"true": 1, "false": 0}
    #         try:
    #             true_labels = [binary_map.get(e, 0) for e in all_expected]
    #             pred_labels = [binary_map.get(p, 0) for p in all_predicted]
    #         except Exception as label_error:
    #             print(f"Label conversion error: {label_error}", flush=True)
    #             return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": results}

    #         precision, recall, f1 = self._safe_binary_metrics(true_labels, pred_labels)
    #         avg_score = sum(results[pid]["score"] for pid in results) / len(results) if results else 0.0

    #         print(f"True/False Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}", flush=True)

    #         return {
    #             "average": avg_score,
    #             "precision": precision,
    #             "recall": recall,
    #             "f1_score": f1,
    #             "scores": results
    #         }

    #     except Exception as e:
    #         print(f"Error evaluating True/False questions: {e}", flush=True)
    #         return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}

    def evaluate_true_false_questions(self):
        """
        True/False (yes‑no) questions are scored with **accuracy** –
        the proportion of items answered exactly correctly.
        """
        try:
            blob = self.output_data.get("true_false")
            if not blob:
                print("No True/False output data found.", flush=True)
                return {"accuracy": 0.0, "scores": {}}

            inputs       = blob.get("inputs", [])
            predictions  = blob.get("responses", [])

            if len(inputs) != len(predictions):
                print("Mismatch in number of inputs and predictions.", flush=True)
                return {"accuracy": 0.0, "scores": {}}

            results     = {}
            correct_cnt = 0

            for gold, pred in zip(inputs, predictions):
                expected   = str(gold.get("answer", "")).strip().lower()
                predicted  = self._to_text(pred).strip().lower()

                is_correct = expected == predicted
                correct_cnt += int(is_correct)

                para_id = gold.get("source", {}).get("paragraph_id", "unknown")
                results[para_id] = {
                    "question":  gold.get("question", ""),
                    "expected":  expected,
                    "predicted": predicted,
                    "correct":   is_correct,
                    "source":    gold.get("source", {})
                }

            accuracy = correct_cnt / len(inputs) if inputs else 0.0
            print(f"True/False Accuracy: {accuracy:.3f}", flush=True)

            return {"accuracy": accuracy, "scores": results}

        except Exception as e:
            print(f"Error evaluating True/False questions: {e}", flush=True)
            return {"accuracy": 0.0, "scores": {}}


    # def evaluate_multiple_choice_questions(self):
    #     """
    #     Evaluate Multiple Choice questions by normalizing and comparing text-to-text.
    #     Returns average score, precision, recall, F1, and per-question results.
    #     """

    #     def normalise(text: str) -> str:
    #         """Lowercase, strip, collapse whitespace, remove most punctuation."""
    #         text = re.sub(r"[^\w\s%]", "", text.lower())
    #         return re.sub(r"\s+", " ", text).strip()

    #     try:
    #         blob = self.output_data.get("multiple_choice")
    #         if not blob:
    #             print("No Multiple Choice output data found.", flush=True)
    #             return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}

    #         inputs = blob.get("inputs", [])
    #         predictions = blob.get("responses", [])

    #         if len(inputs) != len(predictions):
    #             print("Mismatch in number of inputs and predictions.", flush=True)
    #             return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}

    #         results = {}
    #         all_expected = []
    #         all_predicted = []
    #         raw_scores = []

    #         for gold, pred in zip(inputs, predictions):
    #             try:
    #                 options = gold.get("options", {})
    #                 if isinstance(options, list):
    #                     options = {chr(65 + idx): opt for idx, opt in enumerate(options)}

    #                 # Handle whether correct_answer is a letter ("B") or full text
    #                 correct_raw = str(gold.get("correct_answer", "")).strip()
    #                 if len(correct_raw) == 1 and correct_raw.upper() in options:
    #                     expected_orig = options[correct_raw.upper()]
    #                 else:
    #                     expected_orig = correct_raw

    #                 expected_text = normalise(expected_orig)
    #                 predicted_text = normalise(self._to_text(pred))

    #                 score = 1.0 if predicted_text == expected_text else 0.0
    #                 raw_scores.append(score)

    #                 all_expected.append(expected_text)
    #                 all_predicted.append(predicted_text)

    #                 para_id = gold.get("source", {}).get("paragraph_id", "unknown")
    #                 results[para_id] = {
    #                     "question": gold.get("question", ""),
    #                     "expected": expected_orig,           # raw human-readable expected text
    #                     "predicted": self._to_text(pred),     # raw model output
    #                     "predicted_text": self._to_text(pred),
    #                     "options": options,
    #                     "score": score,
    #                     "source": gold.get("source", {})
    #                 }
    #             except Exception as inner_e:
    #                 print(f"Error evaluating multiple choice QA: {inner_e}", flush=True)

    #         avg_score = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0

    #         # Binary labels for precision/recall/F1
    #         all_expected_bin = [1 for _ in all_expected]  # All expected = 1
    #         all_predicted_bin = [1 if p == e else 0 for p, e in zip(all_predicted, all_expected)]

    #         precision, recall, f1 = self.compute_classification_metrics(
    #             all_expected_bin, all_predicted_bin, average="binary"
    #         )

    #         print(f"Multiple Choice Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}", flush=True)

    #         return {
    #             "average": avg_score,
    #             "precision": precision,
    #             "recall": recall,
    #             "f1_score": f1,
    #             "scores": results
    #         }

    #     except Exception as e:
    #         print(f"Error evaluating Multiple Choice questions: {e}", flush=True)
    #         return {"average": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "scores": {}}


    def evaluate_multiple_choice_questions(self):
        """
        Multiple‑choice questions are scored with **accuracy** –
        the fraction of questions where the selected option is exactly correct.
        """
        def normalise(text: str) -> str:
            text = re.sub(r"[^\w\s%]", "", text.lower())
            return re.sub(r"\s+", " ", text).strip()

        try:
            blob = self.output_data.get("multiple_choice")
            if not blob:
                print("No Multiple Choice output data found.", flush=True)
                return {"accuracy": 0.0, "scores": {}}

            inputs      = blob.get("inputs", [])
            predictions = blob.get("responses", [])

            if len(inputs) != len(predictions):
                print("Mismatch in number of inputs and predictions.", flush=True)
                return {"accuracy": 0.0, "scores": {}}

            results     = {}
            correct_cnt = 0

            for gold, pred in zip(inputs, predictions):
                # Build {letter → option_text} if needed
                opts = gold.get("options", {})
                if isinstance(opts, list):
                    opts = {chr(65 + i): o for i, o in enumerate(opts)}

                gold_raw = str(gold.get("correct_answer", "")).strip()
                gold_txt = opts.get(gold_raw.upper(), gold_raw)
                expected = normalise(gold_txt)

                predicted_txt = normalise(self._to_text(pred))

                is_correct = predicted_txt == expected
                correct_cnt += int(is_correct)

                para_id = gold.get("source", {}).get("paragraph_id", "unknown")
                results[para_id] = {
                    "question":  gold.get("question", ""),
                    "expected":  gold_txt,
                    "predicted": self._to_text(pred),
                    "correct":   is_correct,
                    "options":   opts,
                    "source":    gold.get("source", {})
                }

            accuracy = correct_cnt / len(inputs) if inputs else 0.0
            print(f"Multiple‑Choice Accuracy: {accuracy:.3f}", flush=True)

            return {"accuracy": accuracy, "scores": results}

        except Exception as e:
            print(f"Error evaluating Multiple Choice questions: {e}", flush=True)
            return {"accuracy": 0.0, "scores": {}}


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
                    options = [opt.strip().lower() for opt in gold.get("options", [])]
                    expected_values = [v.strip().lower() for v in gold.get("answer", [])]

                    # Model output → list of option texts
                    pred_text = self._to_text(raw_pred).strip().lower()
                    predicted_values = [p.strip() for p in pred_text.split(",") if p.strip()]

                    # Compute overlap metrics
                    precision, recall, f1 = self.compute_overlap_f1(expected_values, predicted_values)

                    # Global precision/recall counts
                    tp = len(set(expected_values) & set(predicted_values))
                    fp = len(set(predicted_values) - set(expected_values))
                    fn = len(set(expected_values) - set(predicted_values))
                    global_tp += tp
                    global_fp += fp
                    global_fn += fn

                    para_id = gold.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": gold.get("question", ""),
                        "expected": expected_values,
                        "predicted": predicted_values if predicted_values else pred_text,
                        "options": options,
                        "f1_score": f1,
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
                        raw = self._to_text(predicted).strip().lower()
                        predicted = re.sub(r"^incorrect explanation:\s*", "", raw, flags=re.I)

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



    def _extract_step_number(self, txt: str):
        """
        Return the first integer that follows the word 'step' (case‑insensitive),
        or None if not found.
        """
        m = re.search(r"step\s*([0-9]+)", txt, flags=re.I)
        return int(m.group(1)) if m else None


    def _penalise_similarity(self, base_sim: float, step_distance: int):
        """
        Down‐weight the semantic score by how far the predicted step is 
        from the true one. Closer → milder penalty:
            distance=0 → 1.0×
            distance=1 → 0.7×
            distance=2 → 0.3×
            distance=3 → 0.15×
            distance=4 → 0.075×  (and so on halving each extra step)
        """
        if step_distance is None or step_distance == 0:
            return base_sim

        if step_distance == 1:
            factor = 0.7
        elif step_distance == 2:
            factor = 0.3
        else:
            # for d ≥ 3, halve the distance-2 penalty each extra step
            factor = 0.3 / (2 ** (step_distance - 2))

        return base_sim * factor

    
    def evaluate_multi_hop_inverse_questions(self):
        """
        Evaluate Multi-hop Inverse questions using semantic similarity metrics,
        penalized by how far the predicted step is from the true incorrect step.
        """
        try:
            blob = self.output_data.get("multi_hop_inverse")
            if not blob:
                print("No Multi-hop Inverse output data found.", flush=True)
                return {"semantic_match_score": 0.0, "scores": {}}

            inputs      = blob.get("inputs", [])
            predictions = blob.get("responses", [])
            if len(inputs) != len(predictions):
                print("Mismatch in inputs vs responses for Multi-hop Inverse.", flush=True)
                return {"semantic_match_score": 0.0, "scores": {}}

            results = {}
            scores  = []

            for gold, pred in zip(inputs, predictions):
                try:
                    # --- ground truth step & explanation ---
                    irs = gold.get("incorrect_reasoning_step", "")
                    # turn it into a list of non-empty lines
                    if isinstance(irs, list):
                        lines = irs
                    else:
                        lines = [l.strip() for l in irs.splitlines() if l.strip()]

                    # find the line that mentions the step (e.g. "- **Step 4** contains ...")
                    step_line = next(
                        (l for l in lines
                        if re.search(r"\bStep\b.*?(\d+)", l, flags=re.I)),
                        ""
                    )

                    # find the line that contains the explanation
                    expl_line = next(
                        (l for l in lines
                        if re.search(r"Explanation", l, flags=re.I)),
                        ""
                    )

                    # extract the integer step
                    gt_step_number = self._extract_step_number(step_line)

                    # rip off everything up to the colon so we just get the explanation text
                    gt_expl = expl_line.split(":", 1)[-1].strip()

                    # --- model output parsing ---
                    raw_txt = self._to_text(pred).strip()
                    # ppredicted-step extraction
                    step_match = re.search(
                        r"(?:incorrect reasoning step\s*[:\-]?\s*|step\s*[:\-]?\s*)(\d+)",
                        raw_txt,
                        flags=re.IGNORECASE
                    )
                    pred_step = int(step_match.group(1)) if step_match else None

                    # robust explanation extraction
                    # match either header “Incorrect Reasoning Explanation” or just “Explanation”
                    expl_parts = re.split(
                        r"(?:incorrect reasoning explanation|explanation)\s*[:\-]?",
                        raw_txt,
                        flags=re.IGNORECASE,
                        maxsplit=1
                    )
                    pred_expl = expl_parts[1].strip() if len(expl_parts) == 2 else raw_txt

                    # --- compute base sim + penalty ---
                    base_sim = self.evaluate_open_ended(gt_expl, pred_expl)
                    dist     = (abs(gt_step_number - pred_step)
                                if gt_step_number is not None and pred_step is not None
                                else None)
                    sim_score = self._penalise_similarity(base_sim, dist)
                    step_corr = 1.0 if gt_step_number == pred_step else 0.0

                    para_id = gold.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question":              gold.get("question", ""),
                        "expected_step":         gt_step_number,
                        "predicted_step":        pred_step,
                        "step_distance":         dist,
                        "step_correct":          step_corr,
                        "expected_explanation":  gt_expl,
                        "predicted_explanation": pred_expl,
                        "semantic_match_score":  sim_score,
                        "metrics":               self.evaluate_open_ended_metrics(gt_expl, pred_expl),
                        "source":                gold.get("source", {})
                    }
                    scores.append(sim_score)

                except Exception as inner_e:
                    print(f"Error evaluating Multi-hop Inverse QA: {inner_e}", flush=True)

            avg_score = float(np.mean(scores)) if scores else 0.0
            # compute average step-identification rate
            step_rates = [v["step_correct"] for v in results.values()]
            avg_step_identification_rate = float(np.mean(step_rates)) if step_rates else 0.0

            print(f"Average Multi-hop Inverse Semantic Similarity Score: {avg_score:.2f}", flush=True)
            print(f"Step identification accuracy: {avg_step_identification_rate:.2f}", flush=True)

            return {
                "semantic_match_score":       avg_score,
                "step_identification_rate":   avg_step_identification_rate,
                "scores":                     results
            }

        except Exception as e:
            print(f"Error in evaluate_multi_hop_inverse_questions: {e}", flush=True)
            return {"semantic_match_score": 0.0, "scores": {}}



    def plot_ecdfs(self, score_dict, out_dir, metric="semantic"):
        plots_dir = os.path.join(out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8,5))
        for key, scores in score_dict.items():
            if not scores:
                continue
            x = np.sort(scores)
            y = np.arange(1, len(x)+1) / len(x)
            ax.plot(x, y, marker='.', linestyle='none',
                    label=key.replace('_',' ').title())
        ax.set_xlabel(metric.title() + " value")
        ax.set_ylabel("ECDF")
        ax.set_xlim(0,1)
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{metric}_ecdfs.svg"))
        plt.close(fig)

    def plot_boxplots(self, score_dict, out_dir, metric="semantic"):
        plots_dir = os.path.join(out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        labels, data = [], []
        for key, scores in score_dict.items():
            if not scores: continue
            labels.append(key.replace('_',' ').title())
            data.append(scores)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.boxplot(data, labels=labels, showfliers=True, vert=True)
        ax.set_ylabel(metric.title() + " value")
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0,1)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{metric}_boxplots.svg"))
        plt.close(fig)

    def plot_jitter_scatter(self, score_dict, out_dir, metric="semantic"):
        plots_dir = os.path.join(out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8,5))
        for i, (key, scores) in enumerate(score_dict.items()):
            if not scores: continue
            x = scores
            y = np.random.uniform(i - 0.1, i + 0.1, size=len(x))
            ax.plot(x, y, 'o', alpha=0.4,
                    label=key.replace('_',' ').title())
        ax.set_xlabel(metric.title() + " value")
        ax.set_yticks(range(len(score_dict)))
        ax.set_yticklabels([k.replace('_',' ').title()
                            for k in score_dict.keys()])
        ax.set_xlim(0,1)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{metric}_jitter.svg"))
        plt.close(fig)

    def plot_histograms(self, score_dict, out_dir, metric="semantic"):
        plots_dir = os.path.join(out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        bins = np.arange(0.0, 1.0 + self.bin_width, self.bin_width)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        keys = list(score_dict.keys())
        for ax, key in zip(axes.flat, keys):
            scores = score_dict[key]
            if not scores:
                ax.set_visible(False)
                continue
            weights = np.ones_like(scores) / len(scores)
            ax.hist(scores, bins=bins, weights=weights, edgecolor="black")
            ax.set_title(key.replace("_", " ").title())
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.set_xlabel(metric.title() + " value")
            ax.set_ylabel("Proportion")
            ax.set_xticks(bins)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{metric}_histograms.svg"))
        plt.close(fig)

    def plot_dashboard(self, all_score_dicts, out_dir):
        """
        all_score_dicts: dict of metric→score_dict
        Produces an n×3 grid: rows=metrics, cols=(box, jitter, hist)
        """
        plots_dir = os.path.join(out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        metrics = list(all_score_dicts.keys())
        n = len(metrics)

        # 3 columns now (box, jitter, hist)
        fig, axes = plt.subplots(n, 3, figsize=(12, 3*n), squeeze=False)

        for i, metric in enumerate(metrics):
            sd = all_score_dicts[metric]

            # ─── Boxplot (col 0)
            ax = axes[i, 0]
            data   = [sd[k] for k in sd if sd[k]]
            labels = [k.replace('_',' ').title() for k in sd if sd[k]]
            if data:
                ax.boxplot(data, labels=labels, showfliers=True)
                ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_title(f"{metric.title()} Boxplot")
            ax.set_ylim(0, 1)

            # ─── Jitter (col 1)
            ax = axes[i, 1]
            for j, (key, scores) in enumerate(sd.items()):
                if not scores:
                    continue
                y = np.random.uniform(j - 0.1, j + 0.1, size=len(scores))
                ax.plot(scores, y, 'o', alpha=0.4)
            ax.set_title(f"{metric.title()} Jitter")
            ax.set_xlim(0, 1)
            ax.set_yticks(range(len(sd)))
            ax.set_yticklabels([k.replace('_',' ').title() for k in sd])

            # ─── Histogram (col 2)
            ax = axes[i, 2]
            bins = np.arange(0.0, 1.0 + self.bin_width, self.bin_width)
            for key, scores in sd.items():
                if not scores:
                    continue
                weights = np.ones_like(scores) / len(scores)
                ax.hist(
                    scores,
                    bins=bins,
                    weights=weights,
                    alpha=0.5,
                    label=key.replace('_',' ').title()
                )
            ax.set_title(f"{metric.title()} Histogram")
            ax.set_xlim(0, 1)
            ax.legend(loc='upper right', fontsize='small')

        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "all_metrics_dashboard.svg"), format="svg")
        plt.close(fig)




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

            overall_results["multi_hop_inverse_step_identification_rate"] = \
                multi_hop_inverse_results.get("step_identification_rate", 0.0)

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

            open_ended_metrics = {}
            for name, result in [
                ("short", short_results),
                ("short_inverse", short_inverse_results),
                ("multi_hop", multi_hop_results),
                ("multi_hop_inverse", multi_hop_inverse_results),
            ]:
                # extract the per‐item metric dicts
                metrics_list = [v["metrics"] for v in result["scores"].values()]
                bleus   = [m["bleu"]   for m in metrics_list]
                meteors = [m["meteor"] for m in metrics_list]
                rouges  = [m["rouge"]  for m in metrics_list]

                open_ended_metrics[name] = {
                    "avg_bleu":   sum(bleus)   / len(bleus)   if bleus   else 0.0,
                    "avg_meteor": sum(meteors) / len(meteors) if meteors else 0.0,
                    "avg_rouge":  sum(rouges)  / len(rouges)  if rouges  else 0.0,
                }

            # ─── Compute overall average of those three across all open‑ended types ───
            overall_open = {
                "avg_bleu":   sum(m["avg_bleu"]   for m in open_ended_metrics.values()) / len(open_ended_metrics),
                "avg_meteor": sum(m["avg_meteor"] for m in open_ended_metrics.values()) / len(open_ended_metrics),
                "avg_rouge":  sum(m["avg_rouge"]  for m in open_ended_metrics.values()) / len(open_ended_metrics),
            }

            # ─── Attach into your overall_results structure ───
            overall_results["open_ended_metrics_per_type"] = open_ended_metrics
            overall_results["open_ended_metrics_overall"]   = overall_open


            # --------------- gather open‑ended similarity distributions ---------------
            open_ended_scores = {
                "short":           [v["semantic_match_score"] for v in short_results["scores"].values()],
                "short_inverse":   [v["semantic_match_score"] for v in short_inverse_results["scores"].values()],
                "multi_hop":       [v["semantic_match_score"] for v in multi_hop_results["scores"].values()],
                "multi_hop_inverse":[v["semantic_match_score"] for v in multi_hop_inverse_results["scores"].values()],
            }

            # ────── compute single “overall effectiveness” score ──────
            to_avg = [
                overall_results["true_false"]["accuracy"],
                overall_results["multiple_choice"]["accuracy"],
                overall_results["list"]["macro_f1_score"],
                overall_results["short"]["semantic_match_score"],
                overall_results["short_inverse"]["semantic_match_score"],
                overall_results["multi_hop"]["semantic_match_score"],
                overall_results["multi_hop_inverse"]["semantic_match_score"],
            ]
            overall_score = float(np.mean(to_avg))
            overall_results["overall_effectiveness"] = overall_score
            print(f"Overall effectiveness score: {overall_score:.3f}", flush=True)
            # ────────────────────────────────────────────────────────────

            # descriptive stats
            for k, vals in open_ended_scores.items():
                if not vals:
                    continue
                arr = np.array(vals)
                overall_results.setdefault("open_ended_stats", {})[k] = {
                    "min":   float(arr.min()),
                    "p25":   float(np.percentile(arr, 25)),
                    "median":float(np.median(arr)),
                    "p75":   float(np.percentile(arr, 75)),
                    "max":   float(arr.max()),
                }

                # ───────── semantic‐similarity plots ─────────
                # self.plot_ecdfs(open_ended_scores,    output_dir, metric="semantic")
                self.plot_boxplots(open_ended_scores, output_dir, metric="semantic")
                self.plot_jitter_scatter(open_ended_scores, output_dir, metric="semantic")
                self.plot_histograms(open_ended_scores,   output_dir, metric="semantic")

                # ───────── BLEU/METEOR/ROUGE plots ─────────
                all_metrics = {
                    "bleu":   { k: [v["metrics"]["bleu"]   for v in res["scores"].values()]
                                for k,res in zip(
                                    ["short","short_inverse","multi_hop","multi_hop_inverse"],
                                    [short_results, short_inverse_results,
                                    multi_hop_results, multi_hop_inverse_results]
                                )},
                    "meteor": { k: [v["metrics"]["meteor"] for v in res["scores"].values()]
                                for k,res in zip(
                                    ["short","short_inverse","multi_hop","multi_hop_inverse"],
                                    [short_results, short_inverse_results,
                                    multi_hop_results, multi_hop_inverse_results]
                                )},
                    "rouge":  { k: [v["metrics"]["rouge"]  for v in res["scores"].values()]
                                for k,res in zip(
                                    ["short","short_inverse","multi_hop","multi_hop_inverse"],
                                    [short_results, short_inverse_results,
                                    multi_hop_results, multi_hop_inverse_results]
                                )},
                }

                for metric, sd in all_metrics.items():
                    # self.plot_ecdfs(sd,    output_dir, metric=metric)
                    self.plot_boxplots(sd, output_dir, metric=metric)
                    self.plot_jitter_scatter(sd, output_dir, metric=metric)
                    self.plot_histograms(sd,   output_dir, metric=metric)

                # ───────── combined dashboard ─────────
                self.plot_dashboard(
                    {"semantic": open_ended_scores, **all_metrics},
                    output_dir
                )


            #  ────── save raw distributions and stats to JSON ──────
            with open(os.path.join(output_dir, "open_ended_scores.json"), "w") as f:
                json.dump(open_ended_scores, f, indent=2)

            with open(os.path.join(output_dir, "open_ended_stats.json"), "w") as f:
                json.dump(overall_results["open_ended_stats"], f, indent=2)
            # ───────────────────────────────────────────────────────


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
    
    parser.add_argument(
        "--bin_width",
        type=float,
        default=0.05,
        help="bin width for semantic‑similarity histograms (default: 0.05)"
    )


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluator = ClinIQLinkSampleDatasetEvaluate(run_mode=args.mode, results_dir=args.results_dir, bin_width=args.bin_width)
    evaluator.run_all_evaluations()