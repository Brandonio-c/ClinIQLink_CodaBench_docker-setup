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
from collections import Counter, OrderedDict
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
import unicodedata

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
    def __init__(self, run_mode="container", results_dir="submission_output", bin_width=0.05, output_dir="evaluation_output"):
        self.run_mode = run_mode.lower()
        self.results_dir = results_dir
        self.bin_width  = bin_width
        self.output_dir = output_dir
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
        

    # def _load_outputs(self):
    #     data = {}
    #     for k, fname in self.QA_FILES.items():
    #         fpath = os.path.join(self.results_dir, fname)
    #         if not os.path.exists(fpath):
    #             print(f"[WARN] {fpath} not found – skipping {k}", flush=True)
    #             data[k] = None
    #             continue
    #         data[k] = self.load_json(fpath)
    #     return data

    def _load_outputs(self):
        data = {}
        for k, fname in self.QA_FILES.items():
            fpath = os.path.join(self.results_dir, fname)
            if not os.path.exists(fpath):
                print(f"[WARN] {fpath} not found – skipping {k}", flush=True)
                data[k] = None
                continue

            # Load original JSON blob
            blob = self.load_json(fpath)

            if blob and "responses" in blob:
                responses = blob["responses"]
                cleaned_responses, reasoning_steps = self._extract_reasoning_steps(responses)
                blob["responses"] = cleaned_responses
                blob["reasoning_model_thinking_steps"] = reasoning_steps

            data[k] = blob
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
    

    def _extract_reasoning_steps(self, responses):
        """
        Given a list of model responses, extract `<think>...</think>` blocks,
        remove them from the main text, and return:
        - cleaned responses
        - list of extracted reasoning steps (as strings or list of strings)
        """
        cleaned_responses = []
        reasoning_steps = []

        for resp in responses:
            text = self._to_text(resp)

            # Extract all <think>...</think> blocks
            steps = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)

            # Remove all <think>...</think> from the original
            cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

            cleaned_responses.append(cleaned)
            reasoning_steps.append(steps if steps else [])

        return cleaned_responses, reasoning_steps

    def normalise(self, text: str) -> str:
        # Unicode NFKD normalization & strip diacritics
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))

        # case-fold for full Unicode case-insensitive matching
        text = text.casefold()

        # replace parentheses and other delimiters with spaces
        text = re.sub(r"[–—_/\\\[\]\(\)\{\};\-]", " ", text)

        # remove anything that’s not a letter, digit, mark, space, percent or comma
        #    keep commas so we can split on them later
        cleaned = []
        for c in text:
            cat = unicodedata.category(c)
            if cat[0] in ("L", "N", "M") or c.isspace() or c in {"%", ",", ":"}:
                cleaned.append(c)
            # else drop it
        text = "".join(cleaned)

        # collapse any run of whitespace into a single space
        return re.sub(r"\s+", " ", text).strip()

    
    def _list_match_letter_or_text(
                                    self,
                                    expected_values,
                                    raw_resp,
                                    letter_to_text,
                                    options,
                                ):
        """
        Determine match type for multi-select (list) questions directly from raw_resp.
        Returns (match_type, malformed, predicted_values), where predicted_values is the
        list of normalized option texts that were detected (in the order seen).
        """
        
        raw_resp = raw_resp.split("\n\n")[0].strip()

        expected_set = {self.normalise(v) for v in expected_values}

        norm_opts_set = {self.normalise(o) for o in options}
        norm_predicted_items = {self.normalise(seg) for seg in re.split(r",\s*", raw_resp) if seg.strip()}
        if norm_predicted_items and norm_predicted_items.issubset(norm_opts_set):
            return "full", False, list(norm_predicted_items), []

        # Strip parentheses and prepare raw segments
        cleaned = re.sub(r"[()]", "", raw_resp)

        # build a regex class from your actual option letters (so J gets picked up too)
        letters = "".join(letter_to_text.keys())            # e.g. "ABCDEFGHIJ"
        letter_pattern = rf"([{letters}])[:\.]\s*(.+?)(?=(?:\s*[{letters}][\:\.])|$)"
        matches = re.findall(letter_pattern, cleaned, flags=re.IGNORECASE)

        if matches:
            # if there's exactly one letter-prefix but it contains commas,
            # drop the "A:" and split the contents into separate items
            if len(matches) == 1 and "," in matches[0][1]:
                raw_list = matches[0][1]
                base_segments = [
                    item.strip().rstrip(",")
                    for item in raw_list.split(",")
                    if item.strip()
                ]
            else:
                base_segments = [
                    f"{ltr}: {txt.strip().rstrip(',').strip()}"
                    for ltr, txt in matches
                ]
        else:
            base_segments = [
                seg.strip().rstrip(",")
                for seg in cleaned.split(",")
                if seg.strip()
            ]

        #  # Clean up parentheses & trailing periods
        # cleaned = re.sub(r"[()]", "", raw_resp).rstrip()

        # # Build a regex class for your actual letters:
        # #    if letter_to_text has keys "A".."Q", this becomes "[A-Q]"
        # letter_class = "[" + "".join(letter_to_text.keys()) + "]"

        # #  Split on any "<LETTER>:" or "<LETTER>." prefix
        # parts = re.split(rf"\s*{letter_class}\s*[:\.]\s*", cleaned)

        # # parts[0] is the text before the first prefix (usually empty), so drop it
        # raw_segments = parts[1:]

        # # Strip *all* trailing punctuation, lowercase, normalise
        # base_segments = [
        #     seg.strip().rstrip(".,;:") 
        #     for seg in raw_segments 
        #     if seg.strip()
        # ]

        # Normalized option map and max comma span for reconstruction
        opt_norm = {self.normalise(o): o for o in options}
        norm_opts_set = set(opt_norm.keys()) 
        max_parts = max(o.count(",") + 1 for o in options)

        has_composite_options = any("," in o for o in options)
        segments = []  
        unmatched_segments = []
        if has_composite_options:
            i = 0
            while i < len(base_segments):
                if has_composite_options:
                    found = False
                    for window in range(max_parts, 0, -1):
                        if i + window > len(base_segments):
                            continue
                        candidate = ", ".join(base_segments[i:i + window])
                        norm_candidate = self.normalise(candidate)
                        if norm_candidate in opt_norm:
                            segments.append(opt_norm[norm_candidate])
                            i += window
                            found = True
                            break
                    if not found:
                        norm_single = self.normalise(base_segments[i])
                        if norm_single in opt_norm:
                            segments.append(opt_norm[norm_single])
                        else:
                            segments.append(base_segments[i])  # preserve original for fallback match
                        i += 1
                else:
                    norm_seg = self.normalise(base_segments[i])
                    segments.append(opt_norm.get(norm_seg, norm_seg))
                    i += 1
        else:
            for seg in base_segments:
                norm_seg = self.normalise(seg)
                segments.append(opt_norm.get(norm_seg, norm_seg))


        predicted_values = []
        pred_letters = set()
        pred_texts = set()
        malformed = False
        match_type = "none"

        for seg in segments:
            seg_matched = False

            # pure letter → letter-only match
            if re.fullmatch(r"[A-Za-z]", seg):
                L = seg.upper()
                if L in letter_to_text:
                    txt = self.normalise(letter_to_text[L])
                    predicted_values.append(txt)
                    pred_letters.add(L)
                    seg_matched = True
                    if match_type == "none":
                        match_type = "letter"

            # letter:text → mixed match
            m = re.match(r"^([A-Za-z])[:\.]\s*(.+)$", seg)
            if m:
                L = m.group(1).upper()
                # strip trailing commas before normalising
                candidate = m.group(2).strip().rstrip(",").strip()
                norm_txt = self.normalise(candidate)

                # exact letter→text
                if L in letter_to_text and self.normalise(letter_to_text[L]) == norm_txt:
                    predicted_values.append(norm_txt)
                    pred_letters.add(L)
                    pred_texts.add(norm_txt)
                    seg_matched = True
                    match_type = "mixed"

                # fallback: if the raw text itself is a valid option, take it
                elif norm_txt in opt_norm:
                    predicted_values.append(norm_txt)
                    pred_texts.add(norm_txt)
                    seg_matched = True
                    if match_type == "none":
                        match_type = "full"
                    elif match_type == "letter":
                        match_type = "mixed"

                # cleanup
                seg = seg.replace(":", "").strip()

            # free-text → full-text match
            if not seg_matched:
                norm_seg = self.normalise(seg)
                if norm_seg in opt_norm:
                    predicted_values.append(norm_seg)
                    pred_texts.add(norm_seg)
                    seg_matched = True
                    if match_type == "none":
                        match_type = "full"
                    elif match_type == "letter":
                        match_type = "mixed"

            # if nothing matched
            if not seg_matched:
                # Only mark malformed if the segment is not even in the normalized options
                norm_seg = self.normalise(seg)
                predicted_values.append(norm_seg)

                if norm_seg not in norm_opts_set:
                    unmatched_segments.append((seg, norm_seg))
                    malformed = True
                    # print(f"[MALFORMED] HALLUCINATED RESPONSE DETECTED! - Raw response: '{raw_resp}'")
                    # print(f"  → norm_seg: '{norm_seg}'")
                    # print(f"  → norm_opts_set: {sorted(norm_opts_set)}")
                    # for orig, norm in unmatched_segments:
                    #     print(f"  → Segment '{orig}' normalized to '{norm}' not in options.")

        # override to full if exact match
        # Final match type resolution

        if set(predicted_values) == expected_set:
            match_type = "full"

        return match_type, malformed, predicted_values, unmatched_segments



    
    def _mc_match_letter_or_text(self, expected_answer, prediction, options):
        """
        Determine match type for multiple-choice predictions:
        - 'full': predicted text matches correct option text
        - 'letter': predicted letter matches correct letter
        - 'mixed': both letter and text given and correct
        - 'none': neither letter nor text matches
        - 'invalid': more than one answer detected

        Only one correct answer is allowed for multiple-choice.
        """
        # Immediate full-text match check
        if prediction == expected_answer:
            return "full"
        
        # Build letter → option text mapping
        if isinstance(options, list):
            options = {chr(65 + i): opt for i, opt in enumerate(options)}
        norm_options = {k: self.normalise(v) for k, v in options.items()}
        rev_options = {v: k for k, v in norm_options.items()}  # normalized text → letter
        

        expected_answer_norm = self.normalise(expected_answer)
        expected_letter = rev_options.get(expected_answer_norm, None)
        prediction_raw = self._to_text(prediction).strip()

        # Split on commas to detect multi-option answers
        # splits = [s.strip() for s in prediction_raw.split(",") if s.strip()]
        # if len(splits) > 1:
        #     return "invalid"  # multiple answers not allowed for MC

        prediction = self.normalise(self._to_text(prediction))


        # Direct full-text match
        is_full = prediction == expected_answer

        # Direct letter match
        is_letter = prediction.upper() in options and prediction.upper() == expected_letter

        # Mixed form: e.g., "A: Hypertension"
        if ":" in prediction_raw:
            parts = prediction_raw.split(":", 1)
            ltr = parts[0].strip().upper()
            txt = parts[1].strip().lower()
            mixed_valid = (
                ltr in options and
                txt == options[ltr].strip().lower() and
                ltr == expected_letter and
                txt == expected_answer
            )
            if mixed_valid:
                return "mixed"

        if is_full and is_letter:
            return "mixed"
        elif is_full:
            return "full"
        elif is_letter:
            return "letter"
        else:
            return "none"



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
            true_counter = Counter(map(str, true_list))
            pred_counter = Counter(map(str, pred_list))

            tp = sum(min(true_counter[k], pred_counter[k]) for k in pred_counter if k in true_counter)
            fp = sum(pred_counter[k] - true_counter.get(k, 0) for k in pred_counter if k not in true_counter or pred_counter[k] > true_counter[k])
            fn = sum(true_counter[k] - pred_counter.get(k, 0) for k in true_counter if k not in pred_counter or true_counter[k] > pred_counter.get(k, 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            return precision, recall, f1
        except Exception as e:
            print(f"Error computing F1 score: {e}", flush=True)
            return 0.0, 0.0, 0.0


    def evaluate_true_false(self, expected, prediction):
        """
        Evaluate True/False questions: returns 1.0 if normalized answers match, else 0.0.
        Normalization includes lowercasing, trimming whitespace, and removing trailing punctuation.
        """
        try:
            def normalize(ans):
                ans = ans.strip().lower()
                return ans.rstrip(string.punctuation)  # Remove trailing punctuation

            return 1.0 if normalize(expected) == normalize(prediction) else 0.0
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
        Also records how many responses were malformed (i.e., not 'true' or 'false').
        **True / False** - *Accuracy*  
        accuracy = #correct / N 
        Invalid responses (not “true” or “false”) are flagged and counted
        """
        try:
            blob = self.output_data.get("true_false")
            if not blob:
                print("No True/False output data found.", flush=True)
                return {"accuracy": 0.0, "true_false_invalid_count": 0, "scores": {}}

            inputs      = blob.get("inputs", [])
            predictions = blob.get("responses", [])

            if len(inputs) != len(predictions):
                print("Mismatch in number of inputs and predictions.", flush=True)
                return {"accuracy": 0.0, "true_false_invalid_count": 0, "scores": {}}

            results = {}
            correct_cnt = 0
            invalid_count = 0

            for gold, pred in zip(inputs, predictions):
                # pull out the canonical expected
                expected = str(gold.get("answer", "")).strip().lower()

                raw_resp = self._to_text(pred).strip()

                #  try to match “true” or “false” at the very start
                m = re.match(r'^(true|false)', raw_resp, flags=re.IGNORECASE)
                if m:
                    predicted = m.group(1).lower()
                    # anything beyond that is “extra”
                    extra = raw_resp[len(m.group(0)):].strip()
                    # mark malformed if there *is* any extra text
                    malformed = bool(extra)
                else:
                    # neither “true” nor “false” at the front → invalid + malformed
                    predicted = ""
                    malformed = True

                # validity check (only count as invalid if it's not true/false at all)
                if predicted not in {"true", "false"}:
                    invalid_count += 1
                    is_correct = False
                else:
                    is_correct = (predicted == expected)
                    correct_cnt += int(is_correct)

                # build your result entry (raw_resp still carries the full text)
                para_id = gold.get("source", {}).get("paragraph_id", "unknown")
                results[para_id] = {
                    "question":     gold.get("question", ""),
                    "expected":     expected,
                    "raw response": raw_resp,
                    "predicted":    predicted,
                    "correct":      is_correct,
                    "malformed":    malformed,
                    "source":       gold.get("source", {}),
                }

            accuracy = correct_cnt / len(inputs) if inputs else 0.0
            print(f"True/False Accuracy: {accuracy:.3f}  |  Invalid Predictions: {invalid_count} / {len(inputs)}", flush=True)

            return {
                "accuracy": accuracy,
                "true_false_invalid_count": invalid_count,
                "scores": results
            }

        except Exception as e:
            print(f"Error evaluating True/False questions: {e}", flush=True)
            return {"accuracy": 0.0, "true_false_invalid_count": 0, "scores": {}}
        

    # def evaluate_multiple_choice_questions(self):
    #     """
    #     Evaluate Multiple Choice questions by normalizing and comparing text-to-text.
    #     Returns average score, precision, recall, F1, and per-question results.
    #     """

    #     def self.normalise(text: str) -> str:
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

    #                 expected_text = self.normalise(expected_orig)
    #                 predicted_text = self.normalise(self._to_text(pred))

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
        Multiple‑choice questions are scored with:
        - Accuracy: match after text normalization
        - Full match: prediction matches full correct answer text
        - Letter match: prediction matches correct letter
        - Mixed match: both letter and text are present and correct 
        Also tracks malformed predictions:  
        - multiple answers (e.g., “A, B”)  
        - mismatched letter/text (e.g., “A: Hypertension” but A ≠ Hypertension) 
        - Computes:
        • Overall accuracy (normalized text match)
        • Full match accuracy (match_type == 'full')
        • Combined match accuracy (match_type ∈ {'letter', 'mixed'})
        • Invalid prediction count
        • Match-type counts: full, letter, mixed, total
    """

        try:
            blob = self.output_data.get("multiple_choice")
            if not blob:
                print("No Multiple Choice output data found.", flush=True)
                return {
                    "multiple_choice_average": 0.0,
                    "multiple_choice_full_match_average": 0.0,
                    "multiple_choice_combined_match_average": 0.0,
                    "multiple_choice_invalid_count": 0,
                    "multiple_choice_full_match": 0.0,
                    "multiple_choice_letter_match": 0.0,
                    "multiple_choice_mixed_match": 0.0,
                    "scores": {}
                }

            inputs = blob.get("inputs", [])
            predictions = blob.get("responses", [])

            if len(inputs) != len(predictions):
                print("Mismatch in number of inputs and predictions.", flush=True)
                return {
                    "multiple_choice_average": 0.0,
                    "multiple_choice_full_match_average": 0.0,
                    "multiple_choice_combined_match_average": 0.0,
                    "multiple_choice_invalid_count": 0,
                    "multiple_choice_full_match": 0.0,
                    "multiple_choice_letter_match": 0.0,
                    "multiple_choice_mixed_match": 0.0,
                    "scores": {}
                }

            total = len(inputs)
            overall_correct = 0
            full_match_correct = 0
            combined_match_correct = 0
            invalid_count = 0

            match_counts = {"full": 0, "letter": 0, "mixed": 0, "total": 0}
            results = {}

            for gold, pred in zip(inputs, predictions):
                opts = gold.get("options", [])
                opts = {chr(65 + i): o for i, o in enumerate(opts)}

                gold_raw = str(gold.get("correct_answer", "")).strip()
                gold_txt = opts.get(gold_raw.upper(), gold_raw)
                expected_text = self.normalise(gold_txt)

                predicted_raw = self._to_text(pred)
                predicted_text = self.normalise(predicted_raw)

                match_type = self._mc_match_letter_or_text(expected_text, predicted_text, opts)

                malformed = False
                if match_type == "invalid":
                    invalid_count += 1
                    malformed = True
                else:
                    if match_type in match_counts:
                        match_counts[match_type] += 1
                    match_counts["total"] += 1

                is_correct = predicted_text == expected_text
                overall_correct += int(is_correct)
                if match_type == "full":
                    full_match_correct += 1
                if match_type in {"letter", "mixed"}:
                    combined_match_correct += 1

                para_id = gold.get("source", {}).get("paragraph_id", "unknown")
                results[para_id] = {
                    "question": gold.get("question", ""),
                    "expected": gold_txt,
                    "raw response": predicted_raw,
                    "predicted": predicted_raw,
                    "match_type": match_type,
                    "malformed": malformed,
                    "options": opts,
                    "source": gold.get("source", {}),
                }

            overall_avg = overall_correct / total if total else 0.0
            full_avg = full_match_correct / total if total else 0.0
            combined_avg = combined_match_correct / total if total else 0.0

            print("\n==================== Multiple-Choice Evaluation Summary ====================")
            print(f"→ Overall Average Accuracy:         {overall_avg:.4f}")
            print(f"→ Full Match Accuracy:              {full_avg:.4f}")
            print(f"→ Letter + Mixed Match Accuracy:    {combined_avg:.4f}")
            print(f"→ Invalid Predictions:              {invalid_count} / {total}", flush=True)
            print(f"→ Match Type Distribution:")
            print(f"   • Full:    {match_counts['full']}")
            print(f"   • Letter:  {match_counts['letter']}")
            print(f"   • Mixed:   {match_counts['mixed']}")
            print(f"   • Total:   {match_counts['total']}", flush=True)

            return {
                "multiple_choice_average": overall_avg,
                "multiple_choice_full_match_average": full_avg,
                "multiple_choice_combined_match_average": combined_avg,
                "multiple_choice_invalid_count": invalid_count,
                "multiple_choice_full_match": match_counts["full"] / match_counts["total"] if match_counts["total"] else 0.0,
                "multiple_choice_letter_match": match_counts["letter"] / match_counts["total"] if match_counts["total"] else 0.0,
                "multiple_choice_mixed_match": match_counts["mixed"] / match_counts["total"] if match_counts["total"] else 0.0,
                "scores": results
            }

        except Exception as e:
            print(f"Error evaluating Multiple Choice questions: {e}", flush=True)
            return {
                "multiple_choice_average": 0.0,
                "multiple_choice_full_match_average": 0.0,
                "multiple_choice_combined_match_average": 0.0,
                "multiple_choice_invalid_count": 0,
                "multiple_choice_full_match": 0.0,
                "multiple_choice_letter_match": 0.0,
                "multiple_choice_mixed_match": 0.0,
                "scores": {}
            }


    def compute_macro_micro_metrics(self, expected_sets, predicted_sets):
        """
        Compute macro and micro precision/recall/F1 metrics using sklearn.

        Args:
            expected_sets (List[Set[str]]): Ground truth answers per question.
            predicted_sets (List[Set[str]]): Predicted answers per question.

        Returns:
            Dict[str, float]: Dictionary with macro/micro precision, recall, F1
        """
        mlb = MultiLabelBinarizer()
        # teach it every label we'll ever see
        mlb.fit(expected_sets + predicted_sets)

        y_true = mlb.transform(expected_sets)
        y_pred = mlb.transform(predicted_sets)


        scores = {}

        for avg in ["macro", "micro"]:
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average=avg, zero_division=0
            )
            scores[f"{avg}_precision"] = p
            scores[f"{avg}_recall"] = r
            scores[f"{avg}_f1_score"] = f1

        return scores
    
    def compute_mrr(self, predicted: list[list[str]], ground_truth: list[list[str]]) -> float:
        reciprocal_ranks = []
        for preds, gold in zip(predicted, ground_truth):
            gold_set = set(gold)
            rank = next((i + 1 for i, p in enumerate(preds) if p in gold_set), None)
            reciprocal_ranks.append(1 / rank if rank else 0.0)
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


    def compute_map(self, predicted: list[list[str]], ground_truth: list[list[str]]) -> float:
        average_precisions = []
        for preds, gold in zip(predicted, ground_truth):
            gold_set = set(gold)
            num_hits = 0
            precisions = []
            for i, p in enumerate(preds):
                if p in gold_set:
                    num_hits += 1
                    precisions.append(num_hits / (i + 1))
            average_precisions.append(sum(precisions) / len(gold_set) if gold_set else 0.0)
        return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0


    def compute_precision_at_k(self, predicted: list[list[str]], ground_truth: list[list[str]], k: int = 5) -> float:
        p_at_k = []
        for preds, gold in zip(predicted, ground_truth):
            gold_set = set(gold)
            top_k_preds = preds[:k]
            hits = sum(1 for p in top_k_preds if p in gold_set)
            p_at_k.append(hits / k)
        return sum(p_at_k) / len(p_at_k) if p_at_k else 0.0

    def compute_additional_ir_metrics(self, expected_sets, predicted_lists, results, match_type_filter=None):
        """
        Compute IR-style metrics (MAP, MRR, P@5) optionally filtered by match_type.

        Args:
            expected_sets (List[Set[str]]): Gold labels.
            predicted_lists (List[List[str]]): Ranked predicted answers.
            results (Dict): Full QA record per item.
            match_type_filter (Set[str] or None): Match types to include.

        Returns:
            Dict[str, float]: {"map": ..., "mrr": ..., "p@5": ...}
        """
        filtered_expected = []
        filtered_predicted = []

        for para_id, r in results.items():
            if match_type_filter and r.get("match_type") not in match_type_filter:
                continue
            filtered_expected.append(r["expected"])
            filtered_predicted.append(r["predicted"])

        map_score = self.compute_map(filtered_predicted, filtered_expected)
        mrr_score = self.compute_mrr(filtered_predicted, filtered_expected)
        p_at_5 = self.compute_precision_at_k(filtered_predicted, filtered_expected, k=5)

        return {
            "map": map_score,
            "mrr": mrr_score,
            "p@5": p_at_5,
        }

    def evaluate_list_questions(self):
        """
        Evaluate all List questions.
        Includes standard macro/micro F1 as well as:
        - full match (text-only)
        - letter match (A, B, C...)
        - mixed match (letter + text both correct)
        **List (multi-select)** - token-level  **Macro & Micro F1**  
        Invalid predictions (e.g., text not matching any shown options) are counted separately.
        """

        try:
            blob = self.output_data.get("list")
            if not blob:
                print("No List output data found.", flush=True)
                return {
                    "average": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1_score": 0.0,
                    "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1_score": 0.0,
                    "list_full_match": 0.0, "list_letter_match": 0.0, "list_mixed_match": 0.0,
                    "list_invalid_count":0, "scores": {}
                }

            inputs = blob.get("inputs", [])
            predictions = blob.get("responses", [])

            if len(inputs) != len(predictions):
                print("Mismatch in number of inputs and predictions.", flush=True)
                return {
                    "average": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1_score": 0.0,
                    "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1_score": 0.0,
                    "list_full_match": 0.0, "list_letter_match": 0.0, "list_mixed_match": 0.0,
                    "list_invalid_count":0, "scores": {}
                }

            results = {}
            precision_list = []
            recall_list = []
            f1_list = []
            invalid_count = 0
            global_tp, global_fp, global_fn = 0, 0, 0

            match_counts = {"full": 0, "letter": 0, "mixed": 0, "total": 0}

            for gold, raw_pred in zip(inputs, predictions):
                try:
                    options_raw = gold.get("options", [])
                    options = [self.normalise(opt) for opt in options_raw]

                    expected_values = [self.normalise(v) for v in gold.get("answer", [])]

                    # -------------------- these steps have been added in to ensure that any issues with the list type questions where the letter options are 
                    # -------------------- evident within the answer options (WHICH SHOULD HAVE BEEN PICKED UP BY THE GOD DAM FUCKING ANNOTATORS) are fixed / scrubbed
                    # -------------------- NOTE - this is some hamburger ass code - this is just a workaround to temp fix the issue. To really fix this, we will need to 
                    # -------------------- go back through the original data and remove any of these malformations from the dataset (that REALLY the annotators should have picked up)
                    options_stripped = [
                        re.sub(r"^(?:[A-H]\. |[A-H]: )", "", opt).strip()
                        for opt in options_raw
                    ]

                    # now build your clean maps off of options_stripped
                    letter_to_text = {
                        chr(65 + i): options_stripped[i]
                        for i in range(len(options_stripped))
                    }
                    text_to_letter = {
                        self.normalise(txt): ltr
                        for ltr, txt in letter_to_text.items()
                    }
                    # and rebuild your expected_values from gold["answer"]:
                    expected_clean = []
                    for v in gold.get("answer", []):
                        v = v.strip()
                        # if it’s just a letter, map it:
                        if len(v) == 1 and v.upper() in letter_to_text:
                            expected_clean.append(letter_to_text[v.upper()])
                        else:
                            # maybe it’s "c prostatectomy" — drop the leading "c "
                            m = re.match(r"^([A-Za-z])[\.:]?\s+(.+)$", v)
                            if m and m.group(1).upper() in letter_to_text:
                                expected_clean.append(m.group(2))
                            else:
                                # otherwise assume it really is the full text already
                                expected_clean.append(v)
                    # finally normalize
                    expected_values = [self.normalise(x) for x in expected_clean]

                    # Normalize & parse the raw response by first splitting on commas
                    raw_resp = self._to_text(raw_pred)

                    # Match type
                    match_type, malformed, predicted_values, unmatched_segments = self._list_match_letter_or_text(
                                                                        expected_values, 
                                                                        raw_resp, 
                                                                        letter_to_text, 
                                                                        options_stripped
                                                                    )
                    # if it was malformed, immediately zero everything out
                    if malformed:
                        invalid_count += 1
                        # Normalize unmatched segments and treat as predicted
                        hallucinated = {self.normalise(seg) for seg, norm in unmatched_segments}
                        pred_set = hallucinated
                        true_set = set(expected_values)

                        tp = 0  # no correct predictions allowed in malformed
                        fp = len(pred_set)
                        fn = len(true_set)

                        global_tp += tp
                        global_fp += fp
                        global_fn += fn

                        precision = 0.0 if fp + tp == 0 else tp / (tp + fp)
                        recall = 0.0 if fn + tp == 0 else tp / (tp + fn)
                        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

                        precision_list.append(precision)
                        recall_list.append(recall)
                        f1_list.append(f1)

                    else:
                        # count only the non‐malformed
                        if match_type in match_counts:
                            match_counts[match_type] += 1
                        match_counts["total"] += 1

                        # Compute per-item precision/recall/F1 and build pred_set
                        if match_type == "full":
                            precision, recall, f1 = self.compute_overlap_f1(expected_values, predicted_values)
                            pred_set = set(predicted_values)
                        elif match_type == "letter":
                            # expected_letters = [
                            #     ltr for ltr, opt in letter_to_text.items()
                            #     if self.normalise(opt) in expected_values
                            # ]
                            # pred_letters = [v.upper() for v in predicted_values if v.upper() in letter_to_text]
                            # pred_set = set(pred_letters)
                            # precision, recall, f1 = self.compute_overlap_f1(expected_letters, pred_letters)
                            pred_set = set(predicted_values)
                            precision, recall, f1 = self.compute_overlap_f1(expected_values, predicted_values)

                        elif match_type == "mixed":
                            mapped = []
                            for v in predicted_values:
                                v = v.strip()
                                if ":" in v:
                                    parts = v.split(":", 1)
                                    if len(parts) != 2:
                                        continue  # Malformed
                                    letter = parts[0].strip().upper()
                                    if letter in letter_to_text:
                                        mapped_text = letter_to_text[letter].strip().lower()
                                        mapped.append(mapped_text)
                                elif len(v) == 1 and v.upper() in letter_to_text:
                                    mapped_text = letter_to_text[v.upper()].strip().lower()
                                    mapped.append(mapped_text)
                                else:
                                    mapped.append(v.strip().lower())

                            if not mapped:
                                precision, recall, f1 = 0.0, 0.0, 0.0
                                pred_set = set()
                            else:
                                precision, recall, f1 = self.compute_overlap_f1(expected_values, mapped)
                                pred_set = set(mapped)
                            
                        else:
                            # even if it didn’t qualify as a “full”/“letter”/“mixed” tidy match,
                            # still compute overlap F1 on whatever you found
                            precision, recall, f1 = self.compute_overlap_f1(expected_values, predicted_values)
                            pred_set = set(predicted_values)

                    # Update global TP/FP/FN
                    true_set = set(expected_values)
                    tp = len(true_set & pred_set)
                    fp = len(pred_set - true_set)
                    fn = len(true_set - pred_set)
                    global_tp += tp
                    global_fp += fp
                    global_fn += fn

                    # Record this qa pair 
                    para_id = gold.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question":   gold.get("question", ""),
                        "expected":   expected_values,
                        "raw response": raw_resp,
                        "predicted":  (predicted_values if predicted_values else [raw_resp]),
                        "options":    options,
                        "precision":  precision,
                        "recall":     recall,
                        "f1_score":   f1,
                        "match_type": match_type,
                        "malformed":  malformed,
                        "pred_set":   sorted(pred_set),
                    }

                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_list.append(f1)
                except Exception as inner_e:
                    print(f"Error processing List QA: {inner_e}", flush=True)

            # Ground truth and prediction sets for all examples
            expected_sets_all = [set(r["expected"]) for r in results.values()]
            predicted_sets_all = [set(r["pred_set"]) for r in results.values()]

            # F1 scores: overall
            f1_scores_all = self.compute_macro_micro_metrics(expected_sets_all, predicted_sets_all)
            avg_precision = f1_scores_all["macro_precision"]
            avg_recall = f1_scores_all["macro_recall"]
            avg_f1 = f1_scores_all["macro_f1_score"]
            micro_precision = f1_scores_all["micro_precision"]
            micro_recall = f1_scores_all["micro_recall"]
            micro_f1 = f1_scores_all["micro_f1_score"]

            # Full-match only: keep the model’s pred_set for full matches, empty set elsewhere
            predicted_full_masked = [
                r["pred_set"] if r["match_type"] == "full" else set()
                for r in results.values()
            ]
            f1_scores_full = self.compute_macro_micro_metrics(expected_sets_all, predicted_full_masked)

            full_match_macro_precision = f1_scores_full["macro_precision"]
            full_match_macro_recall = f1_scores_full["macro_recall"]
            full_match_macro_f1 = f1_scores_full["macro_f1_score"]

            full_match_micro_precision = f1_scores_full["micro_precision"]
            full_match_micro_recall = f1_scores_full["micro_recall"]
            full_match_micro_f1 = f1_scores_full["micro_f1_score"]

            # Letter+Mixed only: keep the model’s pred_set for letter or mixed, empty set elsewhere
            predicted_letter_and_mixed_masked = [
                r["pred_set"] if r["match_type"] in {"letter","mixed"} else set()
                for r in results.values()
            ]
            f1_scores_combined = self.compute_macro_micro_metrics(expected_sets_all, predicted_letter_and_mixed_masked)

            combined_match_macro_precision = f1_scores_combined["macro_precision"]
            combined_match_macro_recall = f1_scores_combined["macro_recall"]
            combined_match_macro_f1 = f1_scores_combined["macro_f1_score"]

            combined_match_micro_precision = f1_scores_combined["micro_precision"]
            combined_match_micro_recall = f1_scores_combined["micro_recall"]
            combined_match_micro_f1 = f1_scores_combined["micro_f1_score"]

            # Optional IR-style metrics
            predicted_ranked_lists = [r["predicted"] for r in results.values()]
            gold_ranked_lists = [r["expected"] for r in results.values()]

            metrics_all = self.compute_additional_ir_metrics(
                                                                expected_sets_all, 
                                                                predicted_ranked_lists, 
                                                                results
                                                                )
            
            metrics_full = self.compute_additional_ir_metrics(
                                                                expected_sets_all, 
                                                                predicted_ranked_lists, 
                                                                results, 
                                                                match_type_filter={"full"}
                                                                )
            
            metrics_letter_mixed = self.compute_additional_ir_metrics(
                                                                        expected_sets_all,
                                                                        predicted_ranked_lists,
                                                                        results,
                                                                        match_type_filter={"letter","mixed"}
                                                                    )
            total_non_malformed = match_counts['total']
            strict_matches      = match_counts['full'] + match_counts['letter'] + match_counts['mixed']
            no_matches          = total_non_malformed - strict_matches
            match_rate          = (strict_matches / total_non_malformed) if total_non_malformed else 0.0

            print("\n==================== List QA Evaluation Summary ====================")
            print("→ Overall (all examples):")
            print(f"  Macro Avg Precision:     {avg_precision:.4f}")
            print(f"  Macro Avg Recall:        {avg_recall:.4f}")
            print(f"  Macro Avg F1 Score:      {avg_f1:.4f}")
            print(f"  Micro Precision:         {micro_precision:.4f}")
            print(f"  Micro Recall:            {micro_recall:.4f}")
            print(f"  Micro F1 Score:          {micro_f1:.4f}")
            print("\n==================== NOTE ====================")
            print("  NOTE - the below metrics (looking at full and combined matches) takes out No matches (model gave response(s) from options provided but none were right) from the response set.")
            print(" So the numbers reported for full match + combined <> overall, as they do not include the (negative) results for when the model provided no matches (for non-malformed responses)")
            print(" The Overall (all examples) metric should be used to evaluate the list type QA's, the below are just provided for internal testing to identify models that ")
            print(" provided letter / mixed responses when they explicitly told not to! ")
            print(" If later on, the full text matches are used to evaluate list then the non matches (negative results) MUST be added back into the set prior to computing f1 metrics!!! ")
            print("\n==================== NOTE ====================")
            print("\n→ Full-Match Only:")
            print(f"  Macro Avg Precision:     {full_match_macro_precision:.4f}")
            print(f"  Macro Avg Recall:        {full_match_macro_recall:.4f}")
            print(f"  Macro Avg F1 Score:      {full_match_macro_f1:.4f}")
            print(f"  Micro Precision:         {full_match_micro_precision:.4f}")
            print(f"  Micro Recall:            {full_match_micro_recall:.4f}")
            print(f"  Micro F1 Score:          {full_match_micro_f1:.4f}")

            print("\n→ Combined Match (Letter + Mixed):")
            print(f"  Macro Avg Precision:     {combined_match_macro_precision:.4f}")
            print(f"  Macro Avg Recall:        {combined_match_macro_recall:.4f}")
            print(f"  Macro Avg F1 Score:      {combined_match_macro_f1:.4f}")
            print(f"  Micro Precision:         {combined_match_micro_precision:.4f}")
            print(f"  Micro Recall:            {combined_match_micro_recall:.4f}")
            print(f"  Micro F1 Score:          {combined_match_micro_f1:.4f}")

            print("\n→ Match Type Distribution:")
            print(f"  Total non-malformed responses: {total_non_malformed}")
            print(f"  Strict matches (full/letter/mixed): {strict_matches}")
            print(f"  No matches (model gave response(s) from options provided but none were right):                       {no_matches}")
            print(f"  Strict match rate:                {match_rate:.2%}")

            print(f"  FULL   Match Rate:       {match_counts['full'] / match_counts['total'] if match_counts['total'] else 0.0:.4f}")
            print(f"  LETTER Match Rate:       {match_counts['letter'] / match_counts['total'] if match_counts['total'] else 0.0:.4f}")
            print(f"  MIXED  Match Rate:       {match_counts['mixed'] / match_counts['total'] if match_counts['total'] else 0.0:.4f}")
            print(f"  INVALID Predictions:     {invalid_count} / {len(inputs)}", flush=True)

            print("\n→ IR-Style Metrics:")
            print(f"  MAP (All):               {metrics_all['map']:.4f}")
            print(f"  MRR (All):               {metrics_all['mrr']:.4f}")
            print(f"  P@5 (All):               {metrics_all['p@5']:.4f}")

            print(f"  MAP (Full):              {metrics_full['map']:.4f}")
            print(f"  MRR (Full):              {metrics_full['mrr']:.4f}")
            print(f"  P@5 (Full):              {metrics_full['p@5']:.4f}")

            print(f"  MAP (Combined):          {metrics_letter_mixed['map']:.4f}")
            print(f"  MRR (Combined):          {metrics_letter_mixed['mrr']:.4f}")
            print(f"  P@5 (Combined):          {metrics_letter_mixed['p@5']:.4f}")

            # Aggregate match-type stats
            return {
                "list_overall_macro_avg_precision": avg_precision,
                "list_overall_macro_avg_recall": avg_recall,
                "list_overall_macro_avg_f1_score": avg_f1,
                "list_overall_micro_precision": micro_precision,
                "list_overall_micro_recall": micro_recall,
                "list_overall_micro_f1_score": micro_f1,
                "list_full_match_macro_precision": full_match_macro_precision,
                "list_full_match_macro_recall": full_match_macro_recall,
                "list_full_match_macro_f1_score": full_match_macro_f1,
                "list_full_match_micro_precision": full_match_micro_precision,
                "list_full_match_micro_recall": full_match_micro_recall,
                "list_full_match_micro_f1_score": full_match_micro_f1,
                "list_combined_match_macro_precision": combined_match_macro_precision,
                "list_combined_match_macro_recall": combined_match_macro_recall,
                "list_combined_match_macro_f1_score": combined_match_macro_f1,
                "list_combined_match_micro_precision": combined_match_micro_precision,
                "list_combined_match_micro_recall": combined_match_micro_recall,
                "list_combined_match_micro_f1_score": combined_match_micro_f1,
                "list_full_match": match_counts["full"] / match_counts["total"] if match_counts["total"] else 0.0,
                "list_letter_match": match_counts["letter"] / match_counts["total"] if match_counts["total"] else 0.0,
                "list_mixed_match": match_counts["mixed"] / match_counts["total"] if match_counts["total"] else 0.0,
                "list_invalid_count": invalid_count,
                "list_map": metrics_all["map"],
                "list_mrr": metrics_all["mrr"],
                "list_p_at_5": metrics_all["p@5"],
                "list_map_full": metrics_full["map"],
                "list_mrr_full": metrics_full["mrr"],
                "list_p_at_5_full": metrics_full["p@5"],
                "list_map_combined": metrics_letter_mixed["map"],
                "list_mrr_combined": metrics_letter_mixed["mrr"],
                "list_p_at_5_combined": metrics_letter_mixed["p@5"],
                "scores": results,
            }

        except Exception as e:
            print(f"Error evaluating List questions: {e}", flush=True)
            return {
                "average": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1_score": 0.0,
                "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1_score": 0.0,
                "list_full_match": 0.0, "list_letter_match": 0.0, "list_mixed_match": 0.0,
                "list_invalid_count":0, "scores": {}
            }


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
                    raw_resp = self._to_text(pred)
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
                        "raw response": raw_resp,
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
                    raw_resp = self._to_text(pred)
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
                        "raw response": raw_resp,
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
                    raw_resp = self._to_text(pred)
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
                        "raw response": raw_resp,
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
                    raw_resp = self._to_text(pred)
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
                        "raw response":          raw_resp,
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
            output_dir = os.path.join(self.base_dir, self.output_dir) if not os.path.isabs(self.output_dir) else self.output_dir
            os.makedirs(output_dir, exist_ok=True)

            overall_results = OrderedDict()

            # # Evaluate each QA type individually
            # tf_results = self.evaluate_true_false_questions()
            # mc_results = self.evaluate_multiple_choice_questions()
            # list_results = self.evaluate_list_questions()
            # short_results = self.evaluate_short_questions()
            # short_inverse_results = self.evaluate_short_inverse_questions()
            # multi_hop_results = self.evaluate_multi_hop_questions()
            # multi_hop_inverse_results = self.evaluate_multi_hop_inverse_questions()

            # # ---------------------- individual json files ----------------------
            # # Organize and save grouped results
            # grouped_outputs = {
            #     "true_false_results.json": {
            #         "true_false": tf_results
            #     },
            #     "multiple_choice_results.json": {
            #         "multiple_choice": mc_results
            #     },
            #     "list_results.json": {
            #         "list": list_results
            #     },
            #     "short_results.json": {
            #         "short": short_results,
            #     },
            #     "multi_hop_results.json": {
            #         "multi_hop": multi_hop_results,
            #     },
            #     "short_inverse_results.json": {
            #         "short_inverse": short_inverse_results
            #     },
            #     "multi_hop_inverse_results.json": {
            #         "multi_hop_inverse": multi_hop_inverse_results
            #     }
            # }

            # for filename, data in grouped_outputs.items():
            #     # Inject reasoning steps if they exist
            #     for task_name, task_data in data.items():
            #         original_blob = self.output_data.get(task_name, {})
            #         if "reasoning_model_thinking_steps" in original_blob:
            #             task_data["reasoning_model_thinking_steps"] = original_blob["reasoning_model_thinking_steps"]

            #     file_path = os.path.join(output_dir, filename)
            #     with open(file_path, "w") as f:
            #         json.dump(data, f, indent=4)
            #     print(f"Saved {filename}", flush=True)

            # mapping: (filename, top‐level key, evaluator method)

            # True/False
            tf_results = self.evaluate_true_false_questions()
            with open(os.path.join(output_dir, "true_false_results.json"), "w") as f:
                json.dump({"true_false": tf_results}, f, indent=4)
            print("Saved true_false_results.json", flush=True)

            # Multiple-Choice
            mc_results = self.evaluate_multiple_choice_questions()
            with open(os.path.join(output_dir, "multiple_choice_results.json"), "w") as f:
                json.dump({"multiple_choice": mc_results}, f, indent=4)
            print("Saved multiple_choice_results.json", flush=True)

            # List
            list_results = self.evaluate_list_questions()
            with open(os.path.join(output_dir, "list_results.json"), "w") as f:
                json.dump({"list": list_results}, f, indent=4)
            print("Saved list_results.json", flush=True)

            # Short
            short_results = self.evaluate_short_questions()
            with open(os.path.join(output_dir, "short_results.json"), "w") as f:
                json.dump({"short": short_results}, f, indent=4)
            print("Saved short_results.json", flush=True)

            # Short-Inverse
            short_inverse_results = self.evaluate_short_inverse_questions()
            with open(os.path.join(output_dir, "short_inverse_results.json"), "w") as f:
                json.dump({"short_inverse": short_inverse_results}, f, indent=4)
            print("Saved short_inverse_results.json", flush=True)

            # Multi-Hop
            multi_hop_results = self.evaluate_multi_hop_questions()
            with open(os.path.join(output_dir, "multi_hop_results.json"), "w") as f:
                json.dump({"multi_hop": multi_hop_results}, f, indent=4)
            print("Saved multi_hop_results.json", flush=True)

            # Multi-Hop Inverse
            multi_hop_inverse_results = self.evaluate_multi_hop_inverse_questions()
            with open(os.path.join(output_dir, "multi_hop_inverse_results.json"), "w") as f:
                json.dump({"multi_hop_inverse": multi_hop_inverse_results}, f, indent=4)
            print("Saved multi_hop_inverse_results.json", flush=True)

            # ---------------------- individual json files ----------------------

            # ---------------------- Compute match-type metrics early ----------------------
            mc_match_metrics = {
                "full_match": mc_results.get("multiple_choice_full_match", 0.0),
                "letter_match": mc_results.get("multiple_choice_letter_match", 0.0),
                "mixed_match": mc_results.get("multiple_choice_mixed_match", 0.0),
            }

            list_match_metrics = {
                "full_match": list_results.get("list_full_match", 0.0),
                "letter_match": list_results.get("list_letter_match", 0.0),
                "mixed_match": list_results.get("list_mixed_match", 0.0),
            }

            step_id_score = multi_hop_inverse_results.get("step_identification_rate", 0.0)


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

            # --------------- Compute overall average of those three across all open‑ended types ---------------
            overall_open = {
                "avg_bleu":   sum(m["avg_bleu"]   for m in open_ended_metrics.values()) / len(open_ended_metrics),
                "avg_meteor": sum(m["avg_meteor"] for m in open_ended_metrics.values()) / len(open_ended_metrics),
                "avg_rouge":  sum(m["avg_rouge"]  for m in open_ended_metrics.values()) / len(open_ended_metrics),
            }


            # --------------- gather open‑ended similarity distributions ---------------
            open_ended_scores = {
                "short":           [v["semantic_match_score"] for v in short_results["scores"].values()],
                "short_inverse":   [v["semantic_match_score"] for v in short_inverse_results["scores"].values()],
                "multi_hop":       [v["semantic_match_score"] for v in multi_hop_results["scores"].values()],
                "multi_hop_inverse":[v["semantic_match_score"] for v in multi_hop_inverse_results["scores"].values()],
            }

            # --------------- compute single “overall effectiveness” score ---------------
            # Individual top-level scores (order-preserved)
            tf_acc   = tf_results.get("accuracy", 0.0)
            mc_acc   = mc_results.get("accuracy", 0.0)
            list_f1  = list_results.get("macro_f1_score", 0.0)
            short_ss = short_results.get("semantic_match_score", 0.0)
            short_inv_ss = short_inverse_results.get("semantic_match_score", 0.0)
            hop_ss   = multi_hop_results.get("semantic_match_score", 0.0)
            hop_inv_ss = multi_hop_inverse_results.get("semantic_match_score", 0.0)

            # to_avg = [tf_acc, mc_acc, list_f1, short_ss, short_inv_ss, hop_ss, hop_inv_ss]
            ## overall_score = float(np.mean(to_avg))

            # Add per-type and total invalid counts for close-ended types
            tf_invalid   = tf_results.get("true_false_invalid_count", 0)
            mc_invalid   = mc_results.get("multiple_choice_invalid_count", 0)
            list_invalid = list_results.get("list_invalid_count", 0)

            
            to_avg = [
                tf_results.get("accuracy", 0.0),
                mc_results.get("multiple_choice_average", 0.0),
                list_results.get("list_overall_macro_avg_f1_score", 0.0),
                short_results.get("semantic_match_score", 0.0),
                short_inverse_results.get("semantic_match_score", 0.0),
                multi_hop_results.get("semantic_match_score", 0.0),
                multi_hop_inverse_results.get("semantic_match_score", 0.0),
            ]
            overall_score = float(np.mean(to_avg))
            
            print(f"Overall effectiveness score: {overall_score:.3f}", flush=True)
            # ---------------------------------------------------------------------------
            
            # -------------------------- print to ordered json dict ----------------------

            overall_results["overall_effectiveness"]            = overall_score
            overall_results["true_false_accuracy"]              = tf_acc
            overall_results["multiple_choice_accuracy"]         = mc_acc
            overall_results["list_macro_f1_score"]              = list_f1

            overall_results["total_invalid_close_ended"]       = tf_invalid + mc_invalid + list_invalid
            overall_results["true_false_invalid_count"]        = tf_invalid
            overall_results["multiple_choice_invalid_count"]   = mc_invalid
            overall_results["list_invalid_count"]              = list_invalid
            overall_results["short_semantic_match_score"]      = short_ss
            overall_results["short_inverse_semantic_match_score"] = short_inv_ss
            overall_results["multi_hop_semantic_match_score"]   = hop_ss
            overall_results["multi_hop_inverse_semantic_match_score"] = hop_inv_ss

            overall_results["multiple_choice_match_metrics"]    = mc_match_metrics
            overall_results["list_match_metrics"]               = list_match_metrics
            overall_results["multi_hop_inverse_step_identification_rate"] = step_id_score

            overall_results["multiple_choice_average"] = mc_results.get("multiple_choice_average", 0.0)
            overall_results["multiple_choice_full_match_average"] = mc_results.get("multiple_choice_full_match_average", 0.0)
            overall_results["multiple_choice_combined_match_average"] = mc_results.get("multiple_choice_combined_match_average", 0.0)
            overall_results["multiple_choice_full_match"] = mc_results.get("multiple_choice_full_match", 0.0)
            overall_results["multiple_choice_letter_match"] = mc_results.get("multiple_choice_letter_match", 0.0)
            overall_results["multiple_choice_mixed_match"] = mc_results.get("multiple_choice_mixed_match", 0.0)

            overall_results["list_overall_macro_avg_precision"] = list_results.get("list_overall_macro_avg_precision", 0.0)
            overall_results["list_overall_macro_avg_recall"] = list_results.get("list_overall_macro_avg_recall", 0.0)
            overall_results["list_overall_macro_avg_f1_score"] = list_results.get("list_overall_macro_avg_f1_score", 0.0)
            overall_results["list_overall_micro_precision"] = list_results.get("list_overall_micro_precision", 0.0)
            overall_results["list_overall_micro_recall"] = list_results.get("list_overall_micro_recall", 0.0)
            overall_results["list_overall_micro_f1_score"] = list_results.get("list_overall_micro_f1_score", 0.0)
            overall_results["list_full_match_macro_precision"] = list_results.get("list_full_match_macro_precision", 0.0)
            overall_results["list_full_match_macro_recall"] = list_results.get("list_full_match_macro_recall", 0.0)
            overall_results["list_full_match_macro_f1_score"] = list_results.get("list_full_match_macro_f1_score", 0.0)
            overall_results["list_full_match_micro_precision"] = list_results.get("list_full_match_micro_precision", 0.0)
            overall_results["list_full_match_micro_recall"] = list_results.get("list_full_match_micro_recall", 0.0)
            overall_results["list_full_match_micro_f1_score"] = list_results.get("list_full_match_micro_f1_score", 0.0)
            overall_results["list_combined_match_macro_precision"] = list_results.get("list_combined_match_macro_precision", 0.0)
            overall_results["list_combined_match_macro_recall"] = list_results.get("list_combined_match_macro_recall", 0.0)
            overall_results["list_combined_match_macro_f1_score"] = list_results.get("list_combined_match_macro_f1_score", 0.0)
            overall_results["list_combined_match_micro_precision"] = list_results.get("list_combined_match_micro_precision", 0.0)
            overall_results["list_combined_match_micro_recall"] = list_results.get("list_combined_match_micro_recall", 0.0)
            overall_results["list_combined_match_micro_f1_score"] = list_results.get("list_combined_match_micro_f1_score", 0.0)
            overall_results["list_full_match"] = list_results.get("list_full_match", 0.0)
            overall_results["list_letter_match"] = list_results.get("list_letter_match", 0.0)
            overall_results["list_mixed_match"] = list_results.get("list_mixed_match", 0.0)

            overall_results["open_ended_metrics_per_type"]      = open_ended_metrics
            overall_results["open_ended_metrics_overall"]       = overall_open

            # Include match-type accuracy for MC and List
            # overall_results["multiple_choice_match_metrics"] = {
            #     "full_match": mc_results.get("multiple_choice_full_match", 0.0),
            #     "letter_match": mc_results.get("multiple_choice_letter_match", 0.0),
            #     "mixed_match": mc_results.get("multiple_choice_mixed_match", 0.0),
            # }

            # overall_results["list_match_metrics"] = {
            #     "full_match": list_results.get("list_full_match", 0.0),
            #     "letter_match": list_results.get("list_letter_match", 0.0),
            #     "mixed_match": list_results.get("list_mixed_match", 0.0),
            # }

            # overall_results["multi_hop_inverse_step_identification_rate"] = \
            #     multi_hop_inverse_results.get("step_identification_rate", 0.0)

             # Merge everything for overall results
            overall_results["true_false"]        = tf_results
            overall_results["multiple_choice"]   = mc_results
            overall_results["list"]              = list_results
            overall_results["short"]             = short_results
            overall_results["short_inverse"]     = short_inverse_results
            overall_results["multi_hop"]         = multi_hop_results
            overall_results["multi_hop_inverse"] = multi_hop_inverse_results

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

    parser.add_argument(
        "--output_dir",
        default="evaluation_output",
        help="Where to save evaluation results and plots (default: evaluation_output/)"
    )


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluator = ClinIQLinkSampleDatasetEvaluate(run_mode=args.mode, results_dir=args.results_dir, bin_width=args.bin_width, output_dir=args.output_dir)
    evaluator.run_all_evaluations()