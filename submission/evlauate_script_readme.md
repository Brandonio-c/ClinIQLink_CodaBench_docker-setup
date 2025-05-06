The open‐ended semantic similarity evaluator in this script is built around the insight that simple n-gram overlaps (BLEU, ROUGE, METEOR) often fail to capture true meaning equivalence—two perfectly valid paraphrases might share few exact words yet convey the same idea. To overcome this, our method layers three complementary cosine‐based similarity measures (word, sentence, paragraph) and combines them into a single, interpretable “semantic match score.” Here’s how it all comes together:

---

### 1. Data Preparation & IDF Construction

Before any similarity can be computed, the evaluator must know which terms in your domain are informative and which are “noise.”

1. **Loading Outputs**

   * When you instantiate the evaluator, it loads each of the seven JSON files (true/false, multiple-choice, list, short, etc.) into `self.output_data`. These files each contain an `"inputs"` list (with ground-truth answers) and a `"responses"` list (with model outputs), aligned one-to-one.

2. **Stop-word & Punctuation Setup**

   * We pull in NLTK’s English stop-word list plus Python’s `string.punctuation` to form two sets:

     * `_STOP_WORDS` (very common function words like “the,” “is,” “on”)
     * `_PUNCT` (all punctuation characters)
   * Later, when comparing two free-text answers, we strip out any token that is *entirely* a stop-word or punctuation. This ensures the similarity layers focus on content-bearing words (e.g. “hypertension,” “Cardiomegaly”) rather than uninformative fillers.

3. **Collecting All References**

   * We sweep through each loaded blob’s `"inputs"` and extract every ground-truth answer into a single list `all_refs`. If an answer is already a list (e.g. multi-part answers), we join it into a single string. This full pool of reference texts is then used to build document frequencies.

4. **IDF via WordPiece Tokenizer**

   * Using the same WordPiece tokenizer that SBERT employs (via HuggingFace’s `AutoTokenizer`), we tokenize each reference answer into subword units (e.g. “cardio,” “megaly,” “##pathy”).
   * We count in how many *distinct* reference documents each subword appears (the document frequency, df), and compute the inverse‐document‐frequency for each token as

     ```
       idf(tok) = log((N + 1)/(df + 1)) + 1
     ```

     where $N$ is the total number of reference answers. Rare terms (low df) get higher IDF, making them more influential in the final word-level score.

---

### 2. Word-Level Similarity (40 %)

At the heart of our fine-grained similarity is an *IDF-weighted greedy alignment* of individual subword embeddings.

1. **Token Filtering**

   * We tokenize the *cleaned* expected & predicted strings into WordPiece tokens but only keep those that are purely alphanumeric (discarding stray punctuation leftover).

2. **Embedding & Alignment**

   * Both token lists are passed through SBERT in batch, yielding an embedding vector for each token.
   * We compute a full cosine‐similarity matrix $S$ of shape $(|\text{pred}|,|\text{exp}|)$.
   * For each predicted token, we take the maximum similarity across all expected tokens (this simulates “precision”—how well each predicted token found a match). Likewise, for each expected token, we take its best match among predicted tokens (“recall”).

3. **IDF-Weighted Precision & Recall → F1**

   * We weight each “precision” sim by the token’s IDF in the prediction, and analogously weight each “recall” sim by the token’s IDF in the expected text.
   * Summing and normalizing by the total IDF produces a *soft precision* and *soft recall*, which are then combined into an F1 score:

     $$
       F1 = 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}
     $$
   * If either token list is empty (no valid tokens), we fall back to 0.0.

---

### 3. Sentence-Level Similarity (40 %)

While word-level F1 captures fine lexical correspondences, it can miss the overall coherence of longer paraphrases. Sentence‐level similarity remedies this:

1. **Full-Sentence Embedding**

   * We ask SBERT to encode the entire cleaned sentence (answer or prediction) as one vector (the pooled “CLS” embedding).

2. **Cosine Comparison**

   * A single cosine similarity between these two vectors serves as a holistic measure of how semantically close the two sentences are, even if they share few exact tokens.

---

### 4. Paragraph-Level Similarity (20 %)

To capture even broader context and to guard against SBERT’s intrinsic similarity floor (it tends to give unrelated texts a 0.25–0.35 score), we add a paragraph‐level check:

1. **Raw-Text Encoding**

   * Here we encode the *uncleaned* strings—preserving stop-words and punctuation—so that SBERT’s internal pretraining priors (on full paragraphs) can come into play.

2. **Baseline Adjustment**

   * At initialization, we randomly sample 100 reference pairs that should be *unrelated*, compute their SBERT cosine mean (`para_baseline`), and then adjust raw paragraph similarities $r$ via

     $$
       \text{adjusted} = \frac{r - \text{baseline}}{1 - \text{baseline}}
     $$
   * This maps the expected floor (\~0.3) down to zero—so two truly unrelated paragraphs score 0, not 0.3.

---

### 5. Combining, Clamping & Thresholding

Once we have three scores—word\_F1, sent\_sim, para\_sim—we form a weighted sum:

$$
  \text{semantic}_\text{raw} = 0.4 \times \text{word} + 0.4 \times \text{sentence} + 0.2 \times \text{paragraph}.
$$

To eliminate any residual SBERT bias, we subtract 0.25 (the approximate similarity of unrelated pairs) and floor at zero:

$$
  \text{semantic} = \max(0,\ \text{semantic}_\text{raw} - 0.25).
$$

Finally, to preserve perfect matches, any score ≥ 0.95 is snapped up to 1.0.

---

### 6. Why It Works & How It Compares

* **Robustness to Paraphrase:** By using embeddings, even totally rephrased answers (“Myocardial infarction” vs. “Heart attack”) score highly, whereas n-gram methods would give zero BLEU or ROUGE.
* **IDF Emphasis on Rarer, Crucial Terms:** In clinical contexts, missing “pulmonary” vs. “systemic” can be critical. Our IDF ensures that alignment on rare but meaningful tokens carries more weight than frequent conjunctions.
* **Multi-Scale Context:** Word-level F1 catches precise concept overlaps; sentence-level captures overall gist; paragraph-level anchors against spurious high scores on very short answers.
* **Bias Normalization:** Adjusting by an empirical SBERT baseline prevents “false positives” where unrelated texts still yield moderate cosine scores.

In practice, this layered approach achieves a far closer correlation with human judgments of answer quality than BLEU, METEOR or ROUGE alone—especially in open-domain, free-text QA where lexical variability is high.
