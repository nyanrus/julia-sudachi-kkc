# Julia KKC (Kana-Kanji Conversion) Project Analysis

## Project Overview
A complete Kana-Kanji Conversion system in Julia implementing:
- Dictionary-based morphological analysis (Viterbi)
- Averaged perceptron training for connection costs
- Kneser-Ney language modeling (surface + class-based)
- Brown hierarchical clustering for semantic transitions
- Context-aware decoding with cache language model

---

## 1. SOURCE FILES SUMMARY

### 1.1 brown_clustering.jl (191 lines)
**Purpose**: Semantic word clustering for O(1) transition scoring

**Key Data Structure**:
- `BrownClusters` — Maps word surfaces → cluster IDs (UInt16)

**Main Function**:
- `brown_cluster(training_pairs, n_clusters; min_freq, max_vocab) → BrownClusters`
  - Greedy agglomerative clustering
  - Algorithm:
    1. Count unigrams/bigrams from gold training paths
    2. Keep top-V words by frequency (default V=8000, min_freq=3)
    3. Assign initial n_clusters words to separate clusters
    4. Remaining words added incrementally to closest cluster by PMI affinity
  - Scores cluster affinity using pointwise mutual information (PMI):
    ```
    PMI(w, c) = log(c(w→cw) * N / (count(w) * count(cw)))
    ```
  - BOS/EOS get cluster 0 (reserved)
  - Used at Viterbi time: `W_sem[cluster_prev, cluster_curr]` matrix for semantic bonuses

**Key Helper**:
- `get_cluster(bc, surface) → UInt16` — O(1) lookup, returns 0 for unknown

---

### 1.2 corpus_loaders.jl (430 lines)
**Purpose**: Parse raw training corpora (SNOW, Tanaka, KNBC) with binary cache

**Corpus Types Supported**:
- `:snow` — SNOW T15/T23 CC-BY datasets (CSV)
- `:tanaka` — Tanaka Corpus (CSV or TSV)
- `:knbc` — KNBコーパス flat surface strings
- `:knbc_bunsetsu` — KNBコーパス with gold bunsetsu boundaries (tab-separated)

**Binary Cache Format**:
```
[CORPUS_MAGIC(4)] [VERSION(4)]
[n_sentences(8)] [heap_bytes(8)]
[offsets: n×UInt64]
[lengths: n×UInt32]
[heap: concatenated UTF-8 sentence bytes]
```

**Key Functions**:
- `load_snow(path)` — CSV parser, 原文 + やさしい columns
- `load_tanaka(path)` — CSV or TSV (ID, EN, JP format)
- `parse_knbc_file(path)` → Vec<(surface, reading)> tuples
  - Parses KNP format: lines are `surface reading base pos...`
  - Bunsetsu boundaries marked with `* nD`, tags with `+ mD`
  - Returns (surface, reading) pairs per bunsetsu
- `load_knbc_bunsetsu(dir)` → Vec<"surf1\tsurf2\t..."> (tab-separated chunks)
- `load_or_build_corpus(sources; bin_path, force_rebuild)` — Unified entry point with mtime checking

**POS Bigram Learning**:
- `learn_pos_bigrams(csv_paths, knbc_dir; scale=800f0, λ_pos=1.0f0) → POSBigram`
  - Maps KNP POS tags → Sudachi POS using `_KNP_TO_SUDACHI` table
  - Counts bigrams: `bigram[prev_id, curr_id]` from KNBC annotations
  - Computes Laplace-smoothed -log P(curr|prev) × scale
  - BOS and EOS get special indices (1 and n+2)

---

### 1.3 dict_loader.jl (500+ lines)
**Purpose**: SudachiDict binary serialization + KN language modeling

**Binary Dict Format**:
```
[MAGIC(4)] [VERSION(4)]
[n_strings(8)] [heap_bytes(8)] [n_entries(8)] [n_R(8)] [n_S(8)]
[padding(8)]
[heap: all UTF-8 strings]
[string_offsets: n_strings × UInt32]
[string_lens: n_strings × UInt32]
[entries: n_entries × (surf_id, read_id, left_id, right_id, cost, pad)]
[reading_index R: (key_id, entry_ids)*]
[surface_index S: (key_id, entry_ids)*]
```

**Key Data Structures**:
- `DictEntry(surface, reading, left_id, right_id, cost)`
- `KNCounts` — Surface-based KN with bigram+trigram:
  - `𝒫_pair[a][b]` — count of b following a
  - `𝒫_uni[a]` — unigram count
  - `𝒫_cont[b]` — number of distinct predecessors (continuation)
  - `𝒫_triple[a][b][c]` — trigram counts
  - `𝒫_bi4tri[a][b]` — bigram count for trigram backing-off
- `KNCountsClass` — Same but with Int16 IDs (left_id/right_id)

**Kneser-Ney Bigram Scoring**:
```
ℓ_kn_bigram(a, b; d=0.75):
  if count(a) == 0: return log(continuation_count(b) / total)
  backoff_weight λ = d × |{b' : count(a,b') > 0}| / count(a)
  return log(max(count(a,b) - d, 0) / count(a) + λ × continuation_prob(b))
```
- `κ = 1000.0` — Cost multiplier
- `d = 0.75` — Discount parameter (tunable)

**Kneser-Ney Trigram Scoring** (ALG-2 recursion):
```
ℓ_kn_trigram(a, b, c; d):
  if count(a,b) == 0: return ℓ_kn_bigram(b, c)
  else: recursively back off to bigram with smoothing
```

**KN Cost Caching**:
- Thread-local hash tables for (a, b, c) → cost lookup
- 2^20 entry cache per thread, linear probing with 16 probes max
- Reset before each epoch

**User Dictionary**:
- TSV format: `surface\treading[,\tcost[,\tleft_id\tright_id]]`
- Cost default: 3000, left_id/right_id default: 1 (generic noun)
- Injected as prepended entries (tried first in Viterbi)

---

### 1.4 kkc_inference.jl (410 lines)
**Purpose**: Inference-only KKC using Viterbi decoding

**Key Components**:
- Deserializes `tuned_model.bin` (5-tuple: W, kn, kn_class, brown, W_sem)
- Loads dictionary from mmap binary
- Brown clusters for semantic scoring
- KN counts for surface bigram/trigram LM

**Viterbi Implementation**:
```
build_lattice(input, R, max_span=12):
  - For each char position, collect all dictionary matches (surface→entries)
  - Return lattice[pos] = Vec<LatticeNode> at each position

viterbi(lattice, W, kn; ...):
  - DP table: best_costs[pos][node] = min cost to reach (pos, node)
  - Transition costs:
    W[right_id, left_id] + kn_cost(p2, p1, curr) + entry.cost
  - Optional: dynamic cache boost (-1500) for global_context words
  - Optional: class-based KN with left_id/right_id IDs
  - Optional: semantic matrix W_sem[cluster_prev, cluster_curr]
  - Backtrack to reconstruct path
```

**Features Included in Scoring**:
1. **Connection cost**: W[right_id, left_id]
2. **Dictionary entry cost**: entry.cost
3. **Surface bigram/trigram LM**: kn_cost(prev2, prev1, curr)
4. **Class-based KN**: kn_cost_class(id_prev2, id_prev1, id_curr)
5. **Dynamic cache boost**: -1500 if surface ∈ global_context
6. **Semantic transitions**: W_sem[cluster_prev, cluster_curr]

**Extract Context Vocabulary**:
- Greedy longest-match from document string
- Extracts known surfaces of length ≥ 2
- Used for cache language model (give matches -1500 boost)

---

### 1.5 kkc_tuner_fast.jl (1749 lines)
**Purpose**: Averaged perceptron training for connection matrix and language models

#### Training Data Flow
1. **Load dictionaries** → read_index R, surface_index S
2. **Load corpora** → raw sentences (with optional bunsetsu boundaries)
3. **Build training pairs**:
   - Viterbi-based morphological analysis with Sudachi matrix
   - Or greedy longest-match (if no Sudachi matrix)
   - Bunsetsu-aware if tab-separated input
4. **Collect KN counts** (multi-threaded):
   - Thread-local accumulators → merge at end
   - Bigrams: count all `(prev.surface, curr.surface)` from gold paths
   - Trigrams: count `(pp, prev, curr)` from sequences
5. **Collect class-based KN counts**:
   - Uses `prev.right_id → curr.left_id` IDs instead of surfaces
   - No surface ambiguity in class space
6. **Train averaged perceptron** (multi-epoch)

#### Viterbi Function Signature
```julia
viterbi(lattice, W, kn;
  λ_kn=1.0, kn_d=0.75, beam=0,
  ctx_prev2="BOS", ctx_prev1="BOS",
  pos_bg=nothing,  # POSBigram
  global_context=Set(),  # cache LM
  kn_class=nothing, λ_class=1.0,
  ctx_prev2_rid=Int16(0), ctx_prev1_rid=Int16(0),
  brown=nothing, W_sem=nothing, λ_sem=1.0) → ViterbiResult
```

**Feature Template in Viterbi** (lines 542-597):
```
1. w_conn     = W[right_id, left_id]         # connection cost
2. w_kn       = λ_kn × kn_cost(p2, p1, c)   # surface trigram LM
3. w_pos      = pos_bg.λ_pos × pos_costs[prev_tid, curr_tid]  # POS bigram
4. w_hira     = 600 if all hiragana           # penalize pure hiragana
5. w_cache    = -1500 if in global_context   # dynamic cache boost
6. w_kn_class = λ_class × kn_cost_class(...) # class-based KN
7. w_sem      = λ_sem × W_sem[cluster, cluster]  # semantic transitions
8. node.cost  = dictionary entry cost

Total = prev_cost + w_conn + w_kn + w_pos + w_hira + node.cost + 
        w_cache + w_kn_class + w_sem
```

#### Wakati/POS Bonus-Penalty Mechanism (lines 1326-1374)
**Goal**: Encourage correct word segmentation (wakati) and POS boundaries

**Bonus Learning Rate**: `bonus_lr = eff_lr / 32.0` (small signal)
**Penalty Learning Rate**: `penalty_lr = eff_lr / 64.0` (even smaller, asymmetric)

**Gold segment boundaries**:
```julia
gold_spans = Set{(begin_pos, end_pos)} from correct_path
```

**For each transition in predicted path**:
- Extract surface spans: `(pred[k].begin_pos, pred[k].end_pos)`
- Check if span matches gold: `span_ok = span ∈ gold_spans`
- Update: `push!(buf, idx => (span_ok ? -bonus_lr : penalty_lr))`
  - Correct boundary → light reward
  - Wrong boundary → light penalty

**POS Structured Signal**:
- Build gold connection set: `gold_conns = {right_id → left_id}` from correct path
- For each predicted transition:
  - If matches gold POS: reward with `-bonus_lr`
  - If doesn't match: penalty with `penalty_lr`

**Gold-Side Reward** (corpus-gathered signal):
- Even if a correct connection doesn't appear in predicted path, reward it
- Ensures structured signal covers full gold analysis
- Uses `bonus_lr = eff_lr / 32.0`

#### Semantic (Brown Cluster) Update (lines 1388-1405)
```julia
# Gold cluster transitions: reward with -eff_lr
for k in 1:len(correct)-1
    cluster_a = get_cluster(brown, correct[k].surface)
    cluster_b = get_cluster(brown, correct[k+1].surface)
    push!(W_sem_buf, (cluster_a, cluster_b) => -eff_lr)
end

# Predicted cluster transitions: penalize with eff_lr
for k in 1:len(predicted)-1
    cluster_a = get_cluster(brown, predicted[k].surface)
    cluster_b = get_cluster(brown, predicted[k+1].surface)
    push!(W_sem_buf, (cluster_a, cluster_b) => eff_lr)
end
```

#### Averaged Perceptron Algorithm
```julia
for epoch in 1:epochs
    η = max(η₀ × ρ^(epoch-1), η_min)  # learning rate schedule
    
    for chunk in mini_batches
        for (i, sentence) in chunk
            # Viterbi top-k with current weights
            predicted_paths = viterbi_topk(lattice, W, kn; k=topk_match)
            
            correct_surfs = [nd.surface for nd in gold_path if nd ∉ (BOS, EOS)]
            pred_match = any(p.path matches correct_surfs)
            
            if !pred_match
                # Perceptron update: wrong → update weights
                # For each transition in correct and predicted paths:
                # W[idx] += Δ (correct: -eff_lr, predicted: +eff_lr)
                
                # Additional structured updates:
                # - Wakati bonus/penalty based on segment boundaries
                # - POS bonus/penalty based on connection matching
                # - Semantic matrix updates based on cluster transitions
                # - Gold-side reward for corpus signal coverage
                
                apply_updates(W, W_sem, accum_buffer, eff_lr)
```

**Learning Rate Schedules**:
- Default η₀ = 3.0, ρ = 0.85, η_min = 1.0
- Decay: 3.0 → 2.55 → 2.17 → 1.84 → 1.57 → 1.33 → 1.13 → 1.0 → 1.0...
- Variant samples (synonym-generated): η_scaled = eff_lr × variant_lr_scale (0.2)

**Gradient Accumulation**:
- Accumulate deltas across `grad_accum` chunks before applying
- Lower grad_accum → faster early convergence (default 1)
- Averaging: `W_avg = sum(W_snapshot × timesteps) / total_timesteps`

**Error Metrics** (per epoch):
1. **Sentence-level accuracy**: % of sentences with correct segmentation
2. **Morpheme-level accuracy**: per-word F1 on first 2000 sentences sample

---

### 1.6 synonyms.jl (342 lines)
**Purpose**: Synonym-aware corpus augmentation + soft cost injection

**Data Structure**:
- `SynonymEntry` — one synonym from synonyms.txt (9 fields):
  - group, taigen_flag, expand_flag, lexeme_no, form_type, abbrev_flag, spelling_var, domain, headword
- `SynonymDB` — indexed by group and by word

**(A) Corpus Augmentation**:
```julia
augment_corpus(sentences, db; max_variants=3, expand_flags=[0,1], form_types=[0])
  → (augmented_sentences, is_variant_bool_vec)
```
- For each sentence, find synonym-eligible words
- Generate up to `max_variants` single-token substitutions
- Mark synthetic rows as `is_variant=true` for reduced learning rate
- Reduces learning rate by 0.2× on synthetic examples

**Filter Logic**:
- Only allow `expand_flag ∈ {0, 1}` (0=always, 1=not-trigger)
- Skip expand_flag=2 (never)
- Default form_types=[0] (代表語のみ) to avoid spelling variant noise

**Expansion**: Called before sentence_to_nodes for morphological analysis

**(B) Soft Synonym Cost Injection**:
```julia
inject_synonym_costs!(kn, db; weight=0.3)
```
- For each synonym pair (a, b) in same group:
  - If C(a) > 0 but C(a,b) == 0 (seen a but never followed by b):
    - Inject virtual_count = max(1, round(weight × C(a)))
  - Update continuation count: C(b) += 1
  - Update bigram total
- Weight=0.0 (no injection) to 1.0 (full interchangeability)
- Recommended: weight=0.3 for gentle smoothing

**Integration**: Called after collect_kn_counts! to soften learned costs

---

## 2. DATA DIRECTORY STRUCTURE

```
data/
├── cache/
│   ├── corpus.bin          (4186 KNBC sentences, ~20-50 MB binary)
│   └── sudachidict.bin     (SudachiDict index, mmap zero-copy)
├── knbc/
│   └── KNBC_v1.0_090925_utf8/
│       ├── corpus1/        (4186 bunsetsu-segmented sentences)
│       └── corpus2/        (Sports, Keitai, Gourmet, Kyoto subsets)
├── snow/
│   ├── T15-2020.1.7.csv    (Easy Japanese examples)
│   └── T23-2020.1.7.csv    (CC-BY 4.0)
├── sudachidict/
│   ├── small_lex.csv       (Core vocabulary)
│   ├── core_lex.csv        (Extended vocabulary)
│   ├── dict.bin            (mmap-friendly binary index)
│   ├── matrix.def          (5981×5981 connection cost matrix)
│   └── synonyms.txt        (SudachiDict synonym groups)
├── user_dict.tsv           (Custom entries with reduced costs)
└── tanaka/
    └── examples.utf        (Optional: Tanaka Corpus)

Statistics:
- KNBC:  4186 sentences, ~27792 segments (6.6 bunsetsu/sent avg)
- SNOW:  ~50k sentences from T15/T23
- Total: ~5M+ training tokens
```

---

## 3. KEY DATA FILES

### 3.1 tuned_model.bin (168 MB)
**Format**: Julia serialization (deserialize reads directly)
**Contents** (5-tuple):
1. `W` — Matrix{Int32}(5983×5983) — trained connection costs
2. `kn` — KNCounts — surface-based bigram/trigram LM
3. `kn_class` — KNCountsClass — class-based (POS ID) LM
4. `brown` — BrownClusters — word→cluster mapping (256 clusters)
5. `W_sem` — Matrix{Int32}(257×257) — semantic transition matrix

**Backward Compatibility**:
- Old format (3 items): W, kn, kn_class (no brown/W_sem)
- Old format (2 items): W, kn (no class-based or semantic)

---

### 3.2 kn_bigram.tsv (327k lines)
**Format**:
```
# abkn_cost  (trigram-backed, kappa=1000.0, d=0.75)
 *3845
 and3845
 announcement5603
[...]
```

**Generation**: `export_kn_tsv(kn, "kn_bigram.tsv")`

**Structure**:
- Sorted by surface `a` (first element)
- Then sorted by `b` within each group
- Costs: -log P(a→b) × 1000, clamped to [0, 30000]

---

### 3.3 matrix.def (35.7M, 35.7M lines)
**Format**: Sudachi matrix format (sparse representation)
```
5981 5981
1 0 -2056
2 0 -1193
3 0 -564
[...]
```

**Interpretation**:
- Line 1: `max_right_id max_left_id`
- Other lines: `right_id left_id cost` (only non-zero entries)
- Costs: connection penalty (lower = better)
- Sudachi's pre-trained JMdict/NEologd-based weights

**Usage**: Initializes W matrix during training, provides baseline for Viterbi preprocessing

---

## 4. FEATURE ENGINEERING

### 4.1 Feature Template (Viterbi Cost Decomposition)
```
Total Cost = Σ
  1. Connection cost        W[r_id, l_id]
  2. Dictionary entry cost  entry.cost
  3. Surface LM (trigram)   kn_cost(p2, p1, c)
  4. POS bigram            pos_bg.costs[prev_tag, curr_tag]
  5. Hiragana penalty      +600 if all hiragana
  6. Cache boost           -1500 if in document context
  7. Class-based LM        kn_cost_class(id_p2, id_p1, id_c)
  8. Semantic transitions  W_sem[cluster_prev, cluster_curr]
```

### 4.2 Kneser-Ney Details
- **Discount d=0.75**: Subtracted from observed counts before backoff
- **Backing-off weight λ**: Weight on lower-order distribution
- **Trigram recursion**: If (a,b) not seen, back off to bigram (b,c)
- **Continuation count**: Number of distinct predecessors (for KN smoothing)
- **κ=1000**: Scale costs to integers (avoids floating-point precision issues)

### 4.3 Perceptron Feature Updates
Three levels of structured signals:

**Level 1: Standard Perceptron**
- Update correct path: -eff_lr
- Update predicted path: +eff_lr
- One-pass, all transitions

**Level 2: Wakati/Segmentation Structure** (bonus_lr = eff_lr/32)
- Correct segment boundaries: -bonus_lr (encourage)
- Wrong segment boundaries: +penalty_lr (discourage)
- Prevents cascading segmentation errors

**Level 3: POS Structure** (bonus_lr = eff_lr/32)
- Matching POS transitions: -bonus_lr (encourage)
- Non-matching POS: +penalty_lr (discourage)
- Guides morphological correctness

**Level 4: Gold-Side Coverage** (bonus_lr = eff_lr/32)
- Even if a correct connection doesn't appear in predicted, still reward it
- Ensures corpus signal covers full gold annotation
- Prevents undertraining on unambiguous sequences

**Level 5: Semantic Transitions**
- Updates W_sem with Brown cluster transitions
- Gold cluster pairs: -eff_lr
- Predicted pairs: +eff_lr
- Learns semantic relationship structure

### 4.4 Synonym Variant Learning
- Generated synonyms marked as `is_variant=true`
- Learning rate reduced by `variant_lr_scale=0.2`
- Prevents model from overfitting to autogenerated text
- Effective: `eff_lr = is_variant ? max(1.0, η × 0.2) : η`

---

## 5. TRAINING ALGORITHM

### 5.1 Pseudocode
```
Load dictionary (SudachiDict + user_dict)
Load corpora (SNOW, Tanaka, KNBC with binary cache)
Augment corpus with synonyms (optional)
Build training pairs with Viterbi + Sudachi matrix

# Initialize models
W ← Sudachi matrix.def  (5983×5983, pre-trained baseline)
W_sem ← zeros(257×257)  (Brown clusters + 1 for unknown)
kn ← empty KNCounts()
kn_class ← empty KNCountsClass()
brown ← brown_cluster(training_pairs, 256)

# Collect statistics from gold paths
collect_kn_counts!(kn, training_pairs)      // surface bigram/trigram
collect_kn_class_counts!(kn_class, pairs)   // POS ID bigram/trigram
learn_pos_bigrams(sudachidict_csvs, knbc)   // POS transition costs

# Training loop
W_avg ← zeros_like(W)
for epoch in 1:epochs:
    η ← max(η₀ × ρ^(epoch-1), η_min)
    shuffle(training_pairs)
    
    for mini_batch in mini_batches(pairs, batch_size=4096):
        for (sentence, gold_path) in mini_batch:
            # Top-k Viterbi with current weights
            predicted_topk ← viterbi_topk(lattice, W, kn; k=3)
            correct_match ← gold_path ∈ {p.path for p in predicted_topk}
            
            if !correct_match:
                # Standard perceptron update
                for transition in gold_path:
                    W[idx] -= eff_lr
                for transition in predicted_topk[1].path:
                    W[idx] += eff_lr
                
                # Structured: wakati + POS + gold-side reward
                # Asymmetric: bonus/32 > penalty/64
                
                # Semantic matrix update
                for cluster_pair in gold_path:
                    W_sem[idx] -= eff_lr
                for cluster_pair in predicted:
                    W_sem[idx] += eff_lr
        
        # Accumulate gradients (grad_accum=1: apply each chunk)
        # Use averaged weights formula
        
        # Eval: sentence-level accuracy % on full set
        #       morpheme-level accuracy % on 2000-sentence sample

return averaged_W, kn, kn_class, brown, averaged_W_sem
```

### 5.2 Key Hyperparameters
```
epochs = 10
η₀ (initial learning rate) = 3.0
ρ (decay factor) = 0.85
η_min = 1.0
mini_batch = 4096
beam = 8 (beam width for Viterbi top-k)
topk_match = 3 (correct if in top-3)
variant_lr_scale = 0.2 (scale for synonym examples)
grad_accum = 1 (gradient accumulation chunks)
λ_kn = 1.0 (weight for surface KN cost)
λ_class = 1.0 (weight for class-based KN)
λ_sem = 1.0 (weight for semantic matrix)
kn_d = 0.75 (Kneser-Ney discount)
n_brown_clusters = 256
```

### 5.3 Error Calculation
**Sentence-level**: Binary — either entire segmentation matches gold or not
- Used for convergence detection: `if nerr == 0: break`

**Morpheme-level** (sampled on 2000 sentences):
- Align predicted and gold paths positionally
- Count matching morphemes
- Penalize misaligned lengths
- Less strict than sentence-level, more informative

---

## 6. INFERENCE MODES

### 6.1 Full 1-best Conversion
```julia
result = kkc(input, R, W, kn;
  λ_kn=1.0, kn_d=0.75,
  pos_bg=pos_bigram,
  global_context=Set(),
  kn_class=kn_class, λ_class=1.0,
  brown=brown, W_sem=W_sem, λ_sem=1.0)
```
→ Returns best surface string

### 6.2 Top-k Candidates
```julia
results = kkc_topk(input, R, W, kn; k=3, ...)
```
→ Returns Vec{String} of top-3 alternatives

### 6.3 Segment-Level Editing
```julia
buf = kkc_segment(input, R, S, W, kn; k=5, left_ctx="", ...)
cycle_segment!(buf, idx)  # cycle candidates
reconv_segment!(buf, idx, R, S, W, kn)  # rescore with context
commit_buffer(buf)  # finalize
```
→ KkcBuffer with editable segments and top-k candidates per segment

### 6.4 Partial KKC (with left context)
```julia
candidates = partial_kkc(confirmed, uncommitted, R, S, W, kn; k=3, ...)
```
→ Convert `uncommitted` with `confirmed` as trigram context

### 6.5 Cache Language Model
```julia
ctx_vocab = extract_context_vocabulary(long_document, S)
result = kkc(input, R, W, kn; global_context=ctx_vocab, ...)
```
→ Words in ctx_vocab get -1500 boost (massive probability spike)

---

## 7. PERFORMANCE CHARACTERISTICS

### 7.1 Training Time
- ~5k training examples (KNBC + SNOW)
- 10 epochs with 4096 mini-batch
- Multi-threaded (Threads.@threads :static)
- Estimated: 1-3 hours on modern CPU (threads=auto)

### 7.2 Memory Usage
- Loaded model: 168 MB (W, kn, kn_class, brown, W_sem serialized)
- In-memory during training: ~500 MB (dictionaries, counts, training pairs)
- Inference: <200 MB with dictionary loaded

### 7.3 Inference Speed
- Single-sentence Viterbi: ~10-100 ms (depends on lattice size)
- Lattice build: O(n × max_span) where n=string length, max_span=12
- Viterbi DP: O(lattice_size²) worst case, typically O(n²) in practice
- Top-k: Similar to 1-best (shared forward pass, multiple backtraces)

---

## 8. NOVEL CONTRIBUTIONS

1. **Wakati-Aware Perceptron**: Asymmetric bonus/penalty on segment boundaries
   - Encourages correct word segmentation even when not in gold path
   - Bonus 32× larger than penalty (asymmetric = no drift)

2. **POS Structured Learning**: Separate signal for morphological analysis
   - Learned POS bigrams from KNBC annotations
   - Guides left_id/right_id correctness independent of surface

3. **Semantic Matrix (Brown Clusters)**: Fast semantic similarity
   - O(1) lookup for cluster-pair transitions instead of O(n²) surface pairs
   - Trained agglomeratively from corpus bigrams
   - Learned jointly with connection matrix

4. **Synonym Augmentation + Cost Injection**:
   - Two-phase: expand corpus (before morphological analysis)
   - Inject soft costs (after KN counting)
   - Reduced learning rate on synthetic examples prevents overfitting

5. **Class-Based KN**: POS-aware language model
   - Uses left_id/right_id instead of surfaces
   - Zero surface ambiguity, cleaner statistics
   - Separate from surface KN (both trained jointly)

6. **Averaged Perceptron with Gradient Accumulation**:
   - Accumulate deltas across chunks before W update
   - Compute time-weighted average at end
   - Reduces noise, stabilizes training

