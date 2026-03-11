═══════════════════════════════════════════════════════════════════════════════
JULIA KKC PROJECT - QUICK REFERENCE SUMMARY
═══════════════════════════════════════════════════════════════════════════════

PROJECT STRUCTURE
─────────────────

6 Main .jl Files (in reading order):

1. brown_clustering.jl (191 lines)
   - Greedy agglomerative clustering using PMI affinity
   - Creates 256 clusters from training corpus (default)
   - Maps word surfaces → cluster IDs for O(1) semantic lookup
   - Used: W_sem[cluster_prev, cluster_curr] matrix at inference

2. corpus_loaders.jl (430 lines)
   - Loads SNOW, Tanaka, KNBC with binary mmap cache
   - Gold bunsetsu boundaries encoded as tab-separated chunks
   - POS bigram learning from KNBC gold annotations
   - Maps KNP POS tags → Sudachi POS tag strings

3. dict_loader.jl (500+ lines)
   - SudachiDict CSV → binary mmap format
   - User dictionary injection (TSV format)
   - Kneser-Ney bigram/trigram scoring:
     * Surface-based: (surface, surface, surface)
     * Class-based: (left_id, right_id) integers
   - Thread-local KN cost cache (2^20 entry hash tables)

4. kkc_inference.jl (410 lines)
   - Inference-only Viterbi decoder
   - Loads tuned_model.bin (5-tuple: W, kn, kn_class, brown, W_sem)
   - 8 cost components:
     1. W[right_id, left_id]
     2. entry.cost
     3. kn_cost(p2, p1, curr)
     4. pos_bigram[prev_tag, curr_tag]
     5. +600 if all hiragana
     6. -1500 if in global_context (cache boost)
     7. kn_cost_class(rid_p2, rid_p1, lid_c)
     8. W_sem[cluster_prev, cluster_curr]

5. kkc_tuner_fast.jl (1749 lines)
   - Main training loop (averaged perceptron)
   - Viterbi + top-k search for error detection
   - Structured learning:
     * Wakati bonus/penalty (correct/wrong segment boundaries)
     * POS bonus/penalty (correct/wrong POS transitions)
     * Gold-side reward (corpus signal coverage)
     * Semantic matrix updates (Brown cluster pairs)
   - Learning rate schedule: η = max(η₀ × ρ^(t-1), η_min)
   - Gradient accumulation across mini-batches
   - Multi-threaded with per-thread accumulators

6. synonyms.jl (342 lines)
   - Corpus augmentation (single-token synonym substitution)
   - Soft cost injection into KN counts (weight=0.3 default)
   - Reduced learning rate on synthetic examples (0.2× scale)


DATA FILES
──────────

tuned_model.bin (168 MB)
  Format: Julia serialization
  Contents:
    1. W Matrix{Int32}(5983×5983) — trained connection costs
    2. kn KNCounts — surface bigram/trigram LM
    3. kn_class KNCountsClass — POS ID bigram/trigram LM
    4. brown BrownClusters — word→cluster mapping (256 clusters)
    5. W_sem Matrix{Int32}(257×257) — semantic transition costs

kn_bigram.tsv (327k lines)
  Format: a<TAB>b<TAB>kn_cost
  Trigram-backed costs with κ=1000, d=0.75
  Exported via: export_kn_tsv(kn, "kn_bigram.tsv")

matrix.def (35.7M)
  Format: Sudachi matrix format (sparse)
  Content: Sudachi's pre-trained JMdict/NEologd weights
  Used: Initializes W matrix, provides baseline for corpus preprocessing
  Example line: "right_id left_id cost"


TRAINING DATA
─────────────

KNBC:  4186 sentences, ~27792 segments (6.6 bunsetsu/sent)
SNOW:  ~50k sentences from T15/T23 CC-BY datasets
Total: ~5M+ tokens

Stored as:
  data/cache/corpus.bin (binary mmap, ~50 MB)
  data/knbc/KNBC_v1.0_090925_utf8/corpus1/ (raw KNP format)


KEY ALGORITHMS
──────────────

VITERBI DECODING
  lattice[pos] = Vec<LatticeNode> of entries at each position
  DP: cost[pos][node] = min over prev {cost[prev] + transition_cost}
  Transition cost includes:
    - Dictionary entry cost
    - Connection matrix W[right_id, left_id]
    - LM costs (surface trigram + class-based KN)
    - Optional bonuses (cache boost, semantic)
  Backtrack to reconstruct best path

KNESER-NEY SMOOTHING (KN)
  Bigram: max(count(a,b) - d, 0) / count(a) + λ × p_backoff(b)
  where λ = d × |{b' : count(a,b') > 0}| / count(a)
  Trigram: same formula, back off to bigram if needed
  p_backoff uses continuation count (# of distinct predecessors)
  κ = 1000.0 multiplier for integer costs
  d = 0.75 discount (tunable)

AVERAGED PERCEPTRON TRAINING
  for epoch in 1:epochs:
    for mini_batch in batches:
      for sentence:
        pred_topk = viterbi_topk(lattice, W, kn; k=3)
        if gold_segmentation ∉ pred_topk:
          # Perceptron update
          for trans in gold_path:   W[idx] -= eff_lr
          for trans in pred_path:   W[idx] += eff_lr
          
          # Structured learning
          for boundary in gold:    bonus if correct, penalty if wrong
          for POS in gold:         bonus if correct, penalty if wrong
          for cluster in gold:     W_sem[idx] -= eff_lr
          for cluster in pred:     W_sem[idx] += eff_lr
  
  Averaging: W_avg = sum(W_snapshot × timesteps) / total_timesteps

BROWN CLUSTERING
  1. Count bigrams from gold paths, keep top 8000 words by frequency
  2. Assign top 256 to separate clusters
  3. For remaining words: add to cluster maximizing PMI with existing members
  4. PMI(w, c) = log(c_total(w→c) * N / (count(w) * count(c)))
  5. BOS/EOS → cluster 0

SYNONYMS
  Augmentation: single-token substitution, up to 3 variants per sentence
  Cost injection: virtual_count = weight × count(a) where count(a,b)==0
  Learning: reduce η by variant_lr_scale (0.2) on synthetic examples


HYPERPARAMETERS
────────────────

Training:
  epochs = 10
  η₀ = 3.0 (initial learning rate)
  ρ = 0.85 (decay: η_t = η₀ × ρ^(t-1))
  η_min = 1.0 (minimum learning rate)
  mini_batch = 4096
  grad_accum = 1 (gradient accumulation chunks)
  topk_match = 3 (correct if in top-3)
  variant_lr_scale = 0.2 (synonym variant learning rate scale)

Model:
  n_brown_clusters = 256
  λ_kn = 1.0 (surface KN weight)
  λ_class = 1.0 (class-based KN weight)
  λ_sem = 1.0 (semantic matrix weight)
  kn_d = 0.75 (Kneser-Ney discount)
  beam = 8 (beam width for top-k)
  max_span = 12 (max dictionary lookup span)


WAKATI & POS FEATURES
─────────────────────

WAKATI (Segmentation) Bonus/Penalty:
  gold_spans = {(begin_pos, end_pos) for each segment in gold_path}
  For each predicted segment:
    if span ∈ gold_spans: push(buf, idx => -bonus_lr)   # reward
    else:                 push(buf, idx => penalty_lr)   # penalize
  bonus_lr = eff_lr / 32.0 (small but asymmetric)
  penalty_lr = eff_lr / 64.0 (even smaller)
  → Prevents cascading segmentation errors

POS Structure:
  gold_conns = {right_id → left_id for each transition in gold}
  For each predicted transition:
    if (r_id, l_id) ∈ gold_conns: update with -bonus_lr
    else:                         update with penalty_lr
  → Guides morphological correctness independent of surface

Gold-Side Reward:
  Even correct connections not in predicted path still get -bonus_lr
  → Ensures corpus signal covers full gold analysis


CACHE LANGUAGE MODEL
────────────────────

Dynamic context boost:
  1. Extract context vocabulary from document (greedy longest-match)
  2. Keep surfaces of length ≥ 2 that exist in dictionary
  3. At Viterbi time: if surface ∈ context, add -1500 cost bonus
  4. Effect: ~10× probability spike for context words


FILE FORMATS
────────────

Binary Dict (dict.bin):
  [MAGIC(4)] [VERSION(4)]
  [n_strings(8)] [heap_bytes(8)] [n_entries(8)] [n_R(8)] [n_S(8)]
  [padding(8)]
  [heap: UTF-8 strings concatenated]
  [string_offsets: n×UInt32] [string_lens: n×UInt32]
  [entries: n×(surf_id, read_id, left_id, right_id, cost, pad)]
  [R_index: variable-length reading → entry_id list]
  [S_index: variable-length surface → entry_id list]

Corpus Cache (corpus.bin):
  [MAGIC(4)] [VERSION(4)]
  [n_sentences(8)] [heap_bytes(8)]
  [offsets: n×UInt64] [lengths: n×UInt32]
  [heap: UTF-8 sentences concatenated]

User Dict (user_dict.tsv):
  surface<TAB>reading[<TAB>cost[<TAB>left_id<TAB>right_id]]
  cost default: 3000
  left_id/right_id default: 1 (generic noun)

Matrix (matrix.def):
  right_id_max left_id_max
  right_id left_id cost
  ... (only non-zero entries, sparse format)


PERFORMANCE
───────────

Training:
  Data: ~5k training examples
  Time: ~1-3 hours (10 epochs, 4096 batch, auto-threads)
  Memory: ~500 MB during training

Inference:
  Model size: 168 MB
  Memory: <200 MB with dict loaded
  Speed: ~10-100 ms per sentence (Viterbi)
  Lattice build: O(n × max_span) = O(n × 12)
  DP: O(lattice_size^2) worst case


NOVEL CONTRIBUTIONS
────────────────────

1. Wakati-Aware Perceptron
   - Asymmetric bonus/penalty on segment boundaries
   - Bonus 32× > Penalty (prevents drift)

2. POS Structured Learning
   - Separate signal for morphological correctness
   - Uses POS tags from KNBC annotations

3. Semantic Matrix (Brown Clusters)
   - O(1) cluster-pair lookup instead of O(n²) surface pairs
   - Trained agglomeratively from corpus bigrams

4. Synonym Augmentation + Cost Injection
   - Two-phase: expand corpus, then inject soft costs
   - Reduced learning rate prevents overfitting

5. Class-Based KN
   - Uses left_id/right_id instead of surfaces
   - Zero surface ambiguity, cleaner statistics

6. Averaged Perceptron + Gradient Accumulation
   - Accumulate across chunks before applying
   - Time-weighted averaging reduces noise


DIRECTORY LAYOUT
─────────────────

.
├── brown_clustering.jl        # Brown clusters (semantic)
├── corpus_loaders.jl          # SNOW, Tanaka, KNBC loaders
├── dict_loader.jl             # SudachiDict binary + KN scoring
├── kkc_inference.jl           # Inference-only Viterbi
├── kkc_tuner_fast.jl          # Main training loop
├── synonyms.jl                # Synonym augmentation + injection
├── tuned_model.bin            # Trained W, kn, kn_class, brown, W_sem
├── kn_bigram.tsv              # Exported KN bigram costs
├── matrix.def                 # Sudachi pre-trained matrix
├── data/
│   ├── cache/
│   │   ├── corpus.bin         # Binary corpus cache
│   │   └── sudachidict.bin    # Binary dict cache
│   ├── knbc/                  # KNBC corpus
│   ├── snow/                  # SNOW T15/T23
│   ├── sudachidict/           # SudachiDict CSVs + matrix.def
│   ├── user_dict.tsv          # Custom entries
│   └── tanaka/                # Optional Tanaka corpus
└── README_ANALYSIS.txt        # This file

═══════════════════════════════════════════════════════════════════════════════
See KKC_DETAILED_ANALYSIS.md for complete technical documentation
═══════════════════════════════════════════════════════════════════════════════
