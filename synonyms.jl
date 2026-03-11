# SPDX-License-Identifier: Apache-2.0
#
# synonyms.jl  ―  SudachiDict synonym-aware corpus augmentation
#
# Integrates synonyms.txt into the KKC learning pipeline by:
#   (A) SynonymAugmenter  : expands training sentences with synonym variants
#   (B) SynonymKNBridge   : injects synonym pair soft-costs into KNCounts
#
# Usage (add to kkc_tuner_fast.jl after loading corpora):
#
#   syn = load_synonyms("data/synonyms.txt")
#   augmented = augment_corpus(𝒯_raw_sentences, syn;
#                               max_variants=3, expand_flags=[0,1])
#   # then feed `augmented` into sentence_to_nodes / collect_kn_counts!
#
#   # Optionally soften KN costs between synonyms after counting:
#   inject_synonym_costs!(kn, syn; weight=0.3)

# ════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════

"""
One row in synonyms.txt.

Fields mirror the spec:
  group        col 0  — shared group id (6-digit)
  taigen_flag  col 1  — 1=体言 2=用言 (omittable)
  expand_flag  col 2  — 0=always 1=not-trigger 2=never (omittable)
  lexeme_no    col 3  — lexeme number inside group (omittable)
  form_type    col 4  — 0=代表 1=対訳 2=別称 3=旧称 4=誤用
  abbrev_flag  col 5  — 0=代表語形 1=略語(alpha) 2=略語(other)
  spelling_var col 6  — 0=代表 1=alpha 2=異表記 3=誤表記
  domain       col 7  — free-text domain tag e.g. "(医療)"
  headword     col 8  — the actual surface string
"""
struct SynonymEntry
    group       :: Int
    taigen_flag :: Int8    # 0=unknown 1=体言 2=用言
    expand_flag :: Int8    # 0=always  1=no-trigger  2=never
    lexeme_no   :: Int16
    form_type   :: Int8
    abbrev_flag :: Int8
    spelling_var:: Int8
    domain      :: String
    headword    :: String
end

"""
Loaded synonym database.

  by_group  : group_id → Vector{SynonymEntry}
  by_word   : headword  → Vector{SynonymEntry}  (same word may be in many groups)
"""
struct SynonymDB
    by_group :: Dict{Int, Vector{SynonymEntry}}
    by_word  :: Dict{String, Vector{SynonymEntry}}
end

# ════════════════════════════════════════════════════════════
# Parser
# ════════════════════════════════════════════════════════════

_parse_int8(s::AbstractString, default::Int8) = begin
    t = strip(s)
    isempty(t) && return default
    v = tryparse(Int, t)
    v === nothing ? default : Int8(clamp(v, -128, 127))
end

_parse_int16(s::AbstractString, default::Int16) = begin
    t = strip(s)
    isempty(t) && return default
    v = tryparse(Int, t)
    v === nothing ? default : Int16(clamp(v, -32768, 32767))
end

"""
Load `synonyms.txt` (one entry per line, CSV, blank lines between groups).
Returns a `SynonymDB`.
"""
function load_synonyms(path::String)::SynonymDB
    by_group = Dict{Int, Vector{SynonymEntry}}()
    by_word  = Dict{String, Vector{SynonymEntry}}()
    n = 0

    open(path) do f
        for raw in eachline(f)
            line = strip(raw)
            (isempty(line) || startswith(line, '#')) && continue

            cols = split(line, ','; limit=11)
            length(cols) < 9 && continue

            group = parse(Int, strip(cols[1]))
            hw    = strip(cols[9])
            isempty(hw) && continue

            e = SynonymEntry(
                group,
                _parse_int8( cols[2], Int8(0)),
                _parse_int8( cols[3], Int8(0)),   # expand_flag default=0 (always)
                _parse_int16(cols[4], Int16(0)),
                _parse_int8( cols[5], Int8(0)),
                _parse_int8( cols[6], Int8(0)),
                _parse_int8( cols[7], Int8(0)),
                strip(cols[8]),
                hw,
            )

            push!(get!(Vector{SynonymEntry}, by_group, group), e)
            push!(get!(Vector{SynonymEntry}, by_word,  hw),    e)
            n += 1
        end
    end
    println("Loaded $n synonym entries  ($(length(by_group)) groups)")
    SynonymDB(by_group, by_word)
end

# ════════════════════════════════════════════════════════════
# Expansion helpers
# ════════════════════════════════════════════════════════════

"""
Return all headwords in the same synonym group as `word` that are
eligible for expansion, filtered by `expand_flags` and `form_types`.

  expand_flags  — which `expand_flag` values to allow (default [0,1])
  form_types    — which `form_type` values to allow (default [0,1,2,3])
                  set to [0] for representative-form only
"""
function synonyms_of(
    db          :: SynonymDB,
    word        :: String;
    expand_flags :: Vector{Int} = [0, 1],
    form_types   :: Vector{Int} = [0, 1, 2, 3],
)::Vector{String}
    entries = get(db.by_word, word, SynonymEntry[])
    isempty(entries) && return String[]

    out = String[]
    for e in entries
        # The word itself must not be expand_flag=2 (never)
        Int(e.expand_flag) == 2 && continue
        for peer in get(db.by_group, e.group, SynonymEntry[])
            peer.headword == word && continue
            Int(peer.expand_flag) ∉ expand_flags && continue
            Int(peer.form_type)   ∉ form_types   && continue
            push!(out, peer.headword)
        end
    end
    unique!(out)
end

# ════════════════════════════════════════════════════════════
# (A)  Corpus augmentation
# ════════════════════════════════════════════════════════════

"""
Given a list of raw sentences, generate synonym-substituted variants.

Strategy:
  For each sentence, find tokens that exist in the synonym DB.
  Generate up to `max_variants` alternative sentences by substituting
  one synonym at a time (single-token swap, not combinatorial explosion).

Arguments:
  sentences     — raw Japanese sentence strings
  db            — loaded SynonymDB
  max_variants  — maximum extra sentences per original (default 3)
  expand_flags  — forwarded to `synonyms_of`
  form_types    — forwarded to `synonyms_of`
                  NOTE: default is [0] (代表語のみ) to avoid noise from
                  spelling variants and loanword forms like "1番" vs "一番".
                  Pass [0,1,2,3] to include all forms.

Returns (sentences, is_variant) — parallel Bool vector marks synthetic rows.
Caller should apply reduced perceptron learning rate to variant sentences.
"""
function augment_corpus(
    sentences    :: Vector{String},
    db           :: SynonymDB;
    max_variants :: Int        = 3,
    expand_flags :: Vector{Int} = [0, 1],
    form_types   :: Vector{Int} = [0],      # 代表語のみ — reduces noisy variants
)::Tuple{Vector{String}, Vector{Bool}}
    seen   = Set{String}(sentences)
    out    = copy(sentences)
    is_var = fill(false, length(sentences))

    for sent in sentences
        hits = _find_synonym_spans(sent, db)
        isempty(hits) && continue

        added = 0
        for (span_start, span_end, orig_tok) in hits
            added >= max_variants && break
            syns = synonyms_of(db, orig_tok;
                                expand_flags=expand_flags,
                                form_types=form_types)
            for syn in syns
                added >= max_variants && break
                new_sent = sent[1:prevind(sent, span_start)] *
                           syn *
                           sent[span_end:end]
                if new_sent ∉ seen
                    push!(seen,   new_sent)
                    push!(out,    new_sent)
                    push!(is_var, true)      # mark as synthetic
                    added += 1
                end
            end
        end
    end

    n_orig = length(sentences)
    println("Augmented: $n_orig → $(length(out)) sentences  (+$(length(out)-n_orig) variants)")
    return out, is_var
end

"""
Find non-overlapping longest-match spans in `sent` whose surface exists in `db.by_word`.
Returns Vector{(byte_start, byte_after_end, surface)}.
"""
function _find_synonym_spans(
    sent :: String,
    db   :: SynonymDB,
    max_len :: Int = 12,
)::Vector{Tuple{Int,Int,String}}
    chars  = collect(sent)
    n      = length(chars)
    spans  = Tuple{Int,Int,String}[]
    pos    = 1
    byte_offsets = [nextind(sent, 0, i) for i in 1:n]   # char→byte start

    while pos <= n
        best_len = 0
        best_tok = ""
        for len in min(max_len, n - pos + 1):-1:1
            tok = String(chars[pos:pos+len-1])
            if haskey(db.by_word, tok)
                best_len = len; best_tok = tok; break
            end
        end
        if best_len > 0
            b_start = byte_offsets[pos]
            b_end   = pos + best_len <= n ? byte_offsets[pos + best_len] : ncodeunits(sent) + 1
            push!(spans, (b_start, b_end, best_tok))
            pos += best_len
        else
            pos += 1
        end
    end
    spans
end

# ════════════════════════════════════════════════════════════
# (B)  Soft synonym cost injection into KNCounts
# ════════════════════════════════════════════════════════════

"""
After `collect_kn_counts!`, soften the KN cost between synonym pairs.

For each synonym pair (a, b) in the DB:
  If C(a) > 0 but C(a,b) == 0  (a was seen but never followed by b),
  inject a virtual count as if `b` appeared `weight * C(a)` times after `a`.
  This prevents the model from assigning −∞ log-prob to synonym transitions.

  weight=0.0  → no injection (noop)
  weight=1.0  → treat synonyms as fully interchangeable
  weight=0.3  → gentle smoothing (recommended default)

Only `expand_flag ∈ {0,1}` entries are used.
"""
function inject_synonym_costs!(
    kn     :: KNCounts,
    db     :: SynonymDB;
    weight :: Float64 = 0.3,
)
    injected = 0
    for (word, entries) in db.by_word
        Ca = get(kn.𝒫_uni, word, 0)
        Ca == 0 && continue                    # word never seen in corpus

        syns = synonyms_of(db, word;
                            expand_flags=[0, 1],
                            form_types=[0, 1, 2, 3])
        for syn in syns
            inner = get!(Dictionary{String,Int}, kn.𝒫_pair, word)
            existing = get(inner, syn, 0)
            virtual_count = max(1, round(Int, weight * Ca))

            # Only inject if never seen (don't overwrite real observations)
            if existing == 0
                set!(inner, syn, virtual_count)
                # Also update continuation count for the synonym
                set!(kn.𝒫_cont, syn, get(kn.𝒫_cont, syn, 0) + 1)
                kn.𝑁_pairs += virtual_count
                injected   += 1
            end
        end
    end
    println("Synonym injection: $injected virtual (word→synonym) pairs added  (weight=$weight)")
end

# ════════════════════════════════════════════════════════════
# Integration shim:  call this inside train() after corpora load
# ════════════════════════════════════════════════════════════

"""
    synonym_pipeline!(raw_sentences, kn, syn_path; ...) -> (sentences, is_variant)

Convenience wrapper combining augmentation and/or cost injection.

`is_variant` is a parallel Bool vector: true = synthetically generated row.
Pass it to `train()` so the perceptron can apply reduced η to those rows
and avoid learning spurious surface-form preferences from autogenerated text.

phase:
  :augment — only expand corpus (call BEFORE sentence_to_nodes)
  :inject  — only inject KN soft costs (call AFTER collect_kn_counts!)
  :both    — both in sequence
"""
function synonym_pipeline!(
    raw_sentences :: Vector{String},
    kn            :: KNCounts,
    syn_path      :: String;
    max_variants  :: Int     = 3,
    inject_weight :: Float64 = 0.3,
    phase         :: Symbol  = :augment,
)::Tuple{Vector{String}, Vector{Bool}}
    db     = load_synonyms(syn_path)
    is_var = fill(false, length(raw_sentences))

    if phase ∈ (:augment, :both)
        raw_sentences, is_var = augment_corpus(raw_sentences, db; max_variants)
    end
    if phase ∈ (:inject, :both)
        inject_synonym_costs!(kn, db; weight=inject_weight)
    end
    return raw_sentences, is_var
end