# SPDX-License-Identifier: Apache-2.0
#
# kkc_tuner_fast.jl  ―  KKC Connection Cost Matrix  (Threads.@threads :static + Unicode math)
#
# Training data attribution (see NOTICE file):
#   SudachiDict  (Apache-2.0)  https://github.com/WorksApplications/SudachiDict
#   SNOW T15/T23 (CC-BY 4.0)  http://www.jnlp.org/SNOW/T15
#   Tanaka Corpus (CC-BY)     http://www.edrdg.org/wiki/index.php/Tanaka_Corpus
#
# Launch:  julia --threads=auto kkc_tuner_fast.jl
# Install: julia -e 'using Pkg; Pkg.add(["CSV","DataFrames","InternedStrings",
#            "Dictionaries","StringViews","InlineStrings"])'
# ════════════════════════════════════════════════════════════

using CSV
using DataFrames
using Serialization
using Mmap
using InternedStrings
using Dictionaries
using StringViews
using InlineStrings
using Random
import Base.GC

# ════════════════════════════════════════════════════════════
# Section 1  DATA STRUCTURES
# ════════════════════════════════════════════════════════════

struct DictEntry
    surface  :: String
    reading  :: String
    left_id  :: Int16
    right_id :: Int16
    cost     :: Int32
end

struct LatticeNode
    entry     :: DictEntry
    begin_pos :: Int
    end_pos   :: Int
end

# KN counts: bigram statistics + trigram extension (ALG-2)
mutable struct KNCounts
    𝒫_pair   :: Dictionary{String, Dictionary{String,Int}}
    𝒫_uni    :: Dictionary{String, Int}
    𝒫_cont   :: Dictionary{String, Int}
    𝑁_pairs  :: Int
    𝒫_triple :: Dictionary{String, Dictionary{String, Dictionary{String,Int}}}
    𝒫_bi4tri :: Dictionary{String, Dictionary{String,Int}}
    𝑁_trips  :: Int
end

KNCounts() = KNCounts(
    Dictionary{String, Dictionary{String,Int}}(),
    Dictionary{String, Int}(),
    Dictionary{String, Int}(),
    0,
    Dictionary{String, Dictionary{String, Dictionary{String,Int}}}(),
    Dictionary{String, Dictionary{String,Int}}(),
    0,
)

# Class-based KN counts: Int16 IDs (left_id/right_id) instead of surface strings.
# Bigram: prev.right_id → curr.left_id
# Trigram: pp.right_id → prev.right_id → curr.left_id
mutable struct KNCountsClass
    𝒫_pair   :: Dictionary{Int16, Dictionary{Int16,Int}}
    𝒫_uni    :: Dictionary{Int16, Int}
    𝒫_cont   :: Dictionary{Int16, Int}
    𝑁_pairs  :: Int
    𝒫_triple :: Dictionary{Int16, Dictionary{Int16, Dictionary{Int16,Int}}}
    𝒫_bi4tri :: Dictionary{Int16, Dictionary{Int16,Int}}
    𝑁_trips  :: Int
end

KNCountsClass() = KNCountsClass(
    Dictionary{Int16, Dictionary{Int16,Int}}(),
    Dictionary{Int16, Int}(),
    Dictionary{Int16, Int}(),
    0,
    Dictionary{Int16, Dictionary{Int16, Dictionary{Int16,Int}}}(),
    Dictionary{Int16, Dictionary{Int16,Int}}(),
    0,
)

include("synonyms.jl")
include("brown_clustering.jl")

# ════════════════════════════════════════════════════════════
# Section 1c  POS BIGRAM MODEL
# ════════════════════════════════════════════════════════════

"""
    POSBigram

POS bigram cost table learned from corpus (KNBC gold annotations +
SudachiDict left_id→POS mapping built from CSV).

Scoring at Viterbi time uses `left_id` directly → zero surface ambiguity:
  cost = costs[lid2tag[prev.right_id], lid2tag[curr.left_id]]

Fields:
  tags      — sorted Sudachi POS tag strings
  tag2id    — String → Int (1-based, offset +1 so 1=BOS)
  lid2tag   — Dict{Int16, Int}  left_id  → tag index (1-based)
  rid2tag   — Dict{Int16, Int}  right_id → tag index (for prev node)
  costs     — Matrix{Int32}(n+2, n+2)  [prev_tag, curr_tag]
              row/col 1=BOS, 2..n+1=tags, n+2=EOS
  scale     — -log P multiplier (default 800, ~1 bad connection unit)
  λ_pos     — global weight for inference (tunable)
"""
mutable struct POSBigram
    tags    :: Vector{String}
    tag2id  :: Dict{String,Int}
    lid2tag :: Dict{Int16,Int}
    rid2tag :: Dict{Int16,Int}
    costs   :: Matrix{Int32}
    scale   :: Float32
    λ_pos   :: Float32
end

POSBigram() = POSBigram(
    String[], Dict{String,Int}(),
    Dict{Int16,Int}(), Dict{Int16,Int}(),
    Matrix{Int32}(undef,0,0), 800f0, 1.0f0)

const _POS_BOS = "__BOS__"
const _POS_EOS = "__EOS__"


# ════════════════════════════════════════════════════════════
include("dict_loader.jl")
include("corpus_loaders.jl")

# ════════════════════════════════════════════════════════════
# Section 5  SENTENCE -> TRAINING NODES
# ════════════════════════════════════════════════════════════

function sentence_to_nodes(sentence::String,
                            𝑆::Dictionary{String,Vector{DictEntry}},
                            max_span::Int=12
                           )::Union{Nothing,Tuple{String,Vector{LatticeNode}}}
    chars=collect(sentence); n=length(chars)
    nodes=LatticeNode[]; pos=1
    while pos<=n
        best_len=0; best_e=nothing
        for len in min(max_span,n-pos+1):-1:1
            span=String(chars[pos:pos+len-1])
            if haskey(𝑆,span)
                ents=𝑆[span]
                # Prefer entry whose surface exactly matches the span
                # (avoids picking "1番" over "一番" due to lower dict cost)
                exact = findfirst(e -> e.surface == span, ents)
                best_e = exact !== nothing ? ents[exact] : ents[argmin(e.cost for e in ents)]
                best_len=len; break
            end
        end
        best_e===nothing && return nothing
        push!(nodes, LatticeNode(best_e, pos, pos+best_len))
        pos+=best_len
    end
    hiragana=join(nd.entry.reading for nd in nodes)::String
    return hiragana, nodes
end

"""
    sentence_to_nodes_bunsetsu(tabbed, 𝑆) -> Union{Nothing, Tuple{String,Vector{LatticeNode}}}

Variant of `sentence_to_nodes` for KNBC bunsetsu-segmented training data.

`tabbed` is a tab-separated string of bunsetsu surface chunks, e.g.:
    "と\tいう\tことで、\t京都観光の\t趣旨からは\t…"

For each chunk we do a greedy left-to-right surface lookup — but the
segment boundary is guaranteed by the gold annotation so we never need
to search across boundaries.  This gives the perceptron exact gold-standard
segmentation instead of relying on Viterbi to guess it.

Returns the same `(hiragana, nodes)` type as `sentence_to_nodes` so it
drops into the existing training loop unchanged.
Returns `nothing` if any morpheme in any chunk is missing from 𝑆.
"""
function sentence_to_nodes_bunsetsu(
    tabbed   :: String,
    𝑆        :: Dictionary{String,Vector{DictEntry}};
    max_span :: Int = 12,
)::Union{Nothing,Tuple{String,Vector{LatticeNode}}}

    chunks = split(tabbed, '\t')
    nodes  = LatticeNode[]
    pos    = 1   # character position across the full sentence

    for chunk in chunks
        isempty(chunk) && continue
        chars  = collect(chunk)
        n      = length(chars)
        cpos   = 1   # position within this chunk

        while cpos <= n
            best_len = 0
            best_e   = nothing
            for len in min(max_span, n - cpos + 1):-1:1
                span = String(chars[cpos:cpos+len-1])
                if haskey(𝑆, span)
                    ents   = 𝑆[span]
                    exact  = findfirst(e -> e.surface == span, ents)
                    best_e = exact !== nothing ? ents[exact] : ents[argmin(e.cost for e in ents)]
                    best_len = len; break
                end
            end
            best_e === nothing && return nothing   # OOV morpheme
            push!(nodes, LatticeNode(best_e, pos, pos + best_len))
            pos  += best_len
            cpos += best_len
        end
    end

    isempty(nodes) && return nothing
    hiragana = join(nd.entry.reading for nd in nodes)::String
    return hiragana, nodes
end

# ════════════════════════════════════════════════════════════
# Section 6  COLLECT KN COUNTS  (Threads.@threads :static)
# ════════════════════════════════════════════════════════════

function collect_kn_counts!(kn::KNCounts,
                             𝒯::Vector{Tuple{String,Vector{LatticeNode}}})
    BOS="BOS"; EOS="EOS"
    nslots    = isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : Threads.nthreads() + 4
    local_kns = [KNCounts() for _ in 1:nslots]

    let _T=𝒯, _lkns=local_kns
        Threads.@threads :static for i in eachindex(_T)
            tid = Threads.threadid()
            lkn = _lkns[tid]
            (_,nodes)=_T[i]

            prev=BOS
            for nd in nodes; kn_observe!(lkn,prev,nd.entry.surface); prev=nd.entry.surface; end
            kn_observe!(lkn,prev,EOS)

            if length(nodes)>=2
                p2=BOS; p1=nodes[1].entry.surface
                kn_observe_tri!(lkn,BOS,BOS,p1)
                for j in 2:length(nodes)
                    c=nodes[j].entry.surface
                    kn_observe_tri!(lkn,p2,p1,c)
                    p2=p1; p1=c
                end
                kn_observe_tri!(lkn,p2,p1,EOS)
            end
        end
    end

    for lkn in local_kns
        merge_kn!(kn,lkn)
        empty!(lkn.𝒫_pair); empty!(lkn.𝒫_uni); empty!(lkn.𝒫_cont)
        empty!(lkn.𝒫_triple); empty!(lkn.𝒫_bi4tri)
    end
    GC.gc()
    𝑢=sum(length(v) for v in values(kn.𝒫_pair))
    println("  -> KN bigrams:  $(kn.𝑁_pairs) tokens | $𝑢 unique pairs")
    println("  -> KN trigrams: $(kn.𝑁_trips) tokens")
end

# ════════════════════════════════════════════════════════════
# Section 6b  COLLECT CLASS-BASED KN COUNTS  (Int16 IDs)
# ════════════════════════════════════════════════════════════

function collect_kn_class_counts!(kn_class::KNCountsClass,
                                   𝒯::Vector{Tuple{String,Vector{LatticeNode}}})
    BOS_RID = Int16(0); EOS_LID = Int16(0)
    nslots    = isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : Threads.nthreads() + 4
    local_kns = [KNCountsClass() for _ in 1:nslots]

    let _T=𝒯, _lkns=local_kns
        Threads.@threads :static for i in eachindex(_T)
            tid = Threads.threadid()
            lkn = _lkns[tid]
            (_,nodes)=_T[i]

            # Bigrams: prev.right_id → curr.left_id
            prev_rid = BOS_RID
            for nd in nodes
                kn_observe_class!(lkn, prev_rid, nd.entry.left_id)
                prev_rid = nd.entry.right_id
            end
            kn_observe_class!(lkn, prev_rid, EOS_LID)

            # Trigrams: pp.right_id → prev.right_id → curr.left_id
            if length(nodes) >= 2
                pp_rid = BOS_RID; p_rid = BOS_RID
                kn_observe_tri_class!(lkn, BOS_RID, BOS_RID, nodes[1].entry.left_id)
                p_rid = nodes[1].entry.right_id
                for j in 2:length(nodes)
                    kn_observe_tri_class!(lkn, pp_rid, p_rid, nodes[j].entry.left_id)
                    pp_rid = p_rid
                    p_rid  = nodes[j].entry.right_id
                end
                kn_observe_tri_class!(lkn, pp_rid, p_rid, EOS_LID)
            end
        end
    end

    for lkn in local_kns
        merge_kn_class!(kn_class, lkn)
        empty!(lkn.𝒫_pair); empty!(lkn.𝒫_uni); empty!(lkn.𝒫_cont)
        empty!(lkn.𝒫_triple); empty!(lkn.𝒫_bi4tri)
    end
    GC.gc()
    𝑢=sum(length(v) for v in values(kn_class.𝒫_pair))
    println("  -> Class KN bigrams:  $(kn_class.𝑁_pairs) tokens | $𝑢 unique pairs")
    println("  -> Class KN trigrams: $(kn_class.𝑁_trips) tokens")
end

# ════════════════════════════════════════════════════════════
# Section 7  LATTICE BUILDER
# ════════════════════════════════════════════════════════════

const _BOS = DictEntry("BOS","",Int16(0),Int16(0),Int32(0))
const _EOS = DictEntry("EOS","",Int16(0),Int16(0),Int32(0))

function build_lattice(input::String,
                       𝑅::Dictionary{String,Vector{DictEntry}},
                       max_span::Int=12)::Vector{Vector{LatticeNode}}
    ci = collect(eachindex(input))
    n  = length(ci)
    lattice=[LatticeNode[] for _ in 1:n+2]
    push!(lattice[1], LatticeNode(_BOS,0,1))
    for start in 1:n, len in 1:min(max_span,n-start+1)
        bstart = ci[start]
        bend   = start+len <= n ? prevind(input, ci[start+len]) : lastindex(input)
        span   = SubString(input, bstart, bend)
        if haskey(𝑅, span)
            for e in 𝑅[span]; push!(lattice[start+len], LatticeNode(e,start,start+len)); end
        end
    end
    push!(lattice[n+2], LatticeNode(_EOS,n+1,n+2))
    return lattice
end

# ════════════════════════════════════════════════════════════
# Section 7b  VITERBI PREPROCESSING  (Sudachi matrix.def)
# ════════════════════════════════════════════════════════════

"""
    viterbi_surface(lattice, 𝐖₀) -> Union{Nothing, Vector{LatticeNode}}

Minimal Viterbi using only connection costs (𝐖₀) + node costs.
Used for corpus preprocessing with Sudachi's pre-trained matrix.
Returns the best path (without BOS/EOS), or nothing if unreachable.
"""
function viterbi_surface(
    lattice :: Vector{Vector{LatticeNode}},
    𝐖₀      :: Matrix{Int32},
)::Union{Nothing,Vector{LatticeNode}}
    n  = length(lattice)
    Wr, Wc = size(𝐖₀)
    best_cost = [fill(typemax(Int32), length(lattice[i])) for i in 1:n]
    back_ptr  = [fill((0,0), length(lattice[i])) for i in 1:n]
    best_cost[1][1] = Int32(0)

    for pos in 2:n
        for (j, node) in enumerate(lattice[pos])
            prev_slot = node.begin_pos
            (prev_slot < 1 || isempty(lattice[prev_slot])) && continue
            l = Int(node.entry.left_id) + 1
            for (k, pn) in enumerate(lattice[prev_slot])
                pc = best_cost[prev_slot][k]
                pc == typemax(Int32) && continue
                r = Int(pn.entry.right_id) + 1
                w_conn = (1<=r<=Wr && 1<=l<=Wc) ? @inbounds(𝐖₀[r, l]) : Int32(5000)
                total = pc + w_conn + node.entry.cost
                if total < best_cost[pos][j]
                    best_cost[pos][j] = total
                    back_ptr[pos][j]  = (prev_slot, k)
                end
            end
        end
    end

    j = argmin(best_cost[n])
    best_cost[n][j] == typemax(Int32) && return nothing

    path = LatticeNode[]
    pos = n
    while pos > 0
        push!(path, lattice[pos][j])
        (prev_slot, k) = back_ptr[pos][j]
        prev_slot == 0 && break
        pos = prev_slot; j = k
    end
    reverse!(path)
    return filter(nd -> nd.entry.surface ∉ ("BOS", "EOS"), path)
end

"""
    sentence_to_nodes_viterbi(sentence, 𝑆, 𝐖₀) -> Union{Nothing, Tuple{String, Vector{LatticeNode}}}

Viterbi-based corpus preprocessing using Sudachi's pre-trained matrix.
Replaces greedy longest-match with proper morphological analysis.
Builds a lattice from the surface index 𝑆 and runs Viterbi with 𝐖₀.
"""
function sentence_to_nodes_viterbi(
    sentence :: String,
    𝑆        :: Dictionary{String,Vector{DictEntry}},
    𝐖₀       :: Matrix{Int32};
    max_span :: Int = 12,
)::Union{Nothing,Tuple{String,Vector{LatticeNode}}}
    isempty(sentence) && return nothing
    lattice = build_lattice(sentence, 𝑆, max_span)
    path = viterbi_surface(lattice, 𝐖₀)
    (path === nothing || isempty(path)) && return nothing
    hiragana = join(nd.entry.reading for nd in path)::String
    return hiragana, path
end

"""
    sentence_to_nodes_bunsetsu_viterbi(tabbed, 𝑆, 𝐖₀) -> Union{Nothing, Tuple{String, Vector{LatticeNode}}}

Bunsetsu-segmented variant: gold boundaries from KNBC, Viterbi within
each chunk for correct morpheme-level analysis (POS IDs).
"""
function sentence_to_nodes_bunsetsu_viterbi(
    tabbed   :: String,
    𝑆        :: Dictionary{String,Vector{DictEntry}},
    𝐖₀       :: Matrix{Int32};
    max_span :: Int = 12,
)::Union{Nothing,Tuple{String,Vector{LatticeNode}}}
    chunks = split(tabbed, '\t')
    nodes  = LatticeNode[]
    global_pos = 1

    for chunk in chunks
        isempty(chunk) && continue
        chunk_str = String(chunk)
        n_chars   = length(collect(chunk_str))

        lattice    = build_lattice(chunk_str, 𝑆, max_span)
        chunk_path = viterbi_surface(lattice, 𝐖₀)
        chunk_path === nothing && return nothing

        for nd in chunk_path
            push!(nodes, LatticeNode(nd.entry,
                                     nd.begin_pos + global_pos - 1,
                                     nd.end_pos + global_pos - 1))
        end
        global_pos += n_chars
    end

    isempty(nodes) && return nothing
    hiragana = join(nd.entry.reading for nd in nodes)::String
    return hiragana, nodes
end

# ════════════════════════════════════════════════════════════
# Section 8  VITERBI  (trigram + beam, with context injection)
# ════════════════════════════════════════════════════════════

struct ViterbiResult
    path       :: Vector{LatticeNode}
    total_cost :: Int32
end

"""
viterbi(lattice, 𝐖, kn; ...)

Standard 1-best Viterbi.

Extra keyword args for partial_kkc context injection:
  ctx_prev2  — surface of the node two steps before lattice start (default "BOS")
  ctx_prev1  — surface of the node one step  before lattice start (default "BOS")

Cache language model:
  global_context — Set{String} of surfaces from document context; matched words
                   receive a -1500 cost bonus (massive probability spike)

Class-based N-grams:
  kn_class       — KNCountsClass trained on left_id/right_id transitions
  λ_class        — weight for class-based KN cost (default 1.0)
  ctx_prev2_rid  — right_id of the node two steps before lattice start
  ctx_prev1_rid  — right_id of the node one step before lattice start
"""
function viterbi(lattice   :: Vector{Vector{LatticeNode}},
                 𝐖         :: AbstractMatrix{Int32},
                 kn        :: KNCounts;
                 λ_kn      :: Float32              = 1.0f0,
                 kn_d      :: Float64              = 0.75,
                 beam      :: Int                  = 0,
                 ctx_prev2 :: String               = "BOS",
                 ctx_prev1 :: String               = "BOS",
                 pos_bg    :: Union{POSBigram,Nothing} = nothing,
                 global_context :: Set{String}      = Set{String}(),
                 kn_class  :: Union{KNCountsClass,Nothing} = nothing,
                 λ_class   :: Float32              = 1.0f0,
                 ctx_prev2_rid :: Int16             = Int16(0),
                 ctx_prev1_rid :: Int16             = Int16(0),
                 brown     :: Union{BrownClusters,Nothing} = nothing,
                 𝐖_sem     :: Union{Matrix{Int32},Nothing} = nothing,
                 λ_sem     :: Float32              = 1.0f0)::ViterbiResult

    n  = length(lattice)
    𝑐  = [fill(typemax(Int32), length(lattice[i])) for i in 1:n]
    𝜋  = [fill((0,0),          length(lattice[i])) for i in 1:n]
    σ₂ = [fill(ctx_prev2,      length(lattice[i])) for i in 1:n]
    σ₂_rid = [fill(ctx_prev2_rid, length(lattice[i])) for i in 1:n]
    𝑐[1] .= Int32(0)

    𝑊ᵣ, 𝑊𝑐 = size(𝐖)
    has_ctx   = !isempty(global_context)
    has_class = kn_class !== nothing
    has_sem   = brown !== nothing && 𝐖_sem !== nothing

    for pos in 2:n
        isempty(lattice[pos]) && continue
        _last_ps = -1
        _last_pr = eachindex(lattice[1])
        for (j, node) in enumerate(lattice[pos])
            prev_slot = node.begin_pos
            (prev_slot<1 || isempty(lattice[prev_slot])) && continue

            l    = Int(node.entry.left_id) + 1
            l_ok = 1 <= l <= 𝑊𝑐

            if prev_slot != _last_ps
                _last_ps = prev_slot
                _last_pr = eachindex(lattice[prev_slot])
                if beam>0 && length(lattice[prev_slot])>beam
                    _last_pr = partialsortperm(𝑐[prev_slot], 1:beam)
                end
            end

            for k in _last_pr
                prev_c = 𝑐[prev_slot][k]
                prev_c == typemax(Int32) && continue
                pn = lattice[prev_slot][k]
                r  = Int(pn.entry.right_id) + 1

                𝑤_conn = (l_ok && 1<=r<=𝑊ᵣ) ? @inbounds(𝐖[r,l]) : Int32(5000)

                p1_surf = prev_slot == 1 ? ctx_prev1 : pn.entry.surface
                p2 = σ₂[prev_slot][k]
                𝑤_kn = @fastmath Int32(round(
                    λ_kn * kn_cost_cached(kn, p2, p1_surf, node.entry.surface; d=kn_d)))

                # ── POS bigram cost  (left_id → tag, zero ambiguity) ─
                𝑤_pos = Int32(0)
                if pos_bg !== nothing && !isempty(pos_bg.lid2tag)
                    n_tags = length(pos_bg.tags)
                    prev_tid = prev_slot == 1 ? 1 :
                               get(pos_bg.rid2tag, pn.entry.right_id, 0) + 1
                    curr_tid = get(pos_bg.lid2tag, node.entry.left_id, 0) + 1
                    if 1 <= prev_tid <= n_tags+2 && 1 <= curr_tid <= n_tags+2
                        𝑤_pos = Int32(round(pos_bg.λ_pos *
                                            pos_bg.costs[prev_tid, curr_tid]))
                    end
                end

                # ── Dynamic cache boost ───────────────────────────
                𝑤_cache = (has_ctx && node.entry.surface ∈ global_context) ?
                    Int32(-1500) : Int32(0)

                # ── Class-based KN cost ───────────────────────────
                𝑤_kn_class = Int32(0)
                if has_class
                    p2_rid = σ₂_rid[prev_slot][k]
                    p1_rid = prev_slot == 1 ? ctx_prev1_rid : pn.entry.right_id
                    𝑤_kn_class = @fastmath Int32(round(
                        λ_class * kn_cost_class(kn_class, p2_rid, p1_rid,
                                                node.entry.left_id; d=kn_d)))
                end

                # ── Semantic (Brown cluster) cost ─────────────────
                𝑤_sem = Int32(0)
                if has_sem
                    sc_prev = get_cluster(brown, pn.entry.surface)
                    sc_curr = get_cluster(brown, node.entry.surface)
                    sc_p = Int(sc_prev) + 1
                    sc_c = Int(sc_curr) + 1
                    if 1 <= sc_p <= size(𝐖_sem,1) && 1 <= sc_c <= size(𝐖_sem,2)
                        𝑤_sem = @inbounds Int32(round(λ_sem * 𝐖_sem[sc_p, sc_c]))
                    end
                end

                total = prev_c + 𝑤_conn + 𝑤_kn + 𝑤_pos +
                        node.entry.cost + 𝑤_cache + 𝑤_kn_class + 𝑤_sem
                if total < 𝑐[pos][j]
                    @inbounds 𝑐[pos][j]  = total
                    @inbounds 𝜋[pos][j]  = (prev_slot, k)
                    @inbounds σ₂[pos][j] = p1_surf
                    @inbounds σ₂_rid[pos][j] = prev_slot == 1 ?
                        ctx_prev1_rid : pn.entry.right_id
                end
            end
        end
    end

    path=LatticeNode[]; pos=n; j=argmin(𝑐[n])
    while pos>0
        push!(path, lattice[pos][j])
        (prev_slot, k)=𝜋[pos][j]
        prev_slot==0 && break
        pos=prev_slot; j=k
    end
    reverse!(path)
    ViterbiResult(path, minimum(𝑐[n]))
end

"""
viterbi_topk(lattice, 𝐖, kn; k, ...)

Returns up to k ViterbiResult by back-tracing from the k lowest-cost
terminal states.  Shares the same DP tables as viterbi() — no extra cost
for the forward pass, only k back-traces.

Accepts the same global_context / kn_class / λ_class parameters as viterbi().
"""
function viterbi_topk(lattice :: Vector{Vector{LatticeNode}},
                      𝐖       :: AbstractMatrix{Int32},
                      kn      :: KNCounts;
                      k       :: Int     = 3,
                      λ_kn    :: Float32 = 1.0f0,
                      kn_d    :: Float64 = 0.75,
                      beam    :: Int     = 8,
                      ctx_prev2 :: String = "BOS",
                      ctx_prev1 :: String = "BOS",
                      global_context :: Set{String} = Set{String}(),
                      kn_class  :: Union{KNCountsClass,Nothing} = nothing,
                      λ_class   :: Float32 = 1.0f0,
                      ctx_prev2_rid :: Int16 = Int16(0),
                      ctx_prev1_rid :: Int16 = Int16(0),
                      brown     :: Union{BrownClusters,Nothing} = nothing,
                      𝐖_sem     :: Union{Matrix{Int32},Nothing} = nothing,
                      λ_sem     :: Float32 = 1.0f0)::Vector{ViterbiResult}

    n  = length(lattice)
    𝑐  = [fill(typemax(Int32), length(lattice[i])) for i in 1:n]
    𝜋  = [fill((0,0),          length(lattice[i])) for i in 1:n]
    σ₂ = [fill(ctx_prev2,      length(lattice[i])) for i in 1:n]
    σ₂_rid = [fill(ctx_prev2_rid, length(lattice[i])) for i in 1:n]
    𝑐[1] .= Int32(0)

    𝑊ᵣ, 𝑊𝑐 = size(𝐖)
    has_ctx   = !isempty(global_context)
    has_class = kn_class !== nothing
    has_sem   = brown !== nothing && 𝐖_sem !== nothing

    for pos in 2:n
        isempty(lattice[pos]) && continue
        _last_ps = -1
        _last_pr = eachindex(lattice[1])
        for (j, node) in enumerate(lattice[pos])
            prev_slot = node.begin_pos
            (prev_slot<1 || isempty(lattice[prev_slot])) && continue
            l    = Int(node.entry.left_id) + 1
            l_ok = 1 <= l <= 𝑊𝑐
            if prev_slot != _last_ps
                _last_ps = prev_slot
                _last_pr = eachindex(lattice[prev_slot])
                if beam>0 && length(lattice[prev_slot])>beam
                    _last_pr = partialsortperm(𝑐[prev_slot], 1:beam)
                end
            end
            for kk in _last_pr
                prev_c = 𝑐[prev_slot][kk]
                prev_c == typemax(Int32) && continue
                pn = lattice[prev_slot][kk]
                r  = Int(pn.entry.right_id) + 1
                𝑤_conn = (l_ok && 1<=r<=𝑊ᵣ) ? @inbounds(𝐖[r,l]) : Int32(5000)
                p1_surf = prev_slot == 1 ? ctx_prev1 : pn.entry.surface
                p2 = σ₂[prev_slot][kk]
                𝑤_kn = @fastmath Int32(round(
                    λ_kn * kn_cost_cached(kn, p2, p1_surf, node.entry.surface; d=kn_d)))

                # ── Dynamic cache boost ───────────────────────────
                𝑤_cache = (has_ctx && node.entry.surface ∈ global_context) ?
                    Int32(-1500) : Int32(0)

                # ── Class-based KN cost ───────────────────────────
                𝑤_kn_class = Int32(0)
                if has_class
                    p2_rid = σ₂_rid[prev_slot][kk]
                    p1_rid = prev_slot == 1 ? ctx_prev1_rid : pn.entry.right_id
                    𝑤_kn_class = @fastmath Int32(round(
                        λ_class * kn_cost_class(kn_class, p2_rid, p1_rid,
                                                node.entry.left_id; d=kn_d)))
                end

                # ── Semantic (Brown cluster) cost ─────────────────
                𝑤_sem = Int32(0)
                if has_sem
                    sc_prev = get_cluster(brown, pn.entry.surface)
                    sc_curr = get_cluster(brown, node.entry.surface)
                    sc_p = Int(sc_prev) + 1
                    sc_c = Int(sc_curr) + 1
                    if 1 <= sc_p <= size(𝐖_sem,1) && 1 <= sc_c <= size(𝐖_sem,2)
                        𝑤_sem = @inbounds Int32(round(λ_sem * 𝐖_sem[sc_p, sc_c]))
                    end
                end

                total = prev_c + 𝑤_conn + 𝑤_kn + node.entry.cost +
                        𝑤_cache + 𝑤_kn_class + 𝑤_sem
                if total < 𝑐[pos][j]
                    @inbounds 𝑐[pos][j]  = total
                    @inbounds 𝜋[pos][j]  = (prev_slot, kk)
                    @inbounds σ₂[pos][j] = p1_surf
                    @inbounds σ₂_rid[pos][j] = prev_slot == 1 ?
                        ctx_prev1_rid : pn.entry.right_id
                end
            end
        end
    end

    # Back-trace from top-k terminal states
    n_term  = length(𝑐[n])
    k_actual = min(k, n_term)
    top_js  = partialsortperm(𝑐[n], 1:k_actual)

    results = ViterbiResult[]
    for start_j in top_js
        𝑐[n][start_j] == typemax(Int32) && continue
        path=LatticeNode[]; pos=n; j=start_j
        while pos>0
            push!(path, lattice[pos][j])
            (prev_slot, kk)=𝜋[pos][j]
            prev_slot==0 && break
            pos=prev_slot; j=kk
        end
        reverse!(path)
        push!(results, ViterbiResult(path, 𝑐[n][start_j]))
    end
    results
end

# ════════════════════════════════════════════════════════════
# Section 8b  PARTIAL KKC  (confirmed kanji text + uncommitted hiragana)
# ════════════════════════════════════════════════════════════

"""
    surface_to_ctx(confirmed_text, 𝑆) -> (ctx_prev2, ctx_prev1, rid_prev2, rid_prev1)

Given already-converted kanji/kana text (e.g. "極"), look up the last
1–2 tokens in the surface dictionary 𝑆 (surface→entries) and return
their surfaces as trigram context, plus their right_ids for class-based KN.

Greedy longest-match from the RIGHT end of the string so we get the
most specific segmentation of the tail.
"""
function surface_to_ctx(
    text :: String,
    𝑆    :: Dictionary{String,Vector{DictEntry}};
    max_span :: Int = 12,
)::Tuple{String,String,Int16,Int16}
    isempty(text) && return ("BOS", "BOS", Int16(0), Int16(0))

    chars = collect(text)
    n     = length(chars)

    # Greedy left-to-right segmentation to get all tokens + right_ids
    tokens = Tuple{String,Int16}[]   # (surface, right_id)
    pos = 1
    while pos <= n
        best_len = 0
        best_rid = Int16(0)
        for len in min(max_span, n - pos + 1):-1:1
            span = String(chars[pos:pos+len-1])
            if haskey(𝑆, span)
                ents  = 𝑆[span]
                exact = findfirst(e -> e.surface == span, ents)
                entry = exact !== nothing ? ents[exact] : ents[1]
                best_len = len
                best_rid = entry.right_id
                break
            end
        end
        if best_len > 0
            push!(tokens, (String(chars[pos:pos+best_len-1]), best_rid))
            pos += best_len
        else
            push!(tokens, (String(chars[pos:pos]), Int16(0)))
            pos += 1
        end
    end

    isempty(tokens)     && return ("BOS", "BOS", Int16(0), Int16(0))
    length(tokens) == 1 && return ("BOS", tokens[end][1], Int16(0), tokens[end][2])
    return (tokens[end-1][1], tokens[end][1], tokens[end-1][2], tokens[end][2])
end

# ════════════════════════════════════════════════════════════
# Section 8c  SEGMENT-LEVEL EDITING  (文節変換)
# ════════════════════════════════════════════════════════════

"""
    KkcSegment

One editable segment in a conversion result.

  reading    — original hiragana input for this segment
  surface    — currently selected kanji/kana surface
  candidates — top-k alternatives (index 1 = current best)
  selected   — which candidate is active (1-based)
"""
mutable struct KkcSegment
    reading    :: String
    surface    :: String
    candidates :: Vector{String}
    selected   :: Int
end

"""
    KkcBuffer

Represents the full conversion state of one input string.
Segments can be individually cycled or re-converted with new context.

  segments   — ordered list of KkcSegment
  committed  — already-committed text prefix (kanji, from previous calls)
"""
mutable struct KkcBuffer
    segments  :: Vector{KkcSegment}
    committed :: String
end

"""
    kkc_segment(input, 𝑅, 𝑆, 𝐖, kn; k, left_ctx, λ_kn, kn_d,
                global_context, kn_class, λ_class)
              -> KkcBuffer

Convert a full hiragana string into an editable KkcBuffer.
Each Viterbi node becomes one KkcSegment with top-k candidates.

`left_ctx`       — already-confirmed kanji text to the left (for trigram context).
`global_context` — Set{String} of surfaces from document context (cache LM).
`kn_class`       — class-based KN model (Int16 IDs).
`λ_class`        — weight for class-based KN cost.
"""
function kkc_segment(
    input    :: String,
    𝑅        :: Dictionary{String,Vector{DictEntry}},
    𝑆        :: Dictionary{String,Vector{DictEntry}},
    𝐖        :: AbstractMatrix{Int32},
    kn       :: KNCounts;
    k        :: Int     = 5,
    left_ctx :: String  = "",
    λ_kn     :: Float32 = 1.0f0,
    kn_d     :: Float64 = 0.75,
    global_context :: Set{String} = Set{String}(),
    kn_class :: Union{KNCountsClass,Nothing} = nothing,
    λ_class  :: Float32 = 1.0f0,
)::KkcBuffer

    isempty(input) && return KkcBuffer(KkcSegment[], left_ctx)
    sentinel = ("BOS", "EOS")

    ctx_prev2, ctx_prev1, rid_prev2, rid_prev1 = surface_to_ctx(left_ctx, 𝑆)

    # 1-best path gives us the segmentation boundaries
    best = viterbi(build_lattice(input, 𝑅), 𝐖, kn;
                   λ_kn, kn_d, beam=0, ctx_prev2, ctx_prev1,
                   global_context, kn_class, λ_class,
                   ctx_prev2_rid=rid_prev2, ctx_prev1_rid=rid_prev1)
    nodes_1best = filter(nd -> nd.entry.surface ∉ sentinel, best.path)

    # For each node in 1-best path, get top-k alternatives that span the
    # SAME character range (same begin_pos..end_pos).
    # This keeps segment boundaries fixed while offering surface alternatives.
    segments = KkcSegment[]
    chars    = collect(input)

    # Build char-range → reading slice helper
    char_range_reading(bp, ep) = String(chars[bp:ep-1])

    # Pre-build a position-indexed view of the lattice for alternative lookup
    lattice = build_lattice(input, 𝑅)

    # running left context (updated as we process each segment left→right)
    run_prev2 = ctx_prev2
    run_prev1 = ctx_prev1

    for nd in nodes_1best
        bp = nd.begin_pos
        ep = nd.end_pos
        reading = char_range_reading(bp, ep)

        # Collect all lattice nodes spanning exactly [bp, ep)
        # lattice slot ep contains nodes whose end_pos == ep
        alts = filter(n -> n.begin_pos == bp && n.end_pos == ep,
                      lattice[ep])

        # Score each alternative with trigram cost + node cost
        scored = map(alts) do alt
            w_kn   = kn_cost_cached(kn, run_prev2, run_prev1, alt.entry.surface; d=kn_d)
            total  = Int(alt.entry.cost) + Int(w_kn)
            (total, alt.entry.surface)
        end
        sort!(scored, by = first)

        # Deduplicate surfaces, keep top-k
        seen_surfs = Set{String}()
        cands = String[]
        for (_, surf) in scored
            surf ∈ seen_surfs && continue
            push!(seen_surfs, surf)
            push!(cands, surf)
            length(cands) >= k && break
        end

        # Ensure 1-best surface is always first
        best_surf = nd.entry.surface
        if !isempty(cands) && cands[1] != best_surf
            filter!(s -> s != best_surf, cands)
            pushfirst!(cands, best_surf)
            length(cands) > k && resize!(cands, k)
        end

        push!(segments, KkcSegment(reading, cands[1], cands, 1))

        # Advance running context
        run_prev2 = run_prev1
        run_prev1 = best_surf
    end

    KkcBuffer(segments, left_ctx)
end

"""
    cycle_segment!(buf, idx; forward=true) -> KkcBuffer

Cycle the candidate at segment `idx` forward (or backward).
Updates `surface` and `selected` in place.

    cycle_segment!(buf, 2)           # next candidate for segment 2
    cycle_segment!(buf, 2; forward=false)  # previous candidate
"""
function cycle_segment!(buf::KkcBuffer, idx::Int; forward::Bool=true)::KkcBuffer
    seg = buf.segments[idx]
    n   = length(seg.candidates)
    n == 0 && return buf
    delta = forward ? 1 : -1
    seg.selected = mod1(seg.selected + delta, n)
    seg.surface  = seg.candidates[seg.selected]
    buf
end

"""
    reconv_segment!(buf, idx, 𝑅, 𝑺, 𝐖, kn; k, λ_kn, kn_d) -> KkcBuffer

Re-convert segment `idx` using the current surfaces of neighbouring
segments as fresh trigram context.  Useful after the user has changed
an adjacent segment and wants the target one to update accordingly.

    # segments: [?] [座標]   (idx=1, right neighbour is "座標")
    # reconv scores each candidate c as:
    #   kn(left_prev, left_cur, c) + kn(left_cur, c, "座標") + node_cost(c)
    reconv_segment!(buf, 1, 𝑅, 𝑆, 𝐖̄, kn)

    # After user cycles seg 1 to "局", re-convert seg 2 with left ctx "局":
    #   segments: [局] [?]
    cycle_segment!(buf, 1)
    reconv_segment!(buf, 2, 𝑅, 𝑆, 𝐖̄, kn)
"""
function reconv_segment!(
    buf  :: KkcBuffer,
    idx  :: Int,
    𝑅    :: Dictionary{String,Vector{DictEntry}},
    𝑆    :: Dictionary{String,Vector{DictEntry}},
    𝐖    :: AbstractMatrix{Int32},
    kn   :: KNCounts;
    k    :: Int     = 5,
    λ_kn :: Float32 = 1.0f0,
    kn_d :: Float64 = 0.75,
)::KkcBuffer
    segs = buf.segments
    n    = length(segs)
    idx < 1 || idx > n && return buf

    # Left context: committed text + segments 1..idx-1 (already chosen surfaces)
    left_text = buf.committed *
                join(segs[i].surface for i in 1:idx-1)
    ctx_prev2, ctx_prev1, _, _ = surface_to_ctx(left_text, 𝑆)

    # Right context: the segment immediately to the right (or EOS)
    right_surf = idx < n ? segs[idx+1].surface : "EOS"

    seg     = segs[idx]
    lattice = build_lattice(seg.reading, 𝑅)
    sentinel = ("BOS", "EOS")

    # Score all full-span candidates for this segment's reading.
    # w_kn_l: trigram cost from left context into candidate
    # w_kn_r: trigram cost from candidate into right neighbour
    nchars = length(collect(seg.reading))
    alts   = filter(nd -> nd.begin_pos == 1 && nd.end_pos == nchars + 1,
                    vcat(lattice...))

    scored = map(alts) do alt
        w_kn_l = kn_cost_cached(kn, ctx_prev2, ctx_prev1, alt.entry.surface; d=kn_d)
        w_kn_r = kn_cost_cached(kn, ctx_prev1, alt.entry.surface, right_surf; d=kn_d)
        total  = Int(alt.entry.cost) + Int(w_kn_l) + Int(w_kn_r)
        (total, alt.entry.surface)
    end
    sort!(scored, by = first)

    seen = Set{String}(); cands = String[]
    for (_, surf) in scored
        surf ∈ seen && continue
        push!(seen, surf); push!(cands, surf)
        length(cands) >= k && break
    end

    if !isempty(cands)
        seg.candidates = cands
        seg.selected   = 1
        seg.surface    = cands[1]
    end
    buf
end

"""
    commit_buffer(buf) -> String

Join all segment surfaces into the final converted string.
"""
commit_buffer(buf::KkcBuffer)::String =
    buf.committed * join(seg.surface for seg in buf.segments)

"""
    show_buffer(buf)

Pretty-print the buffer state for debugging / REPL use.

    誰[が]一番[に][着く][か]
    ^--- bracket = segment boundary, current surface shown
"""
function show_buffer(buf::KkcBuffer)
    # Show confirmed left context in angle brackets (context only, not re-converted)
    isempty(buf.committed) || print("<$(buf.committed)>")
    for (i, seg) in enumerate(buf.segments)
        cand_str = length(seg.candidates) > 1 ?
            " {" * join(seg.candidates[1:min(3,end)], "|") * "}" : ""
        print("[$(seg.surface)]$cand_str")
    end
    println()
end

"""
    partial_kkc(confirmed, uncommitted, 𝑅, 𝑺, 𝐖, kn; k, λ_kn, kn_d,
                global_context, kn_class, λ_class)
              -> candidates::Vector{String}

Thin wrapper: convert `uncommitted` hiragana with `confirmed` kanji as
left context.  Returns flat list of top-k candidate strings.

For full segment-level editing use `kkc_segment` instead.
"""
function partial_kkc(
    confirmed   :: String,
    uncommitted :: String,
    𝑅           :: Dictionary{String,Vector{DictEntry}},
    𝑆           :: Dictionary{String,Vector{DictEntry}},
    𝐖           :: AbstractMatrix{Int32},
    kn          :: KNCounts;
    k           :: Int     = 3,
    λ_kn        :: Float32 = 1.0f0,
    kn_d        :: Float64 = 0.75,
    global_context :: Set{String} = Set{String}(),
    kn_class :: Union{KNCountsClass,Nothing} = nothing,
    λ_class  :: Float32 = 1.0f0,
)::Vector{String}

    isempty(uncommitted) && return String[]
    ctx_prev2, ctx_prev1, rid_prev2, rid_prev1 = surface_to_ctx(confirmed, 𝑆)
    sentinel = ("BOS", "EOS")
    results = viterbi_topk(build_lattice(uncommitted, 𝑅), 𝐖, kn;
                            k, λ_kn, kn_d, beam=8,
                            ctx_prev2, ctx_prev1,
                            global_context, kn_class, λ_class,
                            ctx_prev2_rid=rid_prev2, ctx_prev1_rid=rid_prev1)
    [join(nd.entry.surface for nd in r.path if nd.entry.surface ∉ sentinel)
     for r in results]
end

# ════════════════════════════════════════════════════════════
# Section 8d  PARTIAL CONVERSION TRAINING DATA
# ════════════════════════════════════════════════════════════

"""
    PartialContext

Left context for a partial conversion training example.
Full-sentence examples use BOS defaults (right_id=0).
"""
struct PartialContext
    ctx_prev2     :: String
    ctx_prev1     :: String
    ctx_prev2_rid :: Int16
    ctx_prev1_rid :: Int16
end

const _BOS_CTX = PartialContext("BOS", "BOS", Int16(0), Int16(0))

"""
    generate_partial_examples(𝒯; max_splits) -> (partials, contexts)

For each training sentence, generate partial conversion examples that
simulate how IME users convert incrementally.  For a sentence with
nodes [n1, n2, ..., nK], splitting at position `sp` produces a partial
example with:
  - input: concatenated readings of nodes[sp+1:end]
  - gold:  nodes[sp+1:end]
  - context: surfaces/right_ids of nodes[sp-1:sp]

This trains the model to predict correctly when starting from
mid-sentence context (not just BOS), matching real IME usage patterns
like `すももも[convert]ももも[convert]もものうち`.
"""
function generate_partial_examples(
    𝒯          :: Vector{Tuple{String,Vector{LatticeNode}}};
    max_splits :: Int = 2,
)
    partials = Tuple{String,Vector{LatticeNode}}[]
    contexts = PartialContext[]

    for (_, nodes) in 𝒯
        real_nodes = filter(n -> n.entry.surface ∉ ("BOS","EOS"), nodes)
        n = length(real_nodes)
        n < 3 && continue

        # Candidate split positions: leave ≥1 node on each side
        candidates = collect(1:n-1)
        n_pts = min(max_splits, length(candidates))
        points = length(candidates) <= n_pts ?
            candidates :
            sort(candidates[randperm(length(candidates))[1:n_pts]])

        for sp in points
            right_nodes = real_nodes[sp+1:end]
            isempty(right_nodes) && continue

            right_reading = join(String(nd.entry.reading) for nd in right_nodes)
            isempty(right_reading) && continue

            left_nodes = real_nodes[1:sp]
            prev1_surf = String(left_nodes[end].entry.surface)
            prev2_surf = length(left_nodes) >= 2 ?
                String(left_nodes[end-1].entry.surface) : "BOS"
            prev1_rid  = left_nodes[end].entry.right_id
            prev2_rid  = length(left_nodes) >= 2 ?
                left_nodes[end-1].entry.right_id : Int16(0)

            push!(partials, (right_reading, right_nodes))
            push!(contexts, PartialContext(prev2_surf, prev1_surf,
                                           prev2_rid, prev1_rid))
        end
    end

    println("  -> $(length(partials)) partial training examples generated")
    return partials, contexts
end

# ════════════════════════════════════════════════════════════
# Section 9  AVERAGED PERCEPTRON  (top-k matching, variant η)
# ════════════════════════════════════════════════════════════

function train(;
    sudachidict_csvs  :: Vector{String},
    corpus_sources    :: Vector{Tuple{Symbol,String}} = Tuple{Symbol,String}[],
    corpus_bin_path   :: String          = "data/cache/corpus.bin",
    syn_path          :: String          = "",
    syn_max_variants  :: Int             = 3,
    syn_inject_weight :: Float64         = 0.3,
    epochs            :: Int             = 10,
    λ_kn              :: Float32         = 1.0f0,
    kn_d              :: Float64         = 0.75,
    beam              :: Int             = 8,
    η₀                :: Float32         = 1.0f0,
    ρ                 :: Float32         = 0.85f0,
    η_min             :: Float32         = 1.0f0,
    output_path       :: String          = "tuned_model.bin",
    dict_bin_path     :: String          = "data/cache/sudachidict.bin",
    user_dict_path    :: String          = "data/user_dict.tsv",
    sudachi_matrix_path :: String        = "data/sudachidict/matrix.def",
    mini_batch        :: Int             = 4096,
    force_rebuild     :: Bool            = false,
    topk_match        :: Int             = 3,      # correct if in top-k predictions
    variant_lr_scale  :: Float32         = 0.2f0,  # η multiplier for synonym variants
    grad_accum        :: Int             = 1,       # chunks to accumulate before applying to 𝐖
    n_brown_clusters  :: Int             = 256,     # Brown cluster count for semantic matrix
    λ_sem             :: Float32         = 1.0f0,   # weight for semantic cost in Viterbi
    partial_splits    :: Int             = 2,        # max split points per sentence for partial IME training
)
    println("\n=== Loading SudachiDict ===")
    𝑅, 𝑆, 𝑚𝑙, 𝑚𝑟 = load_or_build_dict_with_user(sudachidict_csvs;
                                                    bin_path=dict_bin_path,
                                                    user_dict_path,
                                                    force_rebuild)

    # Load Sudachi's pre-trained matrix for corpus preprocessing & 𝐖 initialization
    𝐖₀ = if isfile(sudachi_matrix_path)
        println("\n=== Loading Sudachi Matrix ===")
        load_sudachi_matrix(sudachi_matrix_path)
    else
        println("  (no Sudachi matrix.def — using zero initialization)")
        nothing
    end

    # Initialize 𝐖 from Sudachi's pre-trained matrix (better baseline than zeros)
    𝐖 = zeros(Int32, Int(𝑚𝑟)+2, Int(𝑚𝑙)+2)
    if 𝐖₀ !== nothing
        nr, nc = size(𝐖₀)
        for r in 1:min(nr, size(𝐖,1)), c in 1:min(nc, size(𝐖,2))
            @inbounds 𝐖[r, c] = 𝐖₀[r, c]
        end
        println("  𝐖 initialized from Sudachi matrix ($(min(nr,size(𝐖,1)))×$(min(nc,size(𝐖,2))) overlap)")
    end
    𝐖̄_acc = zeros(Int64, size(𝐖)...)
    println("𝐖 shape: $(size(𝐖,1)) x $(size(𝐖,2))   threads: $(Threads.nthreads())")

    println("\n=== Loading Corpora ===")
    all_sents = load_or_build_corpus(corpus_sources;
                                     bin_path      = corpus_bin_path,
                                     force_rebuild = force_rebuild)
    println("Total raw sentences: $(length(all_sents))")

    # ── Synonym augmentation (phase A: before lattice building) ──
    syn_is_var = fill(false, length(all_sents))
    if !isempty(syn_path)
        all_sents, syn_is_var = synonym_pipeline!(
            all_sents, KNCounts(), syn_path;
            max_variants  = syn_max_variants,
            inject_weight = syn_inject_weight,
            phase         = :augment,
        )
    end

    println("\n=== Building Training Pairs (Threads.@threads :static) ===")
    raw_pairs=Vector{Union{Nothing,Tuple{String,Vector{LatticeNode}}}}(undef,length(all_sents))
    let _rp=raw_pairs, _sents=all_sents, _S=𝑆, _W0=𝐖₀
        Threads.@threads :static for i in eachindex(_sents)
            s = _sents[i]
            # Use Viterbi with Sudachi matrix for proper morphological analysis
            _rp[i] = if _W0 !== nothing
                occursin('\t', s) ?
                    sentence_to_nodes_bunsetsu_viterbi(s, _S, _W0) :
                    sentence_to_nodes_viterbi(s, _S, _W0)
            else
                occursin('\t', s) ?
                    sentence_to_nodes_bunsetsu(s, _S) :
                    sentence_to_nodes(s, _S)
            end
        end
    end

    # Preserve is_variant flag alongside training pairs
    𝒯        = Tuple{String,Vector{LatticeNode}}[]
    𝒯_is_var = Bool[]
    for (i, p) in enumerate(raw_pairs)
        if p !== nothing
            push!(𝒯,        p)
            push!(𝒯_is_var, syn_is_var[i])
        end
    end
    println("Usable: $(length(𝒯)) / $(length(all_sents))")
    empty!(all_sents); GC.gc()

    println("\n=== POS Bigram Model ===")
    knbc_dir = ""
    for (kind, path) in corpus_sources
        kind ∈ (:knbc_bunsetsu, :knbc) && (knbc_dir = path; break)
    end
    pos_bg = if !isempty(knbc_dir) && isdir(knbc_dir)
        learn_pos_bigrams(sudachidict_csvs, knbc_dir)
    else
        println("  (no KNBC dir — POS bigram disabled)")
        POSBigram()
    end

    println("\n=== KN Counts (Threads.@threads :static) ===")
    kn=KNCounts()
    collect_kn_counts!(kn,𝒯)
    reset_kn_cache!()

    println("\n=== Class-based KN Counts ===")
    kn_class = KNCountsClass()
    collect_kn_class_counts!(kn_class, 𝒯)

    # ── Synonym cost injection (phase B: after KN counting) ──────
    if !isempty(syn_path)
        synonym_pipeline!(
            String[], kn, syn_path;
            inject_weight = syn_inject_weight,
            phase         = :inject,
        )
    end

    # ── Brown Clustering ─────────────────────────────────────────
    println("\n=== Brown Clustering ===")
    brown = brown_cluster(𝒯, n_brown_clusters)
    𝑛_sem = brown.n_clusters + 1   # +1 for cluster 0 (BOS/EOS/unknown)
    𝐖_sem      = zeros(Int32, 𝑛_sem, 𝑛_sem)
    𝐖_sem_acc  = zeros(Int64, 𝑛_sem, 𝑛_sem)
    println("  𝐖_sem shape: $(𝑛_sem) x $(𝑛_sem)")

    # 𝑆 (surface→entries) is kept alive for partial_kkc context lookup
    GC.gc()

    # ── Partial conversion training data ─────────────────────────
    println("\n=== Partial Conversion Training ===")
    n_full = length(𝒯)
    𝒯_partial, 𝒯_partial_ctx = generate_partial_examples(𝒯;
                                                           max_splits=partial_splits)

    # Contexts: full sentences use BOS context
    𝒯_ctx = fill(_BOS_CTX, n_full)

    # Append partial examples (after KN counting to avoid skewing n-gram stats)
    append!(𝒯, 𝒯_partial)
    append!(𝒯_is_var, fill(false, length(𝒯_partial)))
    append!(𝒯_ctx, 𝒯_partial_ctx)
    println("  Training set: $n_full full + $(length(𝒯_partial)) partial = $(length(𝒯)) total")

    println("\n=== Sanity Check ===")
    diagnose(𝒯[1],𝑅,𝐖,kn)

    println("\n=== Averaged Perceptron  ($epochs epochs x mini_batch=$mini_batch) ===")
    𝑛t  = isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : Threads.nthreads() + 4
    Δ_buf = [Pair{Int,Float32}[] for _ in 1:𝑛t]
    Δ_sem_buf = [Pair{Int,Float32}[] for _ in 1:𝑛t]   # semantic matrix deltas
    for v in Δ_buf; sizehint!(v, 512); end
    for v in Δ_sem_buf; sizehint!(v, 128); end
    last_step = zeros(Int32, size(𝐖)...)
    last_step_sem = zeros(Int32, size(𝐖_sem)...)
    𝑊ᵣ, 𝑊𝑐 = size(𝐖)
    𝑆ᵣ, 𝑆𝑐 = size(𝐖_sem)
    println("  threads=$𝑛t  sentences=$(length(𝒯))")

    # ── apply_chunk! ─────────────────────────────────────────────
    # accum_merged: accumulated deltas across multiple chunks before applying
    accum_merged  = Dict{Int, Float32}()
    accum_sem_merged = Dict{Int, Float32}()   # semantic matrix accumulator
    accum_steps   = 0   # how many chunks have been accumulated
    sizehint!(accum_merged, 65536)
    sizehint!(accum_sem_merged, 4096)

    function apply_chunk!(chunk_indices::AbstractVector{Int}, lr::Float32,
                          is_var::Vector{Bool}, variant_scale::Float32,
                          topk::Int, grad_accum::Int)
        for buf in Δ_buf; empty!(buf); end
        for buf in Δ_sem_buf; empty!(buf); end
        nerr = 0
        let _T=𝒯, _R=𝑅, _W=𝐖, _kn=kn,
            _lkn=λ_kn, _knd=kn_d, _bm=beam,
            _dB=Δ_buf, _dS=Δ_sem_buf, _Wr=𝑊ᵣ, _Wc=𝑊𝑐, _lr=lr,
            _Sr=𝑆ᵣ, _Sc=𝑆𝑐, _brown=brown,
            _isvar=is_var, _vscale=variant_scale, _topk=topk,
            _idx=chunk_indices, _ctx=𝒯_ctx

            nerr_atomic = Threads.Atomic{Int}(0)
            Threads.@threads :static for ii in eachindex(_idx)
                i   = _idx[ii]
                tid = Threads.threadid()

                eff_lr = _isvar[i] ?
                    max(1.0f0, _lr * _vscale) : _lr

                (input, correct) = _T[i]
                ctx = _ctx[i]
                lattice = build_lattice(input, _R)

                # For partial examples, set BOS right_id to context's right_id
                # so W connections match the real predecessor
                if ctx.ctx_prev1_rid != Int16(0)
                    bos = DictEntry("BOS", "", Int16(0), ctx.ctx_prev1_rid, Int32(0))
                    lattice[1][1] = LatticeNode(bos, 0, 1)
                end

                results = viterbi_topk(lattice, _W, _kn;
                                        k=_topk, λ_kn=_lkn, kn_d=_knd, beam=_bm,
                                        ctx_prev2=ctx.ctx_prev2, ctx_prev1=ctx.ctx_prev1,
                                        ctx_prev2_rid=ctx.ctx_prev2_rid,
                                        ctx_prev1_rid=ctx.ctx_prev1_rid)

                correct_surfs = [nd.entry.surface for nd in correct
                                  if nd.entry.surface ∉ ("BOS","EOS")]

                pred_match = any(results) do res
                    [nd.entry.surface for nd in res.path
                     if nd.entry.surface ∉ ("BOS","EOS")] == correct_surfs
                end

                if !pred_match
                    Threads.atomic_add!(nerr_atomic, 1)
                    buf = _dB[tid]
                    predicted = results[1].path

                    # Standard perceptron update: reward gold, penalize predicted
                    for k in 1:length(correct)-1
                        r=Int(correct[k].entry.right_id)  +1
                        l=Int(correct[k+1].entry.left_id) +1
                        if 1<=r<=_Wr && 1<=l<=_Wc
                            push!(buf, (r + (l-1)*_Wr) => -eff_lr)
                        end
                    end
                    if !isempty(correct)
                        r0=Int(ctx.ctx_prev1_rid)+1; l0=Int(correct[1].entry.left_id)+1
                        if 1<=r0<=_Wr && 1<=l0<=_Wc; push!(buf, (r0+(l0-1)*_Wr) => -eff_lr); end
                        rn=Int(correct[end].entry.right_id)+1; ln=Int(_EOS.left_id)+1
                        if 1<=rn<=_Wr && 1<=ln<=_Wc; push!(buf, (rn+(ln-1)*_Wr) => -eff_lr); end
                    end
                    for k in 1:length(predicted)-1
                        r=Int(predicted[k].entry.right_id)  +1
                        l=Int(predicted[k+1].entry.left_id) +1
                        if 1<=r<=_Wr && 1<=l<=_Wc
                            push!(buf, (r + (l-1)*_Wr) => eff_lr)
                        end
                    end

                    # ── Semantic matrix (Brown cluster) update ─────
                    sem_buf = _dS[tid]
                    # Gold cluster transitions: reward
                    for k in 1:length(correct)-1
                        sc_a = Int(get_cluster(_brown, correct[k].entry.surface)) + 1
                        sc_b = Int(get_cluster(_brown, correct[k+1].entry.surface)) + 1
                        if 1<=sc_a<=_Sr && 1<=sc_b<=_Sc
                            push!(sem_buf, (sc_a + (sc_b-1)*_Sr) => -eff_lr)
                        end
                    end
                    # Predicted cluster transitions: penalize
                    for k in 1:length(predicted)-1
                        sc_a = Int(get_cluster(_brown, predicted[k].entry.surface)) + 1
                        sc_b = Int(get_cluster(_brown, predicted[k+1].entry.surface)) + 1
                        if 1<=sc_a<=_Sr && 1<=sc_b<=_Sc
                            push!(sem_buf, (sc_a + (sc_b-1)*_Sr) => eff_lr)
                        end
                    end
                end
            end
            nerr = nerr_atomic[]
        end

        # ── Gradient accumulation ─────────────────────────────
        # Merge per-thread deltas into accum_merged (not yet applied to 𝐖)
        for buf in Δ_buf
            for (idx, v) in buf
                accum_merged[idx] = get(accum_merged, idx, 0.0f0) + v
            end
        end
        for buf in Δ_sem_buf
            for (idx, v) in buf
                accum_sem_merged[idx] = get(accum_sem_merged, idx, 0.0f0) + v
            end
        end
        accum_steps += 1

        # Apply to 𝐖 and 𝐖_sem only every grad_accum chunks (or at end of epoch)
        if accum_steps >= grad_accum
            𝑛_acc += 1
            for (idx, dv) in accum_merged
                gap = 𝑛_acc - 1 - last_step[idx]
                if gap > 0
                    @inbounds 𝐖̄_acc[idx] += Int64(𝐖[idx]) * Int64(gap)
                end
                @inbounds 𝐖[idx] += round(Int32, dv)
                @inbounds 𝐖̄_acc[idx] += Int64(𝐖[idx])
                @inbounds last_step[idx] = Int32(𝑛_acc)
            end
            for (idx, dv) in accum_sem_merged
                gap = 𝑛_acc - 1 - last_step_sem[idx]
                if gap > 0
                    @inbounds 𝐖_sem_acc[idx] += Int64(𝐖_sem[idx]) * Int64(gap)
                end
                @inbounds 𝐖_sem[idx] += round(Int32, dv)
                @inbounds 𝐖_sem_acc[idx] += Int64(𝐖_sem[idx])
                @inbounds last_step_sem[idx] = Int32(𝑛_acc)
            end
            empty!(accum_merged)
            empty!(accum_sem_merged)
            accum_steps = 0
        end

        return nerr
    end

    # Flush any remaining accumulated gradients (called at epoch end)
    function flush_accum!()
        (isempty(accum_merged) && isempty(accum_sem_merged)) && return
        𝑛_acc += 1
        for (idx, dv) in accum_merged
            gap = 𝑛_acc - 1 - last_step[idx]
            if gap > 0
                @inbounds 𝐖̄_acc[idx] += Int64(𝐖[idx]) * Int64(gap)
            end
            @inbounds 𝐖[idx] += round(Int32, dv)
            @inbounds 𝐖̄_acc[idx] += Int64(𝐖[idx])
            @inbounds last_step[idx] = Int32(𝑛_acc)
        end
        for (idx, dv) in accum_sem_merged
            gap = 𝑛_acc - 1 - last_step_sem[idx]
            if gap > 0
                @inbounds 𝐖_sem_acc[idx] += Int64(𝐖_sem[idx]) * Int64(gap)
            end
            @inbounds 𝐖_sem[idx] += round(Int32, dv)
            @inbounds 𝐖_sem_acc[idx] += Int64(𝐖_sem[idx])
            @inbounds last_step_sem[idx] = Int32(𝑛_acc)
        end
        empty!(accum_merged)
        empty!(accum_sem_merged)
        accum_steps = 0
    end

    𝑛_acc = 0

    for epoch in 1:epochs
        η = max(η₀ * Float32(ρ^(epoch-1)), η_min)
        idx_perm = randperm(length(𝒯))
        𝑛err = 0; 𝑛chunks = 0
        𝑡_epoch = time()
        total_chunks = cld(length(𝒯), mini_batch)
        for chunk in Iterators.partition(idx_perm, mini_batch)
            𝑡_chunk = time()
            𝑛err   += apply_chunk!(collect(chunk), η, 𝒯_is_var, variant_lr_scale, topk_match, grad_accum)
            𝑛chunks += 1
            𝑡_el = round(time() - 𝑡_chunk, digits=1)
            print("    chunk $𝑛chunks/$total_chunks  ($(𝑡_el)s)  errors=$(lpad(𝑛err,6))  pending=$(length(accum_merged))\r")
            flush(stdout)
        end
        flush_accum!()

        # ── Per-epoch diagnostics ──────────────────────────────────────
        # Sentence-level exact-match accuracy (existing metric)
        pct_sent = round(100(1 - 𝑛err/length(𝒯)), digits=1)
        𝑡_ep = round(time() - 𝑡_epoch, digits=1)

        # Morpheme-level accuracy: score each morpheme independently
        # (faster to compute on a sample — use first 2000 sentences)
        n_sample    = min(2000, length(𝒯))
        morph_ok    = 0
        morph_total = 0
        sentinel    = ("BOS", "EOS")
        for i in 1:n_sample
            (input, gold_nodes) = 𝒯[i]
            ctx = 𝒯_ctx[i]
            lattice_diag = build_lattice(input, 𝑅)
            if ctx.ctx_prev1_rid != Int16(0)
                bos = DictEntry("BOS","",Int16(0),ctx.ctx_prev1_rid,Int32(0))
                lattice_diag[1][1] = LatticeNode(bos, 0, 1)
            end
            pred = viterbi(lattice_diag, 𝐖, kn; λ_kn, kn_d, beam=0,
                           ctx_prev2=ctx.ctx_prev2, ctx_prev1=ctx.ctx_prev1,
                           ctx_prev2_rid=ctx.ctx_prev2_rid,
                           ctx_prev1_rid=ctx.ctx_prev1_rid)
            gold_surfs = [nd.entry.surface for nd in gold_nodes if nd.entry.surface ∉ sentinel]
            pred_surfs = [nd.entry.surface for nd in pred.path  if nd.entry.surface ∉ sentinel]
            # align by position: zip to shorter length
            for (g, p) in zip(gold_surfs, pred_surfs)
                morph_total += 1
                g == p && (morph_ok += 1)
            end
            # count unmatched gold morphemes as errors
            morph_total += abs(length(gold_surfs) - length(pred_surfs))
        end
        pct_morph = round(100 * morph_ok / max(1, morph_total), digits=1)

        println("  Epoch $epoch  η=$(round(η, digits=2))  chunks=$𝑛chunks  " *
                "sent: $𝑛err errors / $(length(𝒯))  ($(pct_sent)% sent-exact)  " *
                "morph: $(pct_morph)% (sample=$n_sample)  $(𝑡_ep)s")
        𝑛err==0 && (println("  Converged!"); break)
    end

    @inbounds for k in eachindex(𝐖)
        gap = 𝑛_acc - last_step[k]
        if gap > 0; 𝐖̄_acc[k] += Int64(𝐖[k]) * Int64(gap); end
    end
    @inbounds for k in eachindex(𝐖_sem)
        gap = 𝑛_acc - last_step_sem[k]
        if gap > 0; 𝐖_sem_acc[k] += Int64(𝐖_sem[k]) * Int64(gap); end
    end

    𝐖̄=Matrix{Int32}(undef,size(𝐖)...)
    𝑛_acc_safe = max(𝑛_acc, 1)
    @inbounds for k in eachindex(𝐖̄)
        𝐖̄[k]=Int32(clamp(𝐖̄_acc[k]÷𝑛_acc_safe, typemin(Int32), typemax(Int32)))
    end
    𝐖̄_sem = Matrix{Int32}(undef, size(𝐖_sem)...)
    @inbounds for k in eachindex(𝐖̄_sem)
        𝐖̄_sem[k] = Int32(clamp(𝐖_sem_acc[k] ÷ 𝑛_acc_safe, typemin(Int32), typemax(Int32)))
    end
    println("  𝐖̄ averaged over $𝑛_acc chunk snapshots")
    println("  𝐖̄_sem averaged ($(size(𝐖̄_sem,1))×$(size(𝐖̄_sem,2)))")

    println("\n=== Saving -> $output_path ===")
    serialize(output_path, (𝐖̄, kn, kn_class, brown, 𝐖̄_sem))
    println("Done!")
    return 𝐖̄, kn, kn_class, 𝑅, 𝑆, pos_bg, brown, 𝐖̄_sem
end

# ════════════════════════════════════════════════════════════
# Section 10  DIAGNOSTICS
# ════════════════════════════════════════════════════════════

function diagnose(pair::Tuple{String,Vector{LatticeNode}},
                  𝑅::Dictionary{String,Vector{DictEntry}},
                  𝐖::AbstractMatrix{Int32},
                  kn::KNCounts;
                  λ_kn::Float32=1.0f0, kn_d::Float64=0.75, k::Int=3)
    input,correct_path=pair
    sentinel=("BOS","EOS")
    println("\n-- DIAGNOSE -----------------------------------------------")
    println("Input hiragana:    ", input)
    println("Correct surfaces:  ", [nd.entry.surface for nd in correct_path if nd.entry.surface ∉ sentinel])
    lattice=build_lattice(input,𝑅)
    results=viterbi_topk(lattice,𝐖,kn;k,λ_kn,kn_d)
    for (rank, res) in enumerate(results)
        surfs = [nd.entry.surface for nd in res.path if nd.entry.surface ∉ sentinel]
        println("  Top-$rank predicted: ", surfs, "  (cost=$(res.total_cost))")
    end
    correct_surfs = [nd.entry.surface for nd in correct_path if nd.entry.surface ∉ sentinel]
    in_topk = any(r -> [nd.entry.surface for nd in r.path if nd.entry.surface ∉ sentinel] == correct_surfs, results)
    println("Match in top-$k: ", in_topk)
    println("-----------------------------------------------------------\n")
end

# ════════════════════════════════════════════════════════════
# Section 11  INFERENCE
# ════════════════════════════════════════════════════════════

"""1-best full conversion."""
function kkc(input::String,
             𝑅 :: Dictionary{String,Vector{DictEntry}},
             𝐖 :: AbstractMatrix{Int32},
             kn :: KNCounts;
             λ_kn   :: Float32                 = 1.0f0,
             kn_d   :: Float64                 = 0.75,
             pos_bg :: Union{POSBigram,Nothing} = nothing,
             global_context :: Set{String}      = Set{String}(),
             kn_class :: Union{KNCountsClass,Nothing} = nothing,
             λ_class  :: Float32               = 1.0f0,
             brown    :: Union{BrownClusters,Nothing} = nothing,
             𝐖_sem    :: Union{Matrix{Int32},Nothing} = nothing,
             λ_sem    :: Float32               = 1.0f0)::String
    result = viterbi(build_lattice(input,𝑅), 𝐖, kn;
                     λ_kn, kn_d, beam=0, pos_bg, global_context, kn_class, λ_class,
                     brown, 𝐖_sem, λ_sem)
    join(nd.entry.surface for nd in result.path if nd.entry.surface ∉ ("BOS","EOS"))
end

"""Top-k full conversion."""
function kkc_topk(input::String,
                  𝑅      :: Dictionary{String,Vector{DictEntry}},
                  𝐖      :: AbstractMatrix{Int32},
                  kn     :: KNCounts;
                  k      :: Int                    = 3,
                  λ_kn   :: Float32                = 1.0f0,
                  kn_d   :: Float64                = 0.75,
                  pos_bg :: Union{POSBigram,Nothing} = nothing,
                  global_context :: Set{String}     = Set{String}(),
                  kn_class :: Union{KNCountsClass,Nothing} = nothing,
                  λ_class  :: Float32              = 1.0f0,
                  brown    :: Union{BrownClusters,Nothing} = nothing,
                  𝐖_sem    :: Union{Matrix{Int32},Nothing} = nothing,
                  λ_sem    :: Float32              = 1.0f0)::Vector{String}
    results = viterbi_topk(build_lattice(input,𝑅), 𝐖, kn;
                           k, λ_kn, kn_d, beam=8, global_context, kn_class, λ_class,
                           brown, 𝐖_sem, λ_sem)
    [join(nd.entry.surface for nd in r.path if nd.entry.surface ∉ ("BOS","EOS"))
     for r in results]
end

load_model(path::String) = deserialize(path)

# ════════════════════════════════════════════════════════════
# Section 12  EXPORT
# ════════════════════════════════════════════════════════════

function export_matrix_def(𝐖::AbstractMatrix{Int32}, path::String)
    𝑑ᵣ,𝑑𝑐=size(𝐖); buf=IOBuffer()
    println(buf,"$(𝑑ᵣ-1) $(𝑑𝑐-1)")
    for l in 1:𝑑𝑐, r in 1:𝑑ᵣ
        @inbounds v=𝐖[r,l]
        v!=0 && println(buf,"$(r-1) $(l-1) $v")
    end
    write(path,take!(buf)); println("Exported -> $path")
end

function export_kn_tsv(kn::KNCounts, path::String; kn_d::Float64=0.75)
    open(path,"w") do f
        println(f,"# a\tb\tkn_cost  (trigram-backed, kappa=$κ, d=$kn_d)")
        for (a,inner) in sort(collect(pairs(kn.𝒫_pair)),by=first)
            for b in sort(collect(keys(inner)))
                println(f,"$a\t$b\t$(kn_cost(kn,"BOS",a,b;d=kn_d))")
            end
        end
    end
    println("Exported -> $path ($(sum(length(v) for v in values(kn.𝒫_pair))) pairs)")
end

# ════════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════════

𝐖̄, kn, kn_class, 𝑅, 𝑆, pos_bg, brown, 𝐖̄_sem = train(
    sudachidict_csvs  = ["data/sudachidict/small_lex.csv",
                         "data/sudachidict/core_lex.csv"],
    corpus_sources    = [(:snow,   "data/snow/T15-2020.1.7.csv"),
                         (:snow,   "data/snow/T23-2020.1.7.csv"),
                         # (:tanaka, "data/tanaka/examples.utf"),
                         (:knbc_bunsetsu, "data/knbc/KNBC_v1.0_090925_utf8/corpus1")],
    corpus_bin_path   = "data/cache/corpus.bin",
    syn_path          = "data/sudachidict/synonyms.txt",
    syn_max_variants  = 3,
    syn_inject_weight = 0.3,
    epochs            = 10,
    λ_kn              = 1.0f0,
    kn_d              = 0.75,
    beam              = 8,
    η₀                = 10.0f0,   # schedule: 3→2→2→1→1…  (was 1.0 = always clamped to η_min)
    ρ                 = 0.85f0,
    η_min             = 1.0f0,
    mini_batch        = 4096,
    topk_match        = 3,
    variant_lr_scale  = 0.2f0,
    grad_accum        = 1,    # update 𝐖 every chunk (~4k sentences) — lower=faster early convergence
    user_dict_path    = "data/user_dict.tsv",
    sudachi_matrix_path = "data/sudachidict/matrix.def",
    n_brown_clusters  = 256,
    λ_sem             = 1.0f0,
    partial_splits    = 2,        # split each sentence up to 2 ways for partial IME training
)

# ── 1-best inference (with class-based KN + semantic) ──────
println(kkc("とうきょうとっきょきょかきょく", 𝑅, 𝐖̄, kn; pos_bg, kn_class, brown, 𝐖_sem=𝐖̄_sem))
println(kkc("きしゃのきしゃはきしゃできしゃした", 𝑅, 𝐖̄, kn; pos_bg, kn_class, brown, 𝐖_sem=𝐖̄_sem))

# ── top-3 inference ───────────────────────────────────────
println("\nTop-3 candidates:")
for c in kkc_topk("とうきょうとっきょきょかきょく", 𝑅, 𝐖̄, kn; k=3, pos_bg, kn_class, brown, 𝐖_sem=𝐖̄_sem)
    println("  ", c)
end

# ── segment editing demo ─────────────────────────────────
println("\n=== Segment Editing Demo ===")

# Example 1: compound word — user edits first char context
println("\nきょくざひょう (full):")
buf = kkc_segment("きょくざひょう", 𝑅, 𝑆, 𝐖̄, kn; k=5, kn_class)
show_buffer(buf)
println("  commit: ", commit_buffer(buf))

# Cycle segment 1 to next candidate
println("  after cycle_segment!(1):")
cycle_segment!(buf, 1)
show_buffer(buf)

# Example 2: middle segment edit with context propagation
println("\nわたくしにはわかりません:")
buf2 = kkc_segment("わたくしにはわかりません", 𝑅, 𝑆, 𝐖̄, kn; k=5, kn_class)
show_buffer(buf2)

# User changes segment 1 (私→わたくし), then re-converts segment 2 with new context
println("  cycle seg 1, then reconv seg 2:")
cycle_segment!(buf2, 1)
reconv_segment!(buf2, 2, 𝑅, 𝑆, 𝐖̄, kn)
show_buffer(buf2)
println("  commit: ", commit_buffer(buf2))

# Example 3: left context from confirmed kanji
println("\n極 (confirmed) + ざひょう (uncommitted):")
buf3 = kkc_segment("ざひょう", 𝑅, 𝑆, 𝐖̄, kn; k=5, left_ctx="極", kn_class)
show_buffer(buf3)
println("  commit: ", commit_buffer(buf3))

println("\n曲 (confirmed) + ちょう (uncommitted):")
buf4 = kkc_segment("ちょう", 𝑅, 𝑆, 𝐖̄, kn; k=5, left_ctx="曲", kn_class)
show_buffer(buf4)
println("  commit: ", commit_buffer(buf4))

# ── Context-aware demo (Cache Language Model) ─────────────
println("\n=== Context-Aware Demo (Cache LM) ===")
doc_context = "貴社の記者は汽車で帰社した。記者会見は午後三時から始まる。"
ctx_vocab = extract_context_vocabulary(doc_context, 𝑆)
println("Context vocabulary: ", ctx_vocab)

println("\nWithout context:")
println("  ", kkc("きしゃのきしゃはきしゃできしゃした", 𝑅, 𝐖̄, kn; kn_class, brown, 𝐖_sem=𝐖̄_sem))
println("With context:")
println("  ", kkc("きしゃのきしゃはきしゃできしゃした", 𝑅, 𝐖̄, kn;
                   kn_class, global_context=ctx_vocab, brown, 𝐖_sem=𝐖̄_sem))

export_matrix_def(𝐖̄, "matrix.def")
export_kn_tsv(kn,     "kn_bigram.tsv")