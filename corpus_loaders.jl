# SPDX-License-Identifier: Apache-2.0
#
# corpus_loaders.jl  ―  Corpus loaders for KKC training
#
#   load_snow(path)        SNOW T15/T23  (CC-BY 4.0)
#   load_tanaka(path)      Tanaka Corpus (CC-BY)
#   load_knbc(dir)         KNBコーパス   (BSD-3)  — flat surface strings
#   load_knbc_bunsetsu(dir) KNBコーパス  — bunsetsu-segmented pairs
#
#   load_or_build_corpus(sources; bin_path, force_rebuild)
#       Unified entry-point with binary cache.
#       source kinds: :snow  :tanaka  :knbc  :knbc_bunsetsu
#
#       :knbc_bunsetsu returns sentences as tab-separated bunsetsu strings:
#           "surface1\tsurface2\t..."
#       The caller (sentence_to_nodes_bunsetsu) splits on '\t' to get
#       gold-standard segment boundaries for training.
#
# Depends on: CSV, DataFrames, Serialization (loaded by kkc_tuner_fast.jl)

const CORPUS_MAGIC   = UInt32(0x4B434F52)   # "KCOR"
const CORPUS_VERSION = UInt32(1)

# ── raw parsers ───────────────────────────────────────────────────────────────

function load_snow(path::String)::Vector{String}
    df   = CSV.read(path, DataFrame; header=true, delim=',', silencewarnings=true)
    cols = names(df)
    g    = something(findfirst(c -> occursin("原文",c) && !occursin("英語",c), cols), 2)
    y    = something(findfirst(c -> occursin("やさしい",c), cols), 3)
    out  = String[]
    for row in eachrow(df), col in (g, y)
        v = row[col]; ismissing(v) && continue
        s = strip(string(v)); isempty(s) || push!(out, s)
    end
    seen       = Set{String}()
    unique_out = filter(s -> s ∉ seen && (push!(seen, s); true), out)
    println("  -> $(length(unique_out)) sentences: $path")
    return unique_out
end

function load_tanaka(path::String)::Vector{String}
    out = String[]
    if endswith(path, ".csv")
        df = CSV.read(path, DataFrame; header=true, silencewarnings=true)
        jc = something(findfirst(
            c -> !occursin("ID", uppercase(c)) && !occursin("EN", uppercase(c)),
            names(df)), 1)
        for row in eachrow(df)
            v = row[jc]; ismissing(v) && continue
            s = strip(string(v)); isempty(s) || push!(out, s)
        end
    else
        open(path) do f
            for line in eachline(f)
                s = strip(split(line, '\t')[1])
                s = startswith(s, "A: ") ? s[4:end] : s
                isempty(s) || push!(out, s)
            end
        end
    end
    println("  -> $(length(out)) sentences: $path")
    return out
end

"""
    parse_knbc_file(path) -> Vector{Tuple{String,String}}

Parse one KNP sentence file and return a list of (surface, reading)
pairs — one per bunsetsu (文節).

KNP format rules:
  `# ...`  — sentence header           → skip
  `* nD …` — bunsetsu boundary start   → flush current segment, start new
  `+ mD …` — tag boundary (within seg) → skip
  `EOS`    — end of sentence            → flush last segment
  otherwise: `surface reading base pos…` morpheme line → collect col 1 & 2
"""
function parse_knbc_file(path::String)::Vector{Tuple{String,String}}
    segments = Tuple{String,String}[]
    cur_surfs = String[]
    cur_reads = String[]

    flush!() = if !isempty(cur_surfs)
        push!(segments, (join(cur_surfs), join(cur_reads)))
        empty!(cur_surfs); empty!(cur_reads)
    end

    open(path) do f
        for raw in eachline(f)
            line = strip(raw)
            isempty(line) && continue
            if startswith(line, "# ") || line == "EOS"
                flush!(); continue
            end
            if startswith(line, "* ")   # bunsetsu boundary
                flush!(); continue
            end
            startswith(line, "+ ") && continue   # tag boundary within bunsetsu
            cols = split(line, ' ')
            length(cols) < 2 && continue
            push!(cur_surfs, cols[1])
            push!(cur_reads, cols[2])
        end
    end
    flush!()
    return segments
end

"""
    load_knbc(dir) -> Vector{String}

Load KNBC sentences as flat surface strings (original behaviour).
Used when bunsetsu boundaries are not needed.
"""
function load_knbc(dir::String)::Vector{String}
    paths = _knbc_paths(dir)
    out   = String[]
    skip  = 0
    for path in paths
        segs = parse_knbc_file(path)
        if isempty(segs); skip += 1; continue; end
        sent = join(s for (s, _) in segs)
        isempty(strip(sent)) && (skip += 1; continue)
        push!(out, sent)
    end
    println("  -> $(length(out)) sentences  ($skip skipped): $dir")
    return out
end

"""
    load_knbc_bunsetsu(dir) -> Vector{String}

Load KNBC sentences with gold bunsetsu boundaries encoded as
tab-separated surface chunks:

    "と\tいう\tことで、\t「京都観光」の\t趣旨からは\t…"

`sentence_to_nodes_bunsetsu` splits on '\\t' to recover each segment and
uses the boundary positions as gold-standard labels for perceptron training,
bypassing the need for Viterbi to guess segmentation from scratch.

Average: ~6.6 bunsetsu per sentence, 4186 sentences, 27792 total segments.
"""
function load_knbc_bunsetsu(dir::String)::Vector{String}
    paths = _knbc_paths(dir)
    out   = String[]
    skip  = 0
    for path in paths
        segs = parse_knbc_file(path)
        if length(segs) < 2; skip += 1; continue; end
        # Encode as TAB-separated surface chunks
        # (readings are implicit — lattice lookup will recover them)
        push!(out, join((s for (s, _) in segs), '\t'))
    end
    println("  -> $(length(out)) bunsetsu-segmented sentences  ($skip skipped): $dir")
    return out
end

# helper: collect all leaf file paths under a KNBC dir (or single file)
function _knbc_paths(dir::String)::Vector{String}
    isfile(dir) && return [dir]
    paths = String[]
    for (root, _, files) in walkdir(dir)
        for f in sort(files); push!(paths, joinpath(root, f)); end
    end
    return paths
end

# ── binary cache ──────────────────────────────────────────────────────────────

# Format:
#   UInt32  CORPUS_MAGIC
#   UInt32  CORPUS_VERSION
#   UInt64  n_sentences
#   UInt64  heap_bytes
#   n × UInt64  byte-offsets into heap
#   n × UInt32  byte-lengths
#   heap  (all UTF-8 sentence bytes concatenated)

function write_corpus_bin(sentences::Vector{String}, path::String)
    n       = length(sentences)
    offsets = Vector{UInt64}(undef, n)
    lengths = Vector{UInt32}(undef, n)
    heap_io = IOBuffer()
    for (i, s) in enumerate(sentences)
        offsets[i] = UInt64(position(heap_io))
        lengths[i] = UInt32(ncodeunits(s))
        write(heap_io, s)
    end
    heap = take!(heap_io)
    mkpath(dirname(path))
    open(path, "w") do f
        write(f, CORPUS_MAGIC);       write(f, CORPUS_VERSION)
        write(f, UInt64(n));          write(f, UInt64(length(heap)))
        write(f, offsets);            write(f, lengths)
        write(f, heap)
    end
    sz = round(filesize(path) / 1e6, digits=1)
    println("  -> $path  ($sz MB,  $n sentences)")
end

function load_corpus_bin(path::String)::Vector{String}
    data = read(path)
    pos  = Ref(1)
    rd_u32() = (v = reinterpret(UInt32, data[pos[]:pos[]+3])[1]; pos[] += 4; v)
    rd_u64() = (v = reinterpret(UInt64, data[pos[]:pos[]+7])[1]; pos[] += 8; v)
    rd_u32() == CORPUS_MAGIC   || error("Bad magic in $path")
    rd_u32() == CORPUS_VERSION || error("Corpus cache version mismatch: $path")
    n          = Int(rd_u64())
    heap_bytes = Int(rd_u64())
    offsets    = [rd_u64() for _ in 1:n]
    lengths    = [rd_u32() for _ in 1:n]
    heap_start = pos[]
    heap       = @view data[heap_start : heap_start + heap_bytes - 1]
    sentences  = Vector{String}(undef, n)
    for i in 1:n
        off          = Int(offsets[i]) + 1
        nb           = Int(lengths[i])
        sentences[i] = String(copy(heap[off : off + nb - 1]))
    end
    println("  -> $n sentences loaded from cache: $path")
    return sentences
end

# ── unified entry-point ───────────────────────────────────────────────────────

"""
    load_or_build_corpus(sources; bin_path, force_rebuild) -> Vector{String}

Parse raw corpus files once and cache to binary, or load from cache
when it is still fresh.  The cache is rebuilt when any source file/dir
has a newer mtime than `bin_path`.

`sources` — iterable of `(kind::Symbol, path::String)` pairs.
  Supported kinds: `:snow`, `:tanaka`, `:knbc`, `:knbc_bunsetsu`

`:knbc_bunsetsu` sentences are tab-separated bunsetsu surfaces.
Pass them to `sentence_to_nodes_bunsetsu` instead of `sentence_to_nodes`.

Example:
    sents = load_or_build_corpus(
        [(:snow,           "data/snow/T15-2020.1.csv"),
         (:tanaka,         "data/tanaka/examples.utf"),
         (:knbc_bunsetsu,  "data/knbc/KNBC_v1.0_090925_utf8/corpus1")];
        bin_path = "data/cache/corpus.bin",
    )
"""
function load_or_build_corpus(
    sources;
    bin_path      :: String = "data/cache/corpus.bin",
    force_rebuild :: Bool   = false,
)::Vector{String}

    source_mtimes(kind::Symbol, path::String) =
        ((kind == :knbc || kind == :knbc_bunsetsu) && isdir(path)) ?
            [mtime(joinpath(r, f)) for (r,_,fs) in walkdir(path) for f in fs] :
            isfile(path) ? [mtime(path)] : Float64[]

    need_rebuild = force_rebuild || !isfile(bin_path)
    if !need_rebuild
        bin_t = mtime(bin_path)
        for (kind, path) in sources
            if any(t -> t > bin_t, source_mtimes(kind, path))
                println("  Source newer than cache: $path")
                need_rebuild = true; break
            end
        end
    end

    if need_rebuild
        println("Building corpus cache -> $bin_path")
        all_sents = String[]
        for (kind, path) in sources
            sents = if kind == :snow;            load_snow(path)
                    elseif kind == :tanaka;       load_tanaka(path)
                    elseif kind == :knbc;         load_knbc(path)
                    elseif kind == :knbc_bunsetsu; load_knbc_bunsetsu(path)
                    else error("Unknown corpus kind: $kind")
                    end
            append!(all_sents, sents)
        end
        println("  Total: $(length(all_sents)) sentences")
        write_corpus_bin(all_sents, bin_path)
        return all_sents
    else
        println("Corpus cache is fresh")
        return load_corpus_bin(bin_path)
    end
end

# ════════════════════════════════════════════════════════════
# POS BIGRAM LEARNING
# ════════════════════════════════════════════════════════════

# KNP (KNBC) → Sudachi POS tag mapping
# KNP uses JUMAN tag scheme; Sudachi uses its own hierarchy.
# This table maps the most common KNP "品詞-品詞細分類" strings
# to the corresponding Sudachi "品詞1-品詞2" strings.
const _KNP_TO_SUDACHI = Dict{String,String}(
    "名詞-普通名詞"         => "名詞-普通名詞",
    "名詞-固有名詞"         => "名詞-固有名詞",
    "名詞-サ変名詞"         => "名詞-普通名詞",   # Sudachi: サ変可能 is a sub-tag
    "名詞-数詞"             => "名詞-数詞",
    "名詞-形式名詞"         => "名詞-普通名詞",
    "名詞-副詞的名詞"       => "副詞",
    "名詞-時相名詞"         => "名詞-普通名詞",
    "名詞-地名"             => "名詞-固有名詞",
    "名詞-人名"             => "名詞-固有名詞",
    "名詞-組織名"           => "名詞-固有名詞",
    "動詞"                  => "動詞",
    "形容詞"                => "形容詞",
    "形容動詞"              => "形状詞",
    "副詞"                  => "副詞",
    "接続詞"                => "接続詞",
    "感動詞"                => "感動詞",
    "接頭辞-名詞接頭辞"     => "接頭辞",
    "接頭辞-動詞接頭辞"     => "接頭辞",
    "接頭辞-形容詞接頭辞"   => "接頭辞",
    "接頭辞"                => "接頭辞",
    "接尾辞-名詞性名詞接尾辞"   => "接尾辞",
    "接尾辞-名詞性名詞助数辞"   => "接尾辞",
    "接尾辞-動詞性接尾辞"       => "接尾辞",
    "接尾辞-形容詞性接尾辞"     => "接尾辞",
    "接尾辞-形容詞性述語接尾辞" => "接尾辞",
    "接尾辞"                    => "接尾辞",
    "助詞-格助詞"           => "助詞",
    "助詞-副助詞"           => "助詞",
    "助詞-接続助詞"         => "助詞",
    "助詞-終助詞"           => "助詞",
    "助詞-間投助詞"         => "助詞",
    "助詞"                  => "助詞",
    "助動詞"                => "助動詞",
    "判定詞"                => "助動詞",
    "指示詞-名詞形指示詞"   => "代名詞",
    "指示詞-副詞形指示詞"   => "副詞",
    "指示詞"                => "代名詞",
    "特殊-句点"             => "補助記号",
    "特殊-読点"             => "補助記号",
    "特殊-記号"             => "記号",
    "特殊-空白"             => "空白",
    "特殊"                  => "記号",
)

function _knp_to_sudachi(pos::String, pos2::String)::String
    full = pos2 == "*" ? pos : pos * "-" * pos2
    get(_KNP_TO_SUDACHI, full, get(_KNP_TO_SUDACHI, pos, pos))
end

"""
    learn_pos_bigrams(csv_paths, knbc_dir; scale, λ_pos) -> POSBigram

Build a learned POS bigram cost table by:
1. Parsing SudachiDict CSVs → left_id/right_id → Sudachi POS tag (exact)
2. Parsing KNBC gold KNP annotations → POS bigram counts in Sudachi-tag space
3. Fitting -log P(tag_b | tag_a) × scale as Int32 costs

Viterbi uses `node.entry.left_id` (current) and `pn.entry.right_id` (previous)
for a completely unambiguous, surface-free POS lookup.
"""
function learn_pos_bigrams(
    csv_paths :: Vector{String},
    knbc_dir  :: String;
    scale     :: Float32 = 800f0,
    λ_pos     :: Float32 = 1.0f0,
)::POSBigram

    # ── Step 1: left_id / right_id → Sudachi POS tag ─────────────────────
    lid2tag_idx, rid2tag_idx, tag2id, tags = build_lid_pos_table(csv_paths)
    n = length(tags)

    # Reverse: tag index → tag string (for bigram counting by tag name)
    idx2tag = Dict{Int,String}(i => t for (t,i) in tag2id)

    # ── Step 2: count POS bigrams from KNBC annotations ──────────────────
    # KNBC uses KNP/JUMAN POS scheme → map to Sudachi via _knp_to_sudachi
    bos_id  = 1              # index 1 reserved for BOS
    eos_id  = n + 2          # index n+2 reserved for EOS
    id_of_sudachi(t) = get(tag2id, t, 0) + 1  # +1 offset; 0→1 for unknowns

    bigram  = zeros(Int, n+2, n+2)   # [prev, curr]
    prev_marginal = zeros(Int, n+2)

    paths = _knbc_paths(knbc_dir)
    for path in paths
        prev_id = bos_id
        open(path) do f
            for raw in eachline(f)
                line = strip(raw)
                if isempty(line) || startswith(line, '#') ||
                   startswith(line, '*') || startswith(line, '+')
                    continue
                end
                if line == "EOS"
                    bigram[prev_id, eos_id]    += 1
                    prev_marginal[prev_id]      += 1
                    prev_id = bos_id
                    continue
                end
                cols = split(line, ' ')
                length(cols) < 7 && continue
                knp_pos  = String(cols[4])
                knp_pos2 = String(cols[6])
                sud_tag  = _knp_to_sudachi(knp_pos, knp_pos2)
                curr_id  = id_of_sudachi(sud_tag)
                curr_id  = clamp(curr_id, 1, n+2)
                bigram[prev_id, curr_id]  += 1
                prev_marginal[prev_id]    += 1
                prev_id = curr_id
            end
        end
    end

    total_pairs = sum(bigram)
    println("  POS bigram: $n tags, $total_pairs observed transitions from KNBC")

    # ── Step 3: Laplace-smoothed -log P(curr | prev) × scale ─────────────
    mat = Matrix{Int32}(undef, n+2, n+2)
    uniform_cost = Int32(round(scale * log(Float32(n+2))))
    for prev in 1:n+2
        denom = prev_marginal[prev] + n + 2   # add-1 Laplace
        for curr in 1:n+2
            cnt  = bigram[prev, curr]
            p    = (cnt + 1) / denom
            mat[prev, curr] = Int32(round(clamp(scale * (-log(p)), 0f0, 30000f0)))
        end
    end

    return POSBigram(tags, tag2id, lid2tag_idx, rid2tag_idx, mat, scale, λ_pos)
end