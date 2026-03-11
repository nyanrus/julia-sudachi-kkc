# SPDX-License-Identifier: Apache-2.0
#
# dict_loader.jl  ―  SudachiDict binary cache + CSV loader
#
#   load_sudachidict(csv_paths)              parse CSV → (𝑅, 𝑆, max_l, max_r)
#   write_dict_bin(𝑅, 𝑆, path)             serialise to fast mmap binary
#   load_dict_bin(path)                      load from binary cache
#   load_or_build_dict(csv_paths; ...)       auto-select CSV or cache
#
# Depends on: DictEntry (data_structures.jl), CSV, DataFrames,
#             InternedStrings, InlineStrings, Mmap, GC

# Section 1b  BINARY DICT  (mmap cache)
# ════════════════════════════════════════════════════════════

const DICT_MAGIC   = UInt32(0x4B4B4344)
const DICT_VERSION = UInt32(1)

function write_dict_bin(
    𝑅    :: Dictionary{String,Vector{DictEntry}},
    𝑆    :: Dictionary{String,Vector{DictEntry}},
    path :: String,
)
    println("  Writing binary dict -> $path")
    str2id   = Dict{String,UInt32}()
    heap_io  = IOBuffer()
    str_offs = UInt32[]
    str_lens = UInt32[]

    intern_str!(s::String)::UInt32 = get!(str2id, s) do
        id  = UInt32(length(str_offs))
        off = UInt32(position(heap_io))
        nb  = UInt32(sizeof(s))
        write(heap_io, s)
        push!(str_offs, off); push!(str_lens, nb)
        id
    end

    entry2id = Dict{DictEntry,UInt32}(); entries = DictEntry[]
    intern_entry!(e::DictEntry)::UInt32 = get!(entry2id, e) do
        id = UInt32(length(entries)); push!(entries, e); id
    end

    for evec in 𝑅, e in evec; intern_str!(e.surface); intern_str!(e.reading); intern_entry!(e); end
    for evec in 𝑆, e in evec; intern_str!(e.surface); intern_str!(e.reading); intern_entry!(e); end

    heap_bytes = take!(heap_io)
    n_str = length(str_offs); n_ent = length(entries)

    build_buckets(idx) = sort!(
        [(intern_str!(k), [entry2id[e] for e in v]) for (k,v) in pairs(idx)],
        by = first)

    𝑅_bkts = build_buckets(𝑅)
    𝑆_bkts = build_buckets(𝑆)

    open(path, "w") do f
        write(f, DICT_MAGIC); write(f, DICT_VERSION)
        write(f, UInt64(n_str));  write(f, UInt64(length(heap_bytes)))
        write(f, UInt64(n_ent));  write(f, UInt64(length(𝑅_bkts)))
        write(f, UInt64(length(𝑆_bkts))); write(f, UInt64(0))
        write(f, heap_bytes)
        for i in 1:n_str; write(f, str_offs[i]); write(f, str_lens[i]); end
        for e in entries
            write(f, str2id[e.surface]); write(f, str2id[e.reading])
            write(f, e.left_id); write(f, e.right_id)
            write(f, e.cost);    write(f, UInt32(0))
        end
        for bkts in (𝑅_bkts, 𝑆_bkts)
            for (kid, eids) in bkts
                write(f, kid); write(f, UInt32(length(eids)))
                for eid in eids; write(f, eid); end
            end
        end
    end
    sz = round(filesize(path)/1e6, digits=1)
    println("  -> $path  ($sz MB,  $n_str strings, $n_ent entries)")
end

function load_dict_bin(path::String)
    println("  Loading binary dict <- $path")
    io   = open(path, "r")
    data = Mmap.mmap(io)
    pos  = Ref(1)

    rd_u32() = (v=reinterpret(UInt32,data[pos[]:pos[]+3])[1]; pos[]+=4; v)
    rd_i16() = (v=reinterpret(Int16, data[pos[]:pos[]+1])[1]; pos[]+=2; v)
    rd_i32() = (v=reinterpret(Int32, data[pos[]:pos[]+3])[1]; pos[]+=4; v)
    rd_u64() = (v=reinterpret(UInt64,data[pos[]:pos[]+7])[1]; pos[]+=8; v)

    rd_u32() == DICT_MAGIC   || error("Bad magic in $path")
    rd_u32() == DICT_VERSION || error("Dict version mismatch in $path")

    n_str      = Int(rd_u64())
    heap_bytes = Int(rd_u64())
    n_ent      = Int(rd_u64())
    n_𝑅        = Int(rd_u64())
    n_𝑆        = Int(rd_u64())
    rd_u64()

    heap_start = pos[]
    pos[] = heap_start + heap_bytes

    str_offs = Vector{UInt32}(undef, n_str)
    str_lens = Vector{UInt32}(undef, n_str)
    for i in 1:n_str; str_offs[i]=rd_u32(); str_lens[i]=rd_u32(); end

    heap_str(id::UInt32)::String = begin
        off = Int(str_offs[id+1]) + heap_start
        nb  = Int(str_lens[id+1])
        String(StringView(@view data[off:off+nb-1]))
    end

    entries = Vector{DictEntry}(undef, n_ent)
    𝑚𝑙 = Int16(0); 𝑚𝑟 = Int16(0)
    for i in 1:n_ent
        s=rd_u32(); r=rd_u32(); l=rd_i16(); ri=rd_i16(); c=rd_i32(); rd_u32()
        entries[i] = DictEntry(heap_str(s), heap_str(r), l, ri, c)
        𝑚𝑙 = max(𝑚𝑙,l); 𝑚𝑟 = max(𝑚𝑟,ri)
    end

    𝑅 = Dictionary{String,Vector{DictEntry}}()
    𝑆 = Dictionary{String,Vector{DictEntry}}()
    read_bkts!(idx, n) = for _ in 1:n
        kid=rd_u32(); m=Int(rd_u32())
        ev=[entries[Int(rd_u32())+1] for _ in 1:m]
        insert!(idx, heap_str(kid), ev)
    end
    read_bkts!(𝑅, n_𝑅); read_bkts!(𝑆, n_𝑆)
    close(io)
    println("  -> $n_ent entries loaded (mmap, zero-copy)")
    return 𝑅, 𝑆, 𝑚𝑙, 𝑚𝑟
end

"""
    build_lid_pos_table(csv_paths) -> (lid2tag, rid2tag, tag2id, tags)

Parse SudachiDict CSV files and build:
  lid2tag  :: Dict{Int16,Int}  — left_id  → tag index (1-based)
  rid2tag  :: Dict{Int16,Int}  — right_id → tag index (1-based)
  tag2id   :: Dict{String,Int} — Sudachi POS tag string → index
  tags     :: Vector{String}   — sorted tag list

SudachiDict CSV column layout (all variants):
  col 1: surface, col 2: left_id, col 3: right_id, col 4: cost
  col 6: 品詞1 (POS level 1), col 7: 品詞2 (POS level 2)

Tag = "品詞1-品詞2" when 品詞2 ∉ ("*",""), else just "品詞1".
Both left_id and right_id map to the same tag (they encode
the same POS, just from the left/right connection perspective).
"""
function build_lid_pos_table(csv_paths::Vector{String})
    lid2pos = Dict{Int16,String}()
    rid2pos = Dict{Int16,String}()

    for path in csv_paths
        isfile(path) || continue
        df = CSV.read(path, DataFrame;
            header=false, delim=',',
            types=Dict(2=>Int16, 3=>Int16, 4=>Int32),
            silencewarnings=true)
        ncol(df) < 7 && continue
        for row in eachrow(df)
            (ismissing(row[2]) || ismissing(row[3])) && continue
            l   = Int16(row[2])
            r   = Int16(row[3])
            p1  = ismissing(row[6]) ? "" : strip(string(row[6]))
            p2  = ncol(df) >= 7 && !ismissing(row[7]) ? strip(string(row[7])) : "*"
            tag = (isempty(p2) || p2 == "*") ? p1 : p1 * "-" * p2
            isempty(tag) && continue
            haskey(lid2pos, l) || (lid2pos[l] = tag)
            haskey(rid2pos, r) || (rid2pos[r] = tag)
        end
    end

    all_tags = sort!(collect(Set(values(lid2pos)) ∪ Set(values(rid2pos))))
    tag2id   = Dict{String,Int}(t => i for (i,t) in enumerate(all_tags))
    # +1 offset: index 1=BOS, 2..n+1=tags, n+2=EOS  (applied at call site)
    lid2tag  = Dict{Int16,Int}(l => tag2id[t]+1 for (l,t) in lid2pos)
    rid2tag  = Dict{Int16,Int}(r => tag2id[t]+1 for (r,t) in rid2pos)

    println("  POS table: $(length(all_tags)) tags from $(length(lid2tag)) left_ids")
    return lid2tag, rid2tag, tag2id, all_tags
end

function load_or_build_dict(csv_paths::Vector{String};
                             bin_path::String    = "data/sudachidict/dict.bin",
                             force_rebuild::Bool = false)
    need = force_rebuild || !isfile(bin_path) ||
           any(mtime(p) > mtime(bin_path) for p in csv_paths if isfile(p))
    if need
        𝑅, 𝑆, 𝑚𝑙, 𝑚𝑟 = load_sudachidict(csv_paths)
        mkpath(dirname(bin_path))
        write_dict_bin(𝑅, 𝑆, bin_path)
        return 𝑅, 𝑆, 𝑚𝑙, 𝑚𝑟
    else
        println("  Binary dict cache is fresh")
        return load_dict_bin(bin_path)
    end
end


# ════════════════════════════════════════════════════════════
# USER DICTIONARY  (lightweight additions on top of SudachiDict)
# ════════════════════════════════════════════════════════════

"""
    load_user_dict(path) -> Vector{DictEntry}

Load a plain-text user dictionary.  Each non-blank, non-comment line:

    surface\treading\tcost
    surface\treading           # cost defaults to 3000 (lower = preferred)
    surface\treading\tcost\tleft_id\tright_id

Fields:
  surface   — display form, e.g. "許可局"
  reading   — hiragana,    e.g. "きょかきょく"
  cost      — connection cost (lower = more preferred; default 3000)
  left_id / right_id — POS connection IDs (default 1 / 1, generic noun)

Lines starting with `#` are comments.

Example file (user_dict.tsv):
    # 東京特許許可局
    許可局\tきょかきょく\t2000
    東京特許許可局\tとうきょうとっきょきょかきょく\t1500
"""
function load_user_dict(path::String)::Vector{DictEntry}
    entries = DictEntry[]
    open(path) do f
        for raw in eachline(f)
            line = strip(raw)
            isempty(line) || startswith(line, '#') && continue
            cols = split(line, '\t')
            length(cols) < 2 && continue
            surface  = string(cols[1])
            reading  = string(cols[2])
            cost     = length(cols) >= 3 ? tryparse(Int32, cols[3]) : nothing
            cost     = something(cost, Int32(3000))
            left_id  = length(cols) >= 4 ? something(tryparse(Int16, cols[4]), Int16(1)) : Int16(1)
            right_id = length(cols) >= 5 ? something(tryparse(Int16, cols[5]), Int16(1)) : Int16(1)
            push!(entries, DictEntry(surface, reading, left_id, right_id, cost))
        end
    end
    println("  -> $(length(entries)) user dict entries: $path")
    return entries
end

"""
    inject_user_dict!(𝑅, 𝑺, entries) -> (new_𝑚𝑙, new_𝑚𝑟)

Insert user DictEntry list into the live reading-index 𝑅 and surface-index 𝑆.
Call this after `load_or_build_dict` and before training / inference.
Returns updated max left_id and right_id (needed to resize 𝐖 if they grow).

Existing entries for the same reading/surface are kept — the user entry
is prepended so it appears first (lowest-index = tried first in viterbi).
"""
function inject_user_dict!(
    𝑅       :: Dictionary{String,Vector{DictEntry}},
    𝑆       :: Dictionary{String,Vector{DictEntry}},
    entries :: Vector{DictEntry},
)::Tuple{Int16,Int16}
    𝑚𝑙 = Int16(0); 𝑚𝑟 = Int16(0)
    for e in entries
        # Reading index
        if haskey(𝑅, e.reading)
            pushfirst!(𝑅[e.reading], e)
        else
            insert!(𝑅, e.reading, DictEntry[e])
        end
        # Surface index
        if haskey(𝑆, e.surface)
            pushfirst!(𝑆[e.surface], e)
        else
            insert!(𝑆, e.surface, DictEntry[e])
        end
        𝑚𝑙 = max(𝑚𝑙, e.left_id)
        𝑚𝑟 = max(𝑚𝑟, e.right_id)
    end
    println("  injected $(length(entries)) user entries into live dict")
    return 𝑚𝑙, 𝑚𝑟
end

"""
    load_or_build_dict_with_user(csv_paths; bin_path, user_dict_path, force_rebuild)

Convenience wrapper: load SudachiDict (cached) then inject user dict on top.
The user dict is always re-applied from the TSV file (no binary caching needed —
it's tiny and fast to parse every run).
"""
function load_or_build_dict_with_user(
    csv_paths      :: Vector{String};
    bin_path       :: String = "data/sudachidict/dict.bin",
    user_dict_path :: String = "",
    force_rebuild  :: Bool   = false,
)
    𝑅, 𝑆, 𝑚𝑙, 𝑚𝑟 = load_or_build_dict(csv_paths; bin_path, force_rebuild)
    if !isempty(user_dict_path) && isfile(user_dict_path)
        entries = load_user_dict(user_dict_path)
        u𝑚𝑙, u𝑚𝑟 = inject_user_dict!(𝑅, 𝑆, entries)
        𝑚𝑙 = max(𝑚𝑙, u𝑚𝑙)
        𝑚𝑟 = max(𝑚𝑟, u𝑚𝑟)
    end
    return 𝑅, 𝑆, 𝑚𝑙, 𝑚𝑟
end

# ════════════════════════════════════════════════════════════
# Section 2  KNESER-NEY  (bigram + trigram)
# ════════════════════════════════════════════════════════════

const κ = 1000.0

function kn_observe!(kn::KNCounts, a::String, b::String)
    inner  = get!(Dictionary{String,Int}, kn.𝒫_pair, a)
    is_new = !haskey(inner, b)
    set!(inner, b, get(inner, b, 0) + 1)
    set!(kn.𝒫_uni, a, get(kn.𝒫_uni, a, 0) + 1)
    is_new && set!(kn.𝒫_cont, b, get(kn.𝒫_cont, b, 0) + 1)
    kn.𝑁_pairs += 1
end

function kn_observe_tri!(kn::KNCounts, a::String, b::String, c::String)
    ab  = get!(Dictionary{String, Dictionary{String,Int}}, kn.𝒫_triple, a)
    bc  = get!(Dictionary{String,Int}, ab, b)
    set!(bc, c, get(bc, c, 0) + 1)
    ab2 = get!(Dictionary{String,Int}, kn.𝒫_bi4tri, a)
    set!(ab2, b, get(ab2, b, 0) + 1)
    kn.𝑁_trips += 1
end

function merge_kn!(dst::KNCounts, src::KNCounts)
    for (a, inner_src) in pairs(src.𝒫_pair)
        inner_dst = get!(Dictionary{String,Int}, dst.𝒫_pair, a)
        for (b, cnt) in pairs(inner_src)
            prev = get(inner_dst, b, 0)
            set!(inner_dst, b, prev + cnt)
            prev == 0 && set!(dst.𝒫_cont, b, get(dst.𝒫_cont, b, 0) + 1)
        end
        set!(dst.𝒫_uni, a, get(dst.𝒫_uni, a, 0) + get(src.𝒫_uni, a, 0))
    end
    dst.𝑁_pairs += src.𝑁_pairs

    for (a, ab_src) in pairs(src.𝒫_triple)
        ab_dst = get!(Dictionary{String, Dictionary{String,Int}}, dst.𝒫_triple, a)
        for (b, bc_src) in pairs(ab_src)
            bc_dst = get!(Dictionary{String,Int}, ab_dst, b)
            for (c, cnt) in pairs(bc_src); set!(bc_dst, c, get(bc_dst, c, 0) + cnt); end
        end
    end
    for (a, ab_src) in pairs(src.𝒫_bi4tri)
        ab_dst = get!(Dictionary{String,Int}, dst.𝒫_bi4tri, a)
        for (b, cnt) in pairs(ab_src); set!(ab_dst, b, get(ab_dst, b, 0) + cnt); end
    end
    dst.𝑁_trips += src.𝑁_trips
end

function ℓ_kn_bigram(kn::KNCounts, a::String, b::String; d::Float64=0.75)::Float64
    𝑁  = max(kn.𝑁_pairs, 1)
    p̃  = max(get(kn.𝒫_cont, b, 0), 1) / 𝑁
    Cₐ = get(kn.𝒫_uni, a, 0)
    Cₐ == 0 && return log(p̃)
    Cₐᵦ = haskey(kn.𝒫_pair, a) ? get(kn.𝒫_pair[a], b, 0) : 0
    λ   = d * (haskey(kn.𝒫_pair, a) ? length(kn.𝒫_pair[a]) : 0) / Cₐ
    log(max(max(Cₐᵦ - d, 0.0) / Cₐ + λ * p̃, 1e-10))
end

function ℓ_kn_trigram(kn::KNCounts, a::String, b::String, c::String;
                       d::Float64=0.75)::Float64
    (!haskey(kn.𝒫_triple, a) || !haskey(kn.𝒫_triple[a], b)) &&
        return ℓ_kn_bigram(kn, b, c; d)
    C_ab = haskey(kn.𝒫_bi4tri, a) ? get(kn.𝒫_bi4tri[a], b, 0) : 0
    C_ab == 0 && return ℓ_kn_bigram(kn, b, c; d)
    C_abc = get(kn.𝒫_triple[a][b], c, 0)
    λ     = d * length(kn.𝒫_triple[a][b]) / C_ab
    log(max(max(C_abc - d, 0.0) / C_ab + λ * exp(ℓ_kn_bigram(kn, b, c; d)), 1e-10))
end

function kn_cost(kn::KNCounts, a::String, b::String, c::String; d::Float64=0.75)::Int32
    Int32(round(clamp(-ℓ_kn_trigram(kn, a, b, c; d) * κ, 0.0, 30_000.0)))
end

@inline _hash3(a::String, b::String, c::String)::UInt64 =
    xor(objectid(a),
        objectid(b) * 0x9e3779b97f4a7c15,
        objectid(c) * 0x6c62272e07bb0142)

const _KN_BITS = 20
const _KN_SIZE = 1 << _KN_BITS
const _KN_MASK = UInt64(_KN_SIZE - 1)
const _C_size  = max(Threads.nthreads(), 64)
const _kn_keys = [zeros(UInt64, _KN_SIZE) for _ in 1:_C_size]
const _kn_vals = [zeros(Int32,  _KN_SIZE) for _ in 1:_C_size]

reset_kn_cache!() = for ks in _kn_keys; fill!(ks, UInt64(0)); end

@inline function kn_cost_cached(kn::KNCounts, a::String, b::String, c::String;
                                  d::Float64=0.75)::Int32
    tid = Threads.threadid()
    ks  = tid <= length(_kn_keys) ? _kn_keys[tid] : return kn_cost(kn, a, b, c; d)
    vs  = _kn_vals[tid]
    key = _hash3(a, b, c)
    key == UInt64(0) && (key = UInt64(1))
    idx = Int(key & _KN_MASK) + 1
    store_idx = 0
    @inbounds for _ in 1:16
        k = ks[idx]
        k == UInt64(0) && (store_idx = idx; break)
        k == key       && return vs[idx]
        idx = (idx % _KN_SIZE) + 1
    end
    result = kn_cost(kn, a, b, c; d)
    if store_idx > 0
        @inbounds ks[store_idx] = key
        @inbounds vs[store_idx] = result
    end
    result
end

# ════════════════════════════════════════════════════════════
# Section 2b  CLASS-BASED KNESER-NEY  (Int16 IDs)
# ════════════════════════════════════════════════════════════
# Bigram: prev.right_id → curr.left_id
# Trigram: pp.right_id → prev.right_id → curr.left_id

function kn_observe_class!(kn::KNCountsClass, a::Int16, b::Int16)
    inner  = get!(Dictionary{Int16,Int}, kn.𝒫_pair, a)
    is_new = !haskey(inner, b)
    set!(inner, b, get(inner, b, 0) + 1)
    set!(kn.𝒫_uni, a, get(kn.𝒫_uni, a, 0) + 1)
    is_new && set!(kn.𝒫_cont, b, get(kn.𝒫_cont, b, 0) + 1)
    kn.𝑁_pairs += 1
end

function kn_observe_tri_class!(kn::KNCountsClass, a::Int16, b::Int16, c::Int16)
    ab  = get!(Dictionary{Int16, Dictionary{Int16,Int}}, kn.𝒫_triple, a)
    bc  = get!(Dictionary{Int16,Int}, ab, b)
    set!(bc, c, get(bc, c, 0) + 1)
    ab2 = get!(Dictionary{Int16,Int}, kn.𝒫_bi4tri, a)
    set!(ab2, b, get(ab2, b, 0) + 1)
    kn.𝑁_trips += 1
end

function merge_kn_class!(dst::KNCountsClass, src::KNCountsClass)
    for (a, inner_src) in pairs(src.𝒫_pair)
        inner_dst = get!(Dictionary{Int16,Int}, dst.𝒫_pair, a)
        for (b, cnt) in pairs(inner_src)
            prev = get(inner_dst, b, 0)
            set!(inner_dst, b, prev + cnt)
            prev == 0 && set!(dst.𝒫_cont, b, get(dst.𝒫_cont, b, 0) + 1)
        end
        set!(dst.𝒫_uni, a, get(dst.𝒫_uni, a, 0) + get(src.𝒫_uni, a, 0))
    end
    dst.𝑁_pairs += src.𝑁_pairs

    for (a, ab_src) in pairs(src.𝒫_triple)
        ab_dst = get!(Dictionary{Int16, Dictionary{Int16,Int}}, dst.𝒫_triple, a)
        for (b, bc_src) in pairs(ab_src)
            bc_dst = get!(Dictionary{Int16,Int}, ab_dst, b)
            for (c, cnt) in pairs(bc_src); set!(bc_dst, c, get(bc_dst, c, 0) + cnt); end
        end
    end
    for (a, ab_src) in pairs(src.𝒫_bi4tri)
        ab_dst = get!(Dictionary{Int16,Int}, dst.𝒫_bi4tri, a)
        for (b, cnt) in pairs(ab_src); set!(ab_dst, b, get(ab_dst, b, 0) + cnt); end
    end
    dst.𝑁_trips += src.𝑁_trips
end

function ℓ_kn_bigram_class(kn::KNCountsClass, a::Int16, b::Int16; d::Float64=0.75)::Float64
    𝑁  = max(kn.𝑁_pairs, 1)
    p̃  = max(get(kn.𝒫_cont, b, 0), 1) / 𝑁
    Cₐ = get(kn.𝒫_uni, a, 0)
    Cₐ == 0 && return log(p̃)
    Cₐᵦ = haskey(kn.𝒫_pair, a) ? get(kn.𝒫_pair[a], b, 0) : 0
    λ   = d * (haskey(kn.𝒫_pair, a) ? length(kn.𝒫_pair[a]) : 0) / Cₐ
    log(max(max(Cₐᵦ - d, 0.0) / Cₐ + λ * p̃, 1e-10))
end

function ℓ_kn_trigram_class(kn::KNCountsClass, a::Int16, b::Int16, c::Int16;
                             d::Float64=0.75)::Float64
    (!haskey(kn.𝒫_triple, a) || !haskey(kn.𝒫_triple[a], b)) &&
        return ℓ_kn_bigram_class(kn, b, c; d)
    C_ab = haskey(kn.𝒫_bi4tri, a) ? get(kn.𝒫_bi4tri[a], b, 0) : 0
    C_ab == 0 && return ℓ_kn_bigram_class(kn, b, c; d)
    C_abc = get(kn.𝒫_triple[a][b], c, 0)
    λ     = d * length(kn.𝒫_triple[a][b]) / C_ab
    log(max(max(C_abc - d, 0.0) / C_ab + λ * exp(ℓ_kn_bigram_class(kn, b, c; d)), 1e-10))
end

function kn_cost_class(kn::KNCountsClass, a::Int16, b::Int16, c::Int16;
                       d::Float64=0.75)::Int32
    Int32(round(clamp(-ℓ_kn_trigram_class(kn, a, b, c; d) * κ, 0.0, 30_000.0)))
end

# ════════════════════════════════════════════════════════════
# Section 2c  CONTEXT VOCABULARY EXTRACTION
# ════════════════════════════════════════════════════════════

"""
    extract_context_vocabulary(long_context, 𝑆) -> Set{String}

Greedy longest-match extraction of known surfaces (length ≥ 2) from a
long document string. Surfaces of length 1 are skipped to avoid boosting
particles like は, が, の.
"""
function extract_context_vocabulary(
    long_context :: String,
    𝑆            :: Dictionary{String,Vector{DictEntry}},
)::Set{String}
    vocab = Set{String}()
    chars = collect(long_context)
    n     = length(chars)
    pos   = 1
    while pos <= n
        best_len = 0
        for len in min(10, n - pos + 1):-1:2
            span = String(chars[pos:pos+len-1])
            if haskey(𝑆, span)
                push!(vocab, span)
                best_len = len
                break
            end
        end
        pos += max(1, best_len)
    end
    return vocab
end

# ════════════════════════════════════════════════════════════
# Section 3  LOAD SUDACHIDICT
# ════════════════════════════════════════════════════════════

kata2hira(s::AbstractString)::String =
    String(map(c -> (cp=Int(c); 0x30A1 <= cp <= 0x30F6 ? Char(cp-0x60) : c), s))

function detect_reading_col(df::DataFrame)::Int
    is_pure_kana(s) = !isempty(s) && all(c -> begin cp=Int(c)
        (0x3041<=cp<=0x3096)||(0x30A1<=cp<=0x30F6)||cp==0x30FC end, s)
    ncols=ncol(df); nrows=min(nrow(df),50)
    for col in 5:min(ncols,25)
        ok=true; seen=false
        for r in 1:nrows
            v=df[r,col]; ismissing(v) && continue
            s=strip(string(v)); (s=="*"||isempty(s)) && continue
            seen=true
            if !is_pure_kana(kata2hira(String(s))); ok=false; break; end
        end
        if seen && ok; println("    reading col: $col"); return col; end
    end
    println("    reading col: fallback 12"); return 12
end

function load_sudachidict(csv_paths::Vector{String})
    𝑅=Dictionary{String,Vector{DictEntry}}()
    𝑆=Dictionary{String,Vector{DictEntry}}()
    𝑚𝑙=Int16(0); 𝑚𝑟=Int16(0); total=0
    for path in csv_paths
        println("  Loading: $path")
        df=CSV.read(path, DataFrame;
            header=false, delim=',',
            types=Dict(2=>Int16,3=>Int16,4=>Int32),
            silencewarnings=true)
        rcol=detect_reading_col(df)
        for row in eachrow(df)
            (ismissing(row[1])||ismissing(row[rcol])) && continue
            raw=strip(string(row[rcol])); isempty(raw) && continue
            surf=intern(string(row[1]))
            l=Int16(row[2]); r=Int16(row[3]); c=Int32(row[4])
            kana=kata2hira(String(raw))
            read=intern(sizeof(kana)<=15 ? InlineStrings.String15(kana)|>String : kana)
            e=DictEntry(surf,read,l,r,c)
            push!(get!(Vector{DictEntry},𝑅,read), e)
            push!(get!(Vector{DictEntry},𝑆,surf), e)
            𝑚𝑙=max(𝑚𝑙,l); 𝑚𝑟=max(𝑚𝑟,r); total+=1
        end
    end
    GC.gc()
    for v in values(𝑅); sizehint!(v,length(v)); end
    for v in values(𝑆); sizehint!(v,length(v)); end
    println("  -> $total entries | left_id 0..$𝑚𝑙 | right_id 0..$𝑚𝑟")
    return 𝑅, 𝑆, 𝑚𝑙, 𝑚𝑟
end

# ════════════════════════════════════════════════════════════
# Section 4  LOAD SUDACHI matrix.def  (pre-trained connection costs)
# ════════════════════════════════════════════════════════════

"""
    load_sudachi_matrix(path) -> Matrix{Int32}

Load Sudachi's pre-trained connection cost matrix from `matrix.def`.
Format: first line "nr nc", then "right_id left_id cost" per line.
Returns a (nr, nc) matrix indexed as M[right_id+1, left_id+1].
"""
function load_sudachi_matrix(path::String)::Matrix{Int32}
    println("  Loading Sudachi matrix.def <- $path")
    nr = 0; nc = 0
    𝐖₀ = Matrix{Int32}(undef, 0, 0)
    open(path) do f
        header = readline(f)
        nr, nc = parse.(Int, split(header))
        𝐖₀ = zeros(Int32, nr, nc)
        for line in eachline(f)
            cols = split(line)
            length(cols) >= 3 || continue
            r = parse(Int, cols[1]) + 1
            c = parse(Int, cols[2]) + 1
            v = parse(Int32, cols[3])
            if 1 <= r <= nr && 1 <= c <= nc
                @inbounds 𝐖₀[r, c] = v
            end
        end
    end
    println("  -> $(nr)×$(nc) matrix loaded")
    return 𝐖₀
end

# ════════════════════════════════════════════════════════════