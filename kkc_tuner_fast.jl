# SPDX-License-Identifier: Apache-2.0
#
# kkc_tuner_fast.jl  ―  KKC Connection Cost Matrix  (Threads.@threads :static + Unicode math)
#
# Training data attribution (see NOTICE file):
#   SudachiDict  (Apache-2.0)  https://github.com/WorksApplications/SudachiDict
#   SNOW T15/T23 (CC-BY 4.0)  http://www.jnlp.org/SNOW/T15
#   Tanaka Corpus (CC-BY)     http://www.edrdg.org/wiki/index.php/Tanaka_Corpus
#
# Launch:  julia --threads=auto kkc_tuner_v3.jl
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
    𝒫_pair   :: Dictionary{String, Dictionary{String,Int}}                          # C(a,b)
    𝒫_uni    :: Dictionary{String, Int}                                              # C(a)
    𝒫_cont   :: Dictionary{String, Int}                                              # continuation |{a:C(a,b)>0}|
    𝑁_pairs  :: Int
    𝒫_triple :: Dictionary{String, Dictionary{String, Dictionary{String,Int}}}      # C(a,b,c)
    𝒫_bi4tri :: Dictionary{String, Dictionary{String,Int}}                          # C(a,b) denominator
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

# ════════════════════════════════════════════════════════════
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
# Section 2  KNESER-NEY  (bigram + trigram)
# ════════════════════════════════════════════════════════════

const κ = 1000.0    # cost quantisation scale  (was KN_SCALE)

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

"""
ℓ_kn(b|a)  --  interpolated Kneser-Ney bigram log-probability.

  P_KN(b|a) = max(C(a,b) - d, 0) / C(a)  +  λ(a) * p̃(b)
  p̃(b)      = |{a : C(a,b)>0}| / N          (continuation probability)
  λ(a)      = d * |{b : C(a,b)>0}| / C(a)   (back-off interpolation mass)
"""
function ℓ_kn_bigram(kn::KNCounts, a::String, b::String; d::Float64=0.75)::Float64
    𝑁  = max(kn.𝑁_pairs, 1)
    p̃  = max(get(kn.𝒫_cont, b, 0), 1) / 𝑁
    Cₐ = get(kn.𝒫_uni, a, 0)
    Cₐ == 0 && return log(p̃)
    Cₐᵦ = haskey(kn.𝒫_pair, a) ? get(kn.𝒫_pair[a], b, 0) : 0
    λ   = d * (haskey(kn.𝒫_pair, a) ? length(kn.𝒫_pair[a]) : 0) / Cₐ
    log(max(max(Cₐᵦ - d, 0.0) / Cₐ + λ * p̃, 1e-10))
end

"""
ℓ_kn(c|a,b)  --  interpolated Kneser-Ney trigram log-probability.
Backs off to bigram when trigram context is sparse.
"""
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

"""Quantised transition cost:  round( -κ * ℓ_kn(c|a,b) )  in [0, 30000]."""
function kn_cost(kn::KNCounts, a::String, b::String, c::String; d::Float64=0.75)::Int32
    Int32(round(clamp(-ℓ_kn_trigram(kn, a, b, c; d) * κ, 0.0, 30_000.0)))
end

# MATH-2: objectid XOR hash -- stable for InternedStrings, zero string copy
@inline _hash3(a::String, b::String, c::String)::UInt64 =
    xor(objectid(a),
        objectid(b) * 0x9e3779b97f4a7c15,
        objectid(c) * 0x6c62272e07bb0142)

# Fixed-size open-addressing hash table per thread (zero allocation during training)
const _KN_BITS = 20                              # 2^20 = 1M entries per thread
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
    key == UInt64(0) && (key = UInt64(1))   # reserve 0 as empty sentinel
    idx = Int(key & _KN_MASK) + 1
    store_idx = 0
    @inbounds for _ in 1:16                 # max 16 probes
        k = ks[idx]
        k == UInt64(0) && (store_idx = idx; break)   # empty slot
        k == key       && return vs[idx]              # cache hit
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
# Section 4  LOAD CORPORA
# ════════════════════════════════════════════════════════════

function load_snow(path::String)::Vector{String}
    df=CSV.read(path, DataFrame; header=true, delim=',', silencewarnings=true)
    cols=names(df)
    g=something(findfirst(c->occursin("原文",c)&&!occursin("英語",c),cols),2)
    y=something(findfirst(c->occursin("やさしい",c),cols),3)
    out=String[]
    for row in eachrow(df), col in (g,y)
        v=row[col]; ismissing(v) && continue
        s=strip(string(v)); isempty(s)||push!(out,s)
    end
    seen=Set{String}()
    unique_out=filter(s->s∉seen&&(push!(seen,s);true),out)
    println("  -> $(length(unique_out)) sentences: $path"); return unique_out
end

function load_tanaka(path::String)::Vector{String}
    out=String[]
    if endswith(path,".csv")
        df=CSV.read(path,DataFrame;header=true,silencewarnings=true)
        jc=something(findfirst(c->!occursin("ID",uppercase(c))&&!occursin("EN",uppercase(c)),names(df)),1)
        for row in eachrow(df)
            v=row[jc]; ismissing(v) && continue
            s=strip(string(v)); isempty(s)||push!(out,s)
        end
    else
        open(path) do f
            for line in eachline(f)
                s=strip(split(line,'\t')[1])
                s=startswith(s,"A: ") ? s[4:end] : s
                isempty(s)||push!(out,s)
            end
        end
    end
    println("  -> $(length(out)) sentences: $path"); return out
end

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
                ents=𝑆[span]; best_e=ents[argmin(e.cost for e in ents)]
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

# ════════════════════════════════════════════════════════════
# Section 6  COLLECT KN COUNTS  (Threads.@threads :static)
# ════════════════════════════════════════════════════════════

function collect_kn_counts!(kn::KNCounts,
                             𝒯::Vector{Tuple{String,Vector{LatticeNode}}})
    BOS="BOS"; EOS="EOS"

    # threadlocal= is a @batch keyword that only works *inside* the loop body.
    # It cannot be read after the loop ends.  We manage storage manually:
    # allocate one KNCounts per thread slot, index by threadid() inside @batch.
    # maxthreadid() >= nthreads() on Julia >= 1.9 (helper threads can have higher ids)
    nslots    = isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : Threads.nthreads() + 4
    local_kns = [KNCounts() for _ in 1:nslots]

    let _T=𝒯, _lkns=local_kns
        Threads.@threads :static for i in eachindex(_T)
            tid = Threads.threadid()
            lkn = _lkns[tid]
            (_,nodes)=_T[i]

            # bigram pass
            prev=BOS
            for nd in nodes; kn_observe!(lkn,prev,nd.entry.surface); prev=nd.entry.surface; end
            kn_observe!(lkn,prev,EOS)

            # trigram pass (ALG-2)
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

    # merge thread-local KNCounts into global kn
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
# Section 7  LATTICE BUILDER
# ════════════════════════════════════════════════════════════

const _BOS = DictEntry("BOS","",Int16(0),Int16(0),Int32(0))
const _EOS = DictEntry("EOS","",Int16(0),Int16(0),Int32(0))

function build_lattice(input::String,
                       𝑅::Dictionary{String,Vector{DictEntry}},
                       max_span::Int=12)::Vector{Vector{LatticeNode}}
    # Pre-compute byte offsets for character positions (avoids collect + slice allocs)
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
# Section 8  VITERBI  (MATH-4 state naming, trigram + beam)
# ════════════════════════════════════════════════════════════

struct ViterbiResult
    path       :: Vector{LatticeNode}
    total_cost :: Int32
end

"""
Viterbi with:
  - trigram KN transition costs  (ALG-2)
  - beam pruning per slot        (ALG-4, beam=0 => exact)
  - @inbounds + @fastmath

State nomenclature  (MATH-4):
  𝑐[pos][j]   cumulative minimum cost to node j at lattice slot pos
  𝜋[pos][j]   back-pointer  (prev_slot, prev_j)
  σ₂[pos][j]  surface of the node two steps back  (prev2 context, MATH-3)
"""
function viterbi(lattice :: Vector{Vector{LatticeNode}},
                 𝐖       :: AbstractMatrix{Int32},
                 kn      :: KNCounts;
                 λ_kn    :: Float32 = 1.0f0,
                 kn_d    :: Float64 = 0.75,
                 beam    :: Int     = 0)::ViterbiResult

    n  = length(lattice)
    𝑐  = [fill(typemax(Int32), length(lattice[i])) for i in 1:n]
    𝜋  = [fill((0,0),          length(lattice[i])) for i in 1:n]
    σ₂ = [fill("BOS",          length(lattice[i])) for i in 1:n]
    𝑐[1] .= Int32(0)

    𝑊ᵣ, 𝑊𝑐 = size(𝐖)

    for pos in 2:n
        isempty(lattice[pos]) && continue
        # Beam cache: nodes are grouped by begin_pos from build_lattice,
        # so consecutive nodes share the same prev_slot.
        _last_ps = -1
        _last_pr = eachindex(lattice[1])
        for (j, node) in enumerate(lattice[pos])
            prev_slot = node.begin_pos
            (prev_slot<1 || isempty(lattice[prev_slot])) && continue

            l    = Int(node.entry.left_id) + 1
            l_ok = 1 <= l <= 𝑊𝑐

            # ALG-4: beam -- keep only top-B predecessors (cached per prev_slot)
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

                # trigram: σ₂ -> prev_node.surface -> node.surface
                p2 = σ₂[prev_slot][k]
                p1 = pn.entry.surface
                𝑤_kn = @fastmath Int32(round(
                    λ_kn * kn_cost_cached(kn, p2, p1, node.entry.surface; d=kn_d)))

                total = prev_c + 𝑤_conn + 𝑤_kn + node.entry.cost
                if total < 𝑐[pos][j]
                    @inbounds 𝑐[pos][j]  = total
                    @inbounds 𝜋[pos][j]  = (prev_slot, k)
                    @inbounds σ₂[pos][j] = p1
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

# ════════════════════════════════════════════════════════════
# Section 9  AVERAGED PERCEPTRON  (ALG-1, mini-batch, Threads.@threads :static)
#
#   𝐖        current weights  (Int32, updated on each mistake)
#   𝐖̄_acc    epoch-level accumulated sum  (Int64, avoids Int32 overflow)
#   𝐖̄        final averaged model  = 𝐖̄_acc / epochs
#   Δ𝐖[t]   per-thread delta  (Int32, zeroed each epoch, OPT-3/RAM-3)
#   η        learning rate  = max(η₀ * ρ^(epoch-1), η_min)  (ALG-3)
# ════════════════════════════════════════════════════════════

function train(;
    sudachidict_csvs :: Vector{String},
    snow_paths       :: Vector{String}  = String[],
    tanaka_paths     :: Vector{String}  = String[],
    epochs           :: Int             = 10,
    λ_kn             :: Float32         = 1.0f0,
    kn_d             :: Float64         = 0.75,
    beam             :: Int             = 8,
    η₀               :: Float32         = 1.0f0,
    ρ                :: Float32         = 0.85f0,
    η_min            :: Int32           = Int32(1),  # floor (must be >= 1)
    output_path      :: String          = "tuned_model.bin",
    dict_bin_path    :: String          = "data/sudachidict/dict.bin",
    mini_batch       :: Int             = 4096,  # sentences per weight-update step
    force_rebuild    :: Bool            = false,
)
    println("\n=== Loading SudachiDict ===")
    𝑅, 𝑆, 𝑚𝑙, 𝑚𝑟 = load_or_build_dict(sudachidict_csvs;
                                         bin_path=dict_bin_path, force_rebuild)
    𝐖     = zeros(Int32, Int(𝑚𝑟)+2, Int(𝑚𝑙)+2)
    𝐖̄_acc = zeros(Int64, size(𝐖)...)
    println("𝐖 shape: $(size(𝐖,1)) x $(size(𝐖,2))   threads: $(Threads.nthreads())")

    println("\n=== Loading Corpora ===")
    all_sents=String[]
    for p in snow_paths;   append!(all_sents, load_snow(p));   end
    for p in tanaka_paths; append!(all_sents, load_tanaka(p)); end
    println("Total raw sentences: $(length(all_sents))")

    println("\n=== Building Training Pairs (Threads.@threads :static) ===")
    raw_pairs=Vector{Union{Nothing,Tuple{String,Vector{LatticeNode}}}}(undef,length(all_sents))
    let _rp=raw_pairs, _sents=all_sents, _S=𝑆
        Threads.@threads :static for i in eachindex(_sents)
            _rp[i]=sentence_to_nodes(_sents[i],_S)
        end
    end
    𝒯=Tuple{String,Vector{LatticeNode}}[p for p in raw_pairs if p!==nothing]
    println("Usable: $(length(𝒯)) / $(length(all_sents))")
    empty!(all_sents); GC.gc()

    println("\n=== KN Counts (Threads.@threads :static) ===")
    kn=KNCounts()
    collect_kn_counts!(kn,𝒯)
    reset_kn_cache!()
    empty!(𝑆); GC.gc()

    println("\n=== Sanity Check ===")
    diagnose(𝒯[1],𝑅,𝐖,kn)

    println("\n=== Averaged Perceptron  ($epochs epochs x mini_batch=$mini_batch) ===")
    # maxthreadid() covers helper threads that may have id > nthreads()
    𝑛t  = isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : Threads.nthreads() + 4
    # Sparse delta tracking: each thread pushes (linear_index => delta) pairs
    Δ_buf = [Pair{Int,Int32}[] for _ in 1:𝑛t]
    for v in Δ_buf; sizehint!(v, 512); end
    merged = Dict{Int, Int32}()
    sizehint!(merged, 16384)
    # Lazy averaging: track last accumulation step per cell
    last_step = zeros(Int32, size(𝐖)...)
    𝑊ᵣ, 𝑊𝑐 = size(𝐖)
    println("  threads=$𝑛t  sentences=$(length(𝒯))")

    # ── Mini-batch perceptron ────────────────────────────────────────────────
    #
    # Amdahl's law: with one update per full epoch the serial region
    # (Δ𝐖 merge + 𝐖 update) runs once / epoch and limits CPU usage to ~50%.
    # Splitting into chunks of `mini_batch` sentences raises the update
    # frequency to N/B times/epoch.  Each serial region touches only the
    # cells dirtied by that chunk — far less than the full 144 MB matrix.
    # Result: CPU utilisation rises toward ~90% on a 10-core M4.
    #
    # Averaging (ALG-1) accumulates 𝐖 after EVERY chunk apply, not just
    # per epoch, so the average is finer-grained and typically more stable.
    #
    # Shuffling indices each epoch prevents the perceptron from memorising
    # presentation order (order-sensitivity of online learning).
    #
    # ASCII aliases in let blocks avoid any macro variable-capture issues.

    # Inner helper: process one chunk, accumulate sparse deltas, apply to 𝐖/𝐖̄_acc
    # Returns number of prediction errors in this chunk.
    function apply_chunk!(chunk_indices::AbstractVector{Int}, lr::Int32)
        for buf in Δ_buf; empty!(buf); end
        nerr = 0
        let _T=𝒯, _R=𝑅, _W=𝐖, _kn=kn,
            _lkn=λ_kn, _knd=kn_d, _bm=beam,
            _dB=Δ_buf, _Wr=𝑊ᵣ, _Wc=𝑊𝑐, _lr=lr,
            _idx=chunk_indices
            nerr_atomic = Threads.Atomic{Int}(0)
            Threads.@threads :static for ii in eachindex(_idx)
                i   = _idx[ii]
                tid = Threads.threadid()
                (input, correct) = _T[i]
                lattice = build_lattice(input, _R)
                result  = viterbi(lattice, _W, _kn; λ_kn=_lkn, kn_d=_knd, beam=_bm)

                # O(n) lockstep comparison, skipping BOS/EOS sentinels
                pred_match = true
                ci = 1; pi = 1
                nc = length(correct); np = length(result.path)
                while ci<=nc && correct[ci].entry.surface∈("BOS","EOS"); ci+=1; end
                while pi<=np && result.path[pi].entry.surface∈("BOS","EOS"); pi+=1; end
                while true
                    c_done = ci>nc; p_done = pi>np
                    (c_done && p_done) && break
                    if c_done ⊻ p_done; pred_match=false; break; end
                    if correct[ci].entry.surface != result.path[pi].entry.surface
                        pred_match=false; break
                    end
                    ci+=1; pi+=1
                    while ci<=nc && correct[ci].entry.surface∈("BOS","EOS"); ci+=1; end
                    while pi<=np && result.path[pi].entry.surface∈("BOS","EOS"); pi+=1; end
                end

                if !pred_match
                    Threads.atomic_add!(nerr_atomic, 1)
                    buf = _dB[tid]
                    for k in 1:length(correct)-1
                        r=Int(correct[k].entry.right_id)  +1
                        l=Int(correct[k+1].entry.left_id) +1
                        if 1<=r<=_Wr && 1<=l<=_Wc
                            push!(buf, (r + (l-1)*_Wr) => -_lr)
                        end
                    end
                    if !isempty(correct)
                        r0=Int(_BOS.right_id)+1; l0=Int(correct[1].entry.left_id)+1
                        if 1<=r0<=_Wr && 1<=l0<=_Wc; push!(buf, (r0 + (l0-1)*_Wr) => -_lr); end
                        rn=Int(correct[end].entry.right_id)+1; ln=Int(_EOS.left_id)+1
                        if 1<=rn<=_Wr && 1<=ln<=_Wc; push!(buf, (rn + (ln-1)*_Wr) => -_lr); end
                    end
                    for k in 1:length(result.path)-1
                        r=Int(result.path[k].entry.right_id)  +1
                        l=Int(result.path[k+1].entry.left_id) +1
                        if 1<=r<=_Wr && 1<=l<=_Wc
                            push!(buf, (r + (l-1)*_Wr) => _lr)
                        end
                    end
                end
            end
        nerr = nerr_atomic[]
        end  # let

        # Sparse merge: aggregate per-thread deltas by cell
        empty!(merged)
        for buf in Δ_buf
            for (idx, v) in buf
                merged[idx] = get(merged, idx, Int32(0)) + v
            end
        end
        # Apply to 𝐖 with lazy averaging (only touches modified cells)
        𝑛_acc += 1
        for (idx, dv) in merged
            gap = 𝑛_acc - 1 - last_step[idx]
            if gap > 0
                @inbounds 𝐖̄_acc[idx] += Int64(𝐖[idx]) * Int64(gap)
            end
            @inbounds 𝐖[idx] += dv
            @inbounds 𝐖̄_acc[idx] += Int64(𝐖[idx])
            @inbounds last_step[idx] = Int32(𝑛_acc)
        end
        return nerr
    end  # apply_chunk!

    𝑛_acc = 0   # total chunk-apply count (averaged perceptron denominator)

    for epoch in 1:epochs
        # ALG-3: η decays per epoch, floor at η_min (Int32 >= 1)
        η = max(round(Int32, η₀ * ρ^(epoch-1)), η_min)

        # Shuffle each epoch — online perceptron is order-sensitive
        idx_perm = randperm(length(𝒯))

        𝑛err = 0; 𝑛chunks = 0
        𝑡_epoch = time()
        total_chunks = cld(length(𝒯), mini_batch)
        for chunk in Iterators.partition(idx_perm, mini_batch)
            𝑡_chunk = time()
            𝑛err   += apply_chunk!(collect(chunk), η)
            𝑛chunks += 1
            𝑡_el = round(time() - 𝑡_chunk, digits=1)
            print("    chunk $𝑛chunks/$total_chunks  ($(𝑡_el)s)  Δcells=$(length(merged))\r")
            flush(stdout)
        end

        pct = round(100(1 - 𝑛err/length(𝒯)), digits=1)
        𝑡_ep = round(time() - 𝑡_epoch, digits=1)
        println("  Epoch $epoch  η=$η  chunks=$𝑛chunks  errors: $𝑛err / $(length(𝒯))  ($pct% correct)  $(𝑡_ep)s")
        𝑛err==0 && (println("  Converged!"); break)
    end

    # Finalize lazy averaging: catch up all cells to total step count
    @inbounds for k in eachindex(𝐖)
        gap = 𝑛_acc - last_step[k]
        if gap > 0
            𝐖̄_acc[k] += Int64(𝐖[k]) * Int64(gap)
        end
    end

    # ALG-1: 𝐖̄ = 𝐖̄_acc / 𝑛_acc  (fine-grained chunk-level average)
    𝐖̄=Matrix{Int32}(undef,size(𝐖)...)
    𝑛_acc_safe = max(𝑛_acc, 1)
    @inbounds for k in eachindex(𝐖̄)
        𝐖̄[k]=Int32(clamp(𝐖̄_acc[k]÷𝑛_acc_safe, typemin(Int32), typemax(Int32)))
    end
    println("  𝐖̄ averaged over $𝑛_acc chunk snapshots")

    println("\n=== Saving -> $output_path ===")
    serialize(output_path,(𝐖̄,kn))
    println("Done!")
    return 𝐖̄, kn, 𝑅
end

# ════════════════════════════════════════════════════════════
# Section 10  DIAGNOSTICS
# ════════════════════════════════════════════════════════════

function diagnose(pair::Tuple{String,Vector{LatticeNode}},
                  𝑅::Dictionary{String,Vector{DictEntry}},
                  𝐖::AbstractMatrix{Int32},
                  kn::KNCounts;
                  λ_kn::Float32=1.0f0, kn_d::Float64=0.75)
    input,correct_path=pair
    println("\n-- DIAGNOSE -----------------------------------------------")
    println("Input hiragana:    ", input)
    println("Correct surfaces:  ", [nd.entry.surface for nd in correct_path])
    lattice=build_lattice(input,𝑅)
    sentinel=("BOS","EOS")
    result=viterbi(lattice,𝐖,kn;λ_kn,kn_d)
    println("Viterbi predicted: ", [nd.entry.surface for nd in result.path  if nd.entry.surface∉sentinel])
    println("Viterbi correct:   ", [nd.entry.surface for nd in correct_path if nd.entry.surface∉sentinel])
    println("Match: ", [nd.entry.surface for nd in result.path if nd.entry.surface∉sentinel] ==
                       [nd.entry.surface for nd in correct_path if nd.entry.surface∉sentinel])
    println("-----------------------------------------------------------\n")
end

# ════════════════════════════════════════════════════════════
# Section 11  INFERENCE
# ════════════════════════════════════════════════════════════

function kkc(input::String,
             𝑅::Dictionary{String,Vector{DictEntry}},
             𝐖::AbstractMatrix{Int32},
             kn::KNCounts;
             λ_kn::Float32=1.0f0, kn_d::Float64=0.75)::String
    result=viterbi(build_lattice(input,𝑅), 𝐖, kn; λ_kn, kn_d, beam=0)
    join(nd.entry.surface for nd in result.path if nd.entry.surface∉("BOS","EOS"))
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
#   julia --threads=auto kkc_tuner_fast.jl

𝐖̄, kn, 𝑅 = train(
    sudachidict_csvs = ["data/sudachidict/small_lex.csv",
                        "data/sudachidict/core_lex.csv"],
    snow_paths       = ["data/snow/T15-2020.1.7.csv",
                        "data/snow/T23-2020.1.7.csv"],
    # tanaka_paths   = ["data/tanaka/examples.utf"],
    epochs  = 10,
    λ_kn    = 1.0f0,
    kn_d    = 0.75,
    beam    = 8,
    η₀      = 1.0f0,
    ρ       = 0.85f0,
    η_min      = Int32(1),
    mini_batch = 4096,
)

println(kkc("とうきょうとっきょきょかきょく", 𝑅, 𝐖̄, kn))
println(kkc("きしゃのきしゃはきしゃできしゃした", 𝑅, 𝐖̄, kn))

export_matrix_def(𝐖̄, "matrix.def")
export_kn_tsv(kn,     "kn_bigram.tsv")