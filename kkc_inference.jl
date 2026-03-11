# kkc_inference.jl
# ════════════════════════════════════════════════════════════
# Usage: julia kkc_inference.jl "input_hiragana"
# ════════════════════════════════════════════════════════════

using Serialization
using Mmap
using Dictionaries
using StringViews
using InlineStrings

# --- 1. DATA STRUCTURES (Optimized for RAM) ---

struct DictEntry
    surface  :: String127  
    reading  :: String127
    left_id  :: Int16
    right_id :: Int16
    cost     :: Int32
end

struct LatticeNode
    entry     :: DictEntry
    begin_pos :: Int
    end_pos   :: Int
end

mutable struct KNCounts
    𝒫_pair   :: Dictionary{String, Dictionary{String,Int}}
    𝒫_uni    :: Dictionary{String, Int}
    𝒫_cont   :: Dictionary{String, Int}
    𝑁_pairs  :: Int
    𝒫_triple :: Dictionary{String, Dictionary{String, Dictionary{String,Int}}}
    𝒫_bi4tri :: Dictionary{String, Dictionary{String,Int}}
    𝑁_trips  :: Int
end

# --- 2. CORE CONSTANTS ---

const DICT_MAGIC   = UInt32(0x4B4B4344)
const DICT_VERSION = UInt32(1)
const _BOS = DictEntry(String127("BOS"), String127(""), Int16(0), Int16(0), Int32(0))
const _EOS = DictEntry(String127("EOS"), String127(""), Int16(0), Int16(0), Int32(0))

# --- 3. DICTIONARY LOADER (Binary Mmap) ---

function load_dict_bin(path::String)
    io   = open(path, "r")
    data = Mmap.mmap(io)
    pos  = Ref(1)

    rd_u32() = (v=reinterpret(UInt32, data[pos[]:pos[]+3])[1]; pos[]+=4; v)
    rd_i16() = (v=reinterpret(Int16,  data[pos[]:pos[]+1])[1]; pos[]+=2; v)
    rd_i32() = (v=reinterpret(Int32,  data[pos[]:pos[]+3])[1]; pos[]+=4; v)
    rd_u64() = (v=reinterpret(UInt64, data[pos[]:pos[]+7])[1]; pos[]+=8; v)

    rd_u32() == DICT_MAGIC   || error("Bad magic in $path")
    rd_u32() == DICT_VERSION || error("Dict version mismatch")

    n_str = Int(rd_u64()); hb = Int(rd_u64()); n_ent = Int(rd_u64())
    n_R = Int(rd_u64()); n_S = Int(rd_u64()); rd_u64()

    hs = pos[]; pos[] = hs + hb

    str_offs = Vector{UInt32}(undef, n_str)
    str_lens = Vector{UInt32}(undef, n_str)
    for i in 1:n_str
        str_offs[i] = rd_u32()
        str_lens[i] = rd_u32()
    end

    function heap_str(id)
        len = Int(str_lens[id+1])
        # The safeguard: Return an empty String127 sentinel if it's too huge
        if len > 127
            return String127("")
        end
        start_idx = Int(str_offs[id+1]) + hs
        return String127(StringView(@view data[start_idx : start_idx+len-1]))
    end

    entries = Vector{DictEntry}(undef, n_ent)
    for i in 1:n_ent
        s=rd_u32(); r=rd_u32(); l=rd_i16(); ri=rd_i16(); c=rd_i32(); rd_u32()
        entries[i] = DictEntry(heap_str(s), heap_str(r), l, ri, c)
    end

    R = Dictionary{String,Vector{DictEntry}}()
    for _ in 1:n_R
        kid = rd_u32()
        m = Int(rd_u32())
        k_str = heap_str(kid)

        # If the key itself was too long and became "", skip adding it to the dictionary
        # but WE MUST still read the `m` entry IDs to keep the file pointer aligned!
        if k_str == ""
            for _ in 1:m; rd_u32(); end
            continue
        end

        key = String(k_str)
        new_entries = DictEntry[]
        
        for _ in 1:m
            eid = Int(rd_u32()) + 1
            ent = entries[eid]
            # Only add the entry if its internal strings weren't truncated
            if ent.surface != "" && ent.reading != ""
                push!(new_entries, ent)
            end
        end
        
        if !isempty(new_entries)
            if haskey(R, key)
                append!(R[key], new_entries)
            else
                set!(R, key, new_entries)
            end
        end
    end
    close(io)
    return R
end

# --- 4. LANGUAGE MODEL SCORING ---

const κ = 1000.0

function ℓ_kn_bigram(kn::KNCounts, a::String, b::String; d::Float64=0.75)::Float64
    𝑁  = max(kn.𝑁_pairs, 1)
    p̃  = max(get(kn.𝒫_cont, b, 0), 1) / 𝑁
    Cₐ = get(kn.𝒫_uni, a, 0)
    Cₐ == 0 && return log(p̃)
    Cₐᵦ = haskey(kn.𝒫_pair, a) ? get(kn.𝒫_pair[a], b, 0) : 0
    λ   = d * (haskey(kn.𝒫_pair, a) ? length(kn.𝒫_pair[a]) : 0) / Cₐ
    log(max(max(Cₐᵦ - d, 0.0) / Cₐ + λ * p̃, 1e-10))
end

function ℓ_kn_trigram(kn::KNCounts, a::String, b::String, c::String; d::Float64=0.75)::Float64
    (!haskey(kn.𝒫_triple, a) || !haskey(kn.𝒫_triple[a], b)) && return ℓ_kn_bigram(kn, b, c; d)
    C_ab = haskey(kn.𝒫_bi4tri, a) ? get(kn.𝒫_bi4tri[a], b, 0) : 0
    C_ab == 0 && return ℓ_kn_bigram(kn, b, c; d)
    C_abc = get(kn.𝒫_triple[a][b], c, 0)
    λ     = d * length(kn.𝒫_triple[a][b]) / C_ab
    log(max(max(C_abc - d, 0.0) / C_ab + λ * exp(ℓ_kn_bigram(kn, b, c; d)), 1e-10))
end

function get_kn_cost(kn::KNCounts, a::AbstractString, b::AbstractString, c::AbstractString)::Int32
    # Cast explicitly to String to satisfy Dictionaries.jl strict typing.
    # Because these are tiny, short-lived strings, it won't impact our low-RAM goal.
    a_str, b_str, c_str = String(a), String(b), String(c)
    Int32(round(clamp(-ℓ_kn_trigram(kn, a_str, b_str, c_str) * κ, 0.0, 30_000.0)))
end

# --- 5. SEARCH LOGIC (Viterbi) ---

function build_lattice(input::String, R::Dictionary{String,Vector{DictEntry}})
    ci = collect(eachindex(input)); n = length(ci)
    lattice = [LatticeNode[] for _ in 1:n+2]
    push!(lattice[1], LatticeNode(_BOS, 0, 1))
    for start in 1:n, len in 1:min(12, n-start+1)
        b_start = ci[start]
        b_end = start+len <= n ? prevind(input, ci[start+len]) : lastindex(input)
        span = SubString(input, b_start, b_end)
        if haskey(R, span)
            for e in R[span]; push!(lattice[start+len], LatticeNode(e, start, start+len)); end
        end
    end
    push!(lattice[n+2], LatticeNode(_EOS, n+1, n+2))
    return lattice
end

function viterbi(lattice, W, kn)
    n = length(lattice)
    best_costs = [fill(typemax(Int32), length(lattice[i])) for i in 1:n]
    back_ptr   = [fill((0, 0), length(lattice[i])) for i in 1:n]

    best_costs[1][1] = 0

    for pos in 2:n
        for (j, node) in enumerate(lattice[pos])
            prev_slot = node.begin_pos
            l_id = Int(node.entry.left_id) + 1
            
            # Skip unreachable nodes
            (prev_slot < 1 || isempty(lattice[prev_slot])) && continue

            for (k, prev_node) in enumerate(lattice[prev_slot])
                prev_c = best_costs[prev_slot][k]
                prev_c == typemax(Int32) && continue
                
                # Dynamic backtrace mapping
                pp_slot, pp_k = back_ptr[prev_slot][k]
                ctx_surface = pp_slot > 0 ? lattice[pp_slot][pp_k].entry.surface : _BOS.surface
                
                r_id = Int(prev_node.entry.right_id) + 1
                w_conn = (1 <= r_id <= size(W, 1) && 1 <= l_id <= size(W, 2)) ? W[r_id, l_id] : Int32(5000)
                
                w_kn   = get_kn_cost(kn, ctx_surface, prev_node.entry.surface, node.entry.surface)
                
                total = prev_c + w_conn + w_kn + node.entry.cost
                if total < best_costs[pos][j]
                    best_costs[pos][j] = total
                    back_ptr[pos][j]   = (prev_slot, k)
                end
            end
        end
    end

    # Backtrack
    path = String[]
    curr_pos, curr_idx = n, 1
    
    # Catch broken lattices
    if back_ptr[n][1] == (0, 0)
        return "[ERROR: Path unreachable. Lattice disconnected.]"
    end

    while curr_pos > 1
        node = lattice[curr_pos][curr_idx]
        if node.entry.surface ∉ ("BOS", "EOS")
            pushfirst!(path, String(node.entry.surface))
        end
        curr_pos, curr_idx = back_ptr[curr_pos][curr_idx]
    end
    return join(path)
end

# --- 6. EXECUTION ---

function main()
    println("--- Initializing KKC ---")
    W, kn = deserialize("tuned_model.bin")
    GC.gc() 

    R = load_dict_bin("data/sudachidict/dict.bin")
    
    input = length(ARGS) > 0 ? ARGS[1] : "きょうはいいてんきですね"
    println("Input:  ", input)
    
    lattice = build_lattice(input, R)
    result  = viterbi(lattice, W, kn)
    
    println("Result: ", result)
end

main()