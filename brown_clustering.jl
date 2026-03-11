# SPDX-License-Identifier: Apache-2.0
#
# brown_clustering.jl  ―  Brown hierarchical word clustering for KKC
#
# Assigns each word surface a cluster ID (UInt16) based on bigram
# mutual information from the training corpus.  At runtime the Viterbi
# decoder uses a learned 𝐖_sem[cluster_prev, cluster_curr] matrix
# for O(1) semantic transition scoring.
#
# Reference: Brown et al. 1992, "Class-Based n-gram Models of Natural Language"

# ════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════

struct BrownClusters
    word2cluster :: Dict{String, UInt16}
    n_clusters   :: Int
end

# ════════════════════════════════════════════════════════════
# Core algorithm: greedy agglomerative Brown clustering
# ════════════════════════════════════════════════════════════

"""
    brown_cluster(training_pairs, n_clusters; min_freq) -> BrownClusters

Build Brown clusters from training data.

`training_pairs` — Vector{Tuple{String, Vector{LatticeNode}}} from the
training pipeline.  We extract surface bigrams from gold paths.

Algorithm (simplified greedy):
1. Count word unigrams and bigrams from gold paths
2. Keep top-V words by frequency (V = active vocabulary)
3. Initialize each word in its own cluster
4. Greedily merge the pair of clusters that causes the smallest
   decrease in corpus mutual information, until n_clusters remain

For efficiency with large vocabularies, we use the "incremental" variant:
start with n_clusters initial clusters for the most frequent words,
then add remaining words one at a time, merging back to n_clusters.
"""
function brown_cluster(
    𝒯          :: Vector{Tuple{String, Vector{LatticeNode}}},
    n_clusters :: Int = 256;
    min_freq   :: Int = 3,
    max_vocab  :: Int = 8000,
)::BrownClusters

    println("  Brown clustering: counting bigrams...")

    # Step 1: Count unigrams and bigrams from gold paths
    uni_counts = Dict{String, Int}()
    bi_counts  = Dict{Tuple{String,String}, Int}()
    BOS = "BOS"; EOS = "EOS"

    for (_, nodes) in 𝒯
        prev = BOS
        for nd in nodes
            s = nd.entry.surface
            s ∈ (BOS, EOS) && continue
            uni_counts[s] = get(uni_counts, s, 0) + 1
            bi_counts[(prev, s)] = get(bi_counts, (prev, s), 0) + 1
            prev = s
        end
        bi_counts[(prev, EOS)] = get(bi_counts, (prev, EOS), 0) + 1
    end
    uni_counts[BOS] = length(𝒯)
    uni_counts[EOS] = length(𝒯)

    # Step 2: Filter vocabulary by frequency, keep top max_vocab
    vocab = [w for (w, c) in uni_counts if c >= min_freq && w ∉ (BOS, EOS)]
    sort!(vocab, by = w -> -uni_counts[w])
    if length(vocab) > max_vocab
        resize!(vocab, max_vocab)
    end
    vocab_set = Set{String}(vocab)
    push!(vocab_set, BOS, EOS)

    println("  Brown clustering: $(length(vocab)) words (min_freq=$min_freq)")

    # Step 3: Build filtered bigram counts (only vocab words)
    N_total = sum(values(bi_counts))
    filt_bi = Dict{Tuple{String,String}, Int}()
    filt_uni = Dict{String, Int}()
    for ((a, b), c) in bi_counts
        a ∈ vocab_set && b ∈ vocab_set || continue
        filt_bi[(a, b)] = c
    end
    for (w, c) in uni_counts
        w ∈ vocab_set || continue
        filt_uni[w] = c
    end
    N = Float64(sum(values(filt_bi)))

    # Step 4: Incremental clustering
    # Start with n_clusters slots for most frequent words,
    # assign remaining words to the closest existing cluster.
    #
    # For each remaining word w, find the cluster c that maximizes
    # the pointwise mutual information between w and c's members.

    # Initial assignment: top n_clusters words each get their own cluster
    n_init = min(n_clusters, length(vocab))
    word2cid = Dict{String, UInt16}()

    # Assign initial clusters
    for i in 1:n_init
        word2cid[vocab[i]] = UInt16(i)
    end

    # For remaining words: assign to cluster with highest bigram affinity
    if length(vocab) > n_init
        # Precompute cluster → word sets and cluster bigram sums
        cluster_words = Dict{UInt16, Vector{String}}()
        for i in 1:n_init
            cluster_words[UInt16(i)] = [vocab[i]]
        end

        # Cluster unigram mass: sum of unigram counts for all words in cluster
        cluster_uni = Dict{UInt16, Float64}()
        for (cid, words) in cluster_words
            cluster_uni[cid] = sum(get(filt_uni, w, 0) for w in words)
        end

        println("  Brown clustering: assigning $(length(vocab) - n_init) remaining words...")

        for idx in (n_init+1):length(vocab)
            w = vocab[idx]
            w_count = Float64(get(filt_uni, w, 1))

            # Score each existing cluster by bigram affinity
            best_cid = UInt16(1)
            best_score = -Inf

            for cid in UInt16(1):UInt16(n_init)
                score = 0.0
                for cw in cluster_words[cid]
                    # w → cw bigram
                    c_wb = Float64(get(filt_bi, (w, cw), 0))
                    if c_wb > 0
                        pmi = log(c_wb * N / (w_count * Float64(get(filt_uni, cw, 1))))
                        score += c_wb * pmi
                    end
                    # cw → w bigram
                    c_bw = Float64(get(filt_bi, (cw, w), 0))
                    if c_bw > 0
                        pmi = log(c_bw * N / (Float64(get(filt_uni, cw, 1)) * w_count))
                        score += c_bw * pmi
                    end
                end

                if score > best_score
                    best_score = score
                    best_cid = cid
                end
            end

            word2cid[w] = best_cid
            push!(cluster_words[best_cid], w)
            cluster_uni[best_cid] += w_count
        end
    end

    # BOS and EOS get cluster 0 (reserved)
    word2cid[BOS] = UInt16(0)
    word2cid[EOS] = UInt16(0)

    # Print cluster statistics
    cluster_sizes = Dict{UInt16, Int}()
    for (_, cid) in word2cid
        cluster_sizes[cid] = get(cluster_sizes, cid, 0) + 1
    end
    non_empty = count(v -> v > 0, values(cluster_sizes))
    println("  Brown clustering: $non_empty non-empty clusters, " *
            "$(length(word2cid)) words assigned")

    return BrownClusters(word2cid, n_clusters)
end

"""
    get_cluster(bc, surface) -> UInt16

Look up the cluster ID for a word surface.
Returns 0 (BOS/EOS cluster) for unknown words.
"""
@inline function get_cluster(bc::BrownClusters, surface::AbstractString)::UInt16
    get(bc.word2cluster, String(surface), UInt16(0))
end
