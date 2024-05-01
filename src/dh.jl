export DistantHyperplanes

struct DistantHyperplanes{D<:SemiMetric,DB<:AbstractDatabase} <: AbstractSurrogate
    dist::D         # distance
    H::Vector{Pair{Int,Int}} # hyperplanes
    C::DB           # points
end

distance(::DistantHyperplanes) = BinaryHammingDistance()

function sample_pairs(n::Int, k::Int) 
    visited = Set{Pair{Int,Int}}()
    P = Pair{Int,Int}[]
    sizehint!(P, k)
    
    for _ in 1:k
        i = rand(1:n-1)
        j = rand(i+1:n)
        p = i => j
        if p ∉ visited
            push!(visited, p)
            push!(P, p)
        end
    end

    P
end

function encode1(dist::SemiMetric, C::AbstractDatabase, h::Pair, obj)
    evaluate(dist, obj, C[h[1]]) <= evaluate(dist, obj, C[h[2]])
end

function entropy(binvec)
    n = length(binvec) * 64
    c1 = sum(count_ones, binvec)
    c0 = n - c1
    p0 = c0 / n
    p1 = c1 / n
    -p0 * log2(p0) - p1*log2(p1)
end

#=
function entropy_(B_)
    B = reinterpret(UInt8, B_)
    A = [begin
        n = length(binvec) * 8
        c1 = sum(count_ones, binvec)
        c0 = n - c1
        p0 = c0 / n
        p1 = c1 / n
        -p0 * log2(p0) - p1*log2(p1)
    end for binvec in rand(B, 8)]
    #@info A, mean(A), std(A)
    mean(A) / std(A)
end=#

function fit(::Type{DistantHyperplanes},
        dist::SemiMetric,
        X::AbstractDatabase,
        nbits::Int; # number of output bits
        k::Int = nbits * 1024,     # number of centers to evaluate
        minent::Float64 = 0.99, # minimum accepted entropy per hyperplane
        sample_for_hyperplane_selection::Int = 2^13,  # characterizes hyperplanes with this
        minbatch::Int = 2,
        verbose::Bool=true
    )

    nbits % 64 == 0 || throw(ArgumentError("nbits should be a factor of 64"))
    nbits <= k || throw(ArgumentError("k should be bigger than nbits"))
    sample_for_hyperplane_selection % 64 == 0 || throw(ArgumentError("sample_for_hyperplane_selection should a factor of 64"))
    length(X) > sample_for_hyperplane_selection || throw(ArgumentError("sample_for_hyperplane_selection ($sample_for_hyperplane_selection) should be smaller than |X| ($(length(X)))"))

    @show dist, length(X), nbits, minent, sample_for_hyperplane_selection
    XX = X;
    #=XX = let S = distsample(dist, X)
        sort!(S)
        D = neardup(dist, X, S[2])
        X[D.centers] |> MatrixDatabase
    end
    @assert length(X) > sample_for_hyperplane_selection
    =#
    #=XX = let D = fft(dist, X, 2^11)
        X[D.centers] |> MatrixDatabase
    end=#
   
    #= XX = let D = fft(dist, X, 2^12)
        L = view(sort!(collect(countmap(D.nn)), by=last, rev=true), 1:2^11)
        X[first.(L)] |> MatrixDatabase
    end =#

    P, H = let
        P = sample_pairs(length(XX), k)
        S = shuffle!(collect(1:length(X))); resize!(S, sample_for_hyperplane_selection); sort!(S) 
        #S = fft(dist, X, sample_for_hyperplane_selection).centers
        B = BitArray(undef, sample_for_hyperplane_selection, length(P))
        @batch per=thread minbatch=minbatch for i in 1:sample_for_hyperplane_selection
            obj = X[S[i]]
            for j in eachindex(P)
                b = encode1(dist, XX, P[j], obj)
                B[i, j] = b
            end
        end

        H = reshape(B.chunks, (sample_for_hyperplane_selection ÷ 64, length(P))) |> MatrixDatabase
        E = [entropy(H[i]) >= minent for i in eachindex(H)]
        @show size(H), sum(E)
        P[E], H.matrix[:, E] |> MatrixDatabase
    end
    
   #= 
    E = let
        F = fft(BinaryHammingDistance(), H, 2nbits; verbose)
        P = P[F.centers]
        E = [(entropy_(H[i]), P[i]) for i in eachindex(P)]
        sort!(E, by=first, rev=true)
        resize!(E, nbits)
        E
    end
    DistantHyperplanes(dist, last.(E), XX)
    =#
   
    F = fft(DualHammingDistance(length(H[1])), H, nbits; verbose)
    #F = fft(BinaryHammingDistance(), H, nbits; verbose)

    DistantHyperplanes(dist, P[F.centers], XX)
end

import SimilaritySearch: evaluate
struct DualHammingDistance <: SemiMetric
    cache::Matrix{UInt64}
    DualHammingDistance(dim::Int) = new(Matrix{UInt64}(undef, dim, Threads.nthreads()))
end

function evaluate(H::DualHammingDistance, u, v) 
    dist = BinaryHammingDistance()
    w = view(H.cache, :, Threads.threadid())
    for i in eachindex(v)
        w[i] = ~v[i]
    end
    u_ = evaluate(dist, u, v)
    # v_ = evaluate(dist, u, .~v)
    v_ = evaluate(dist, u, w)
    #n = length(u)
    # u_ * v_ / (n * (u_ + v_))
    min(u_, v_)
end

function predict(m::DistantHyperplanes, obj)
    b = BitArray(undef, length(m.H))
    for i in eachindex(m.H)
        h = m.H[i]
        b[i] = encode1(m.dist, m.C, h, obj)
    end

    b.chunks
end

function predict(m::DistantHyperplanes, arr::AbstractDatabase; minbatch=2)
    n = length(m.H)
    b = BitArray(undef, n, length(arr))
    @batch per=thread minbatch=minbatch for j in 1:length(arr)
        obj = arr[j]
        for i in eachindex(m.H)
            h = m.H[i]
            b[i, j] = encode1(m.dist, m.C, h, obj)
        end
    end

    MatrixDatabase(reshape(b.chunks, (n ÷ 64, length(arr))))
end
