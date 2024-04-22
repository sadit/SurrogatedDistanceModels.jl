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

function fit(::Type{DistantHyperplanes},
        dist::SemiMetric,
        X::AbstractDatabase,
        nbits::Int; # number of output bits
        k::Int = nbits * 128,     # number of centers to evaluate
        minent::Float64 = 0.5, # minimum accepted entropy per hyperplane
        sample_for_hyperplane_selection::Int = 2^12,  # characterizes hyperplanes with this
        minbatch::Int = 2,
        verbose::Bool=true
    )

    nbits % 64 == 0 || throw(ArgumentError("nbits should be a factor of 64"))
    nbits <= k || throw(ArgumentError("k should be bigger than nbits"))
    sample_for_hyperplane_selection % 64 == 0 || throw(ArgumentError("sample_for_hyperplane_selection should a factor of 64"))
    length(X) > sample_for_hyperplane_selection || throw(ArgumentError("sample_for_hyperplane_selection ($sample_for_hyperplane_selection) should be smaller than |X| ($(length(X)))"))

    @show dist, length(X), nbits, minent, sample_for_hyperplane_selection

    P, H = let
        S = shuffle!(collect(1:length(X))); resize!(S, sample_for_hyperplane_selection); sort!(S) 
        P = sample_pairs(length(X), k)
        m = length(P)
        B = BitArray(undef, sample_for_hyperplane_selection, m)
        @batch per=thread minbatch=minbatch for i in 1:sample_for_hyperplane_selection
            obj = X[S[i]]
            for j in eachindex(P)
                B[i, j] = encode1(dist, X, P[j], obj)
            end
        end

        H = reshape(B.chunks, (sample_for_hyperplane_selection ÷ 64, length(P))) |> MatrixDatabase
        E = [entropy(H[i]) >= minent for i in eachindex(H)]
        @show size(H), sum(E)
        P[E], H.matrix[:, E] |> MatrixDatabase
    end

    #=
    E = let
        F = fft(BinaryHammingDistance(), H, 4nbits; verbose)
        P = P[F.centers]
        E = [(entropy(H[i]), P[i]) for i in eachindex(P)]
        sort!(E, by=first, rev=true)
        resize!(E, nbits)
        E
    end
    =#
    F = fft(BinaryHammingDistance(), H, nbits; verbose)
    DistantHyperplanes(dist, P[F.centers], X)
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
