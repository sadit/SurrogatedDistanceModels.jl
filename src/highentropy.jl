export HighEntropyHyperplanes

struct HighEntropyHyperplanes{D<:SemiMetric,DB<:AbstractDatabase} <: AbstractSurrogate
    dist::D         # distance
    H::Vector{Pair{Int,Int}} # hyperplanes
    C::DB           # points
end

distance(::HighEntropyHyperplanes) = BinaryHammingDistance()

function encode1(dist::SemiMetric, C::AbstractDatabase, h::Pair, obj)
    evaluate(dist, obj, C[h[1]]) < evaluate(dist, obj, C[h[2]])
end

function entropy(c0, c1)
    n = c0 + c1
    c0/n * log2(n/c0) + c1/n * log2(n/c1)
end

function fit(::Type{HighEntropyHyperplanes},
        dist::SemiMetric,
        X::MatrixDatabase,
        nbits::Int = 512; # number of output bits
        k::Int = 128,     # number of centers to evaluate
        k2::Int = 80,     # number of centers to select (smaller than k)
        sample_for_fft::Int = 2^13,                  # sample size to compute fft
        sample_for_hyperplane_selection::Int = 2^13,  # size of the second sample (largest than first, characterizes hyperplanes with this)
        minbatch::Int = 2,
        verbose::Bool=true
    )

    nbits % 64 == 0 || throw(ArgumentError("nbits should be a factor of 64"))
    k > k2 || throw(ArgumentError("k > k2"))
    nbits <= k2^2/2 - k2/2 || throw(ArgumentError("k2^2/2 - k2/2 should be bigger than nbits"))
    sample_for_hyperplane_selection % 64 == 0 || throw(ArgumentError("sample_for_hyperplane_selection should a factor of 64"))

    S = shuffle!(collect(1:length(X)))
    sample = let
        s = SubDatabase(X, S[1:sample_for_fft])
        H = fft(dist, s, k; verbose)
        s.map[H.nn]
    end
 
    points = let ## most populated centers
        M = sort!(countmap(sample) |> collect, by=last, rev=true)
        M_ = [M[i][1] for i in 1:k2]
        SubDatabase(X, M_) |> MatrixDatabase
    end

    P = let 
        n = k2
        P = Pair{Int,Int}[]
        sizehint!(P, round(Int, n^2/2 - n/2))
        for i in 1:n
            for j in i+1:n
                push!(P, i => j)
            end
        end
        P
    end

    shuffle!(S)
    resize!(S, sample_for_hyperplane_selection)
    sort!(S) 

    @info "========================"
    ent = let 
        votes, B = let
            m = length(P)
            votes = zeros(Int32, 2, m)
            B = BitArray(undef, sample_for_hyperplane_selection, m)
            @batch per=thread minbatch=minbatch for i in 1:sample_for_hyperplane_selection
                #for i in 1:sample_for_hyperplane_selection
                obj = X[S[i]]
                for j in eachindex(P)
                    c = encode1(dist, points, P[j], obj)
                    votes[c+1, j] += 1
                    B[i, j] = c
                end
            end

            votes, B
        end

        dbH = reshape(B.chunks, (sample_for_hyperplane_selection ÷ 64, length(P))) |> MatrixDatabase
        distH = BinaryHammingDistance()
        F = fft(distH, dbH, 2 * nbits; verbose)
        
        ent = [let
                  v0, v1 = votes[:, c]
                  entropy(v0, v1) => c
                  end for c in F.centers] 
        sort!(ent, by=first, rev=true)
        resize!(ent, nbits)
        ent
    end

    HighEntropyHyperplanes(dist, P[last.(ent)], points)
end

function predict(m::HighEntropyHyperplanes, obj)
    b = BitArray(undef, length(m.H))
    for i in eachindex(m.H)
        h = m.H[i]
        b[i] = encode1(m.dist, m.C, h, obj)
    end

    b.chunks
end

function predict(m::HighEntropyHyperplanes, arr::AbstractDatabase; minbatch=2)
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
