using Random

abstract type AbstractSurrogate end

struct BinaryHammingFixedPairs <: AbstractSurrogate
    kscale::Int
end

kscale(m::BinaryHammingFixedPairs) = m.kscale

encode(::BinaryHammingFixedPairs, v, i::Integer)::Bool = v[i] < v[end-i+1]
encode(B::BinaryHammingFixedPairs, v) = (encode.((B,), (v,), 1:length(v) ÷ 2)).chunks

function encode(B::BinaryHammingFixedPairs, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = VectorDatabase([encode(B, db_[i]) for i in eachindex(db_)])
	queries = VectorDatabase([encode(B, queries_[i]) for i in eachindex(queries_)])
    params["surrogate"] = "BHFP"
    params["kscale"] = B.kscale
    (; db, queries, params, dist)
end


function random_sorted_pair(dim)
    a = rand(one(Int32):dim)
    b = rand(one(Int32):dim)
    a < b ? (a, b) : (b, a)
end

#############

struct BinaryHammingSurrogate <: AbstractSurrogate
    kscale::Int
    pairs::Vector{Tuple{Int32,Int32}}
    
    function BinaryHammingSurrogate(kscale::Integer, npairs::Integer, dim::Integer)
        dim = convert(Int32, dim)
        P = Set([random_sorted_pair(dim)])
        for i in 2:npairs
            push!(P, random_sorted_pair(dim))
        end
        
        S = collect(P); sort!(S)
        new(kscale, S)
    end
end

kscale(m::BinaryHammingSurrogate) = m.kscale
encode(B::BinaryHammingSurrogate, v, p::Tuple)::Bool = v[p[1]] < v[p[2]]
encode(B::BinaryHammingSurrogate, v) = (encode.((B,), (v,), B.pairs)).chunks

function encode(B::BinaryHammingSurrogate, X::AbstractDatabase)
    V = Vector{Vector{UInt64}}(undef, length(X))
    Threads.@threads for i in eachindex(X)
        V[i] = encode(B, X[i])
    end

    VectorDatabase(V)
end

function encode(B::BinaryHammingSurrogate, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = encode(B, db_)
	queries = encode(B, queries_)
    params["surrogate"] = "BHS"
    params["kscale"] = B.kscale
    params["npairs"] = length(B.pairs)
    (; db, queries, params, dist)
end

#=

struct MaxPoolSurrogate <: AbstractSurrogate
    kscale::Int
    pool::Matrix{Int32}
    
    function MaxPoolSurrogate(samplesize::Integer, npools::Integer, dim::Integer, kscale::Integer)
        pool = Matrix{Float32}(undef, samplesize, npools)
        perm = Vector{Int32}(1:dim)
      
        for i in 1:npools
            randperm!(perm)
            pool[:, i] .= view(perm, 1:samplesize)
        end
        
        new(kscale, pool)
    end
end

samplesize(M::MaxPoolSurrogate) = size(M.pool, 1)
npools(M::MaxPoolSurrogate) = size(M.pool, 2)
kscale(M::MaxPoolSurrogate) = M.kscale

function encode(M::MaxPoolSurrogate, vout, v)
    for i in eachindex(vout)
        vout[i] = maximum(j -> v[j], view(M.pool, :, i))
    end
    
    vout
end

function encode(M::MaxPoolSurrogate, db_::AbstractDatabase)
    D = Matrix{Float32}(undef, npools(M), length(db_))
    
    for i in eachindex(db_)
        encode(M, view(D, :, i), db_[i])
    end

    MatrixDatabase(D)
end

function encode(M::MaxPoolSurrogate, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = L2Distance()
    db = encode(M, db_)
    queries = encode(M, queries_)
    params["surrogate"] = "maxpool"
    params["kscale"] = M.kscale
    (; db, queries, params, dist)
end

=#

###############

struct MaxHashSurrogate <: AbstractSurrogate
    kscale::Int
    pool::Matrix{Int32}
    
    function MaxHashSurrogate(samplesize::Integer, npools::Integer, dim::Integer, kscale::Integer)
        samplesize < 256 || throw(ArgumentError("samplesize < 256: $samplesize"))
        pool = Matrix{Int32}(undef, samplesize, npools)
        perm = Vector{Int32}(1:dim)
      
        for i in 1:npools
            randperm!(perm)
            pool[:, i] .= view(perm, 1:samplesize)
        end
        
        new(kscale, pool)
    end
end

samplesize(M::MaxHashSurrogate) = size(M.pool, 1)
npools(M::MaxHashSurrogate) = size(M.pool, 2)
kscale(M::MaxHashSurrogate) = M.kscale

function encode(M::MaxHashSurrogate, vout, v)
    for i in eachindex(vout)
        #vout[i] = maximum(j -> v[j], view(M.pool, :, i))
        vout[i] = findmax(j -> v[j], view(M.pool, :, i)) |> last
    end
    
    vout
end

function encode(M::MaxHashSurrogate, db_::AbstractDatabase)
    D = Matrix{UInt8}(undef, npools(M), length(db_))
    
    Threads.@threads for i in eachindex(db_)
        encode(M, view(D, :, i), db_[i])
    end

    MatrixDatabase(D)
end

function encode(M::MaxHashSurrogate, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = StringHammingDistance()
    db = encode(M, db_)
    queries = encode(M, queries_)
    params["surrogate"] = "MaxHash"
    params["samplesize"] = samplesize(M)
    params["npools"] = npools(M)
    params["kscale"] = kscale(M)
    
    (; db, queries, params, dist)
end

###############

struct TopKSurrogate <: AbstractSurrogate
    topk::Int
    dim::Int
    kscale::Int
    
    TopKSurrogate(topk, dim, kscale) = new(topk, ceil(Int, dim / 64) * 64, kscale)
end

kscale(T::TopKSurrogate) = T.kscale
topk(T::TopKSurrogate) = T.topk
dim(T::TopKSurrogate) = T.dim

function encode(M::TopKSurrogate, out, v, res::KnnResult, tmp::BitArray)
    reuse!(res, topk(M))
    fill!(tmp, 0)
    
    for i in eachindex(v)
        push!(res, i, -v[i])
    end
    
    for i in idview(res)
        tmp[i] = 1
    end
    
    copy!(out, tmp.chunks)
end

function encode(M::TopKSurrogate, db_::AbstractDatabase)
    D = Matrix{UInt64}(undef, dim(M) ÷ 64, length(db_))
    R = [KnnResult(M.topk) for i in 1:Threads.nthreads()]
    B = [BitArray(undef, dim(M)) for i in 1:Threads.nthreads()]
    Threads.@threads for i in eachindex(db_)
        tid = Threads.threadid()
        encode(M, view(D, :, i), db_[i], R[tid], B[tid])
    end

    MatrixDatabase(D)
end

function encode(M::TopKSurrogate, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = encode(M, db_)
    queries = encode(M, queries_)
    params["surrogate"] = "TopK"
    params["topk"] = topk(M)
    params["kscale"] = kscale(M)
    
    (; db, queries, params, dist)
end

#==== Smoothed Topk =====#

struct LogisticFunction
    scale::Float64
end

@inline logistic(lfun::LogisticFunction, x) = 1 / (1 + exp(-lfun.scale * x))

function smooth_topk(lfun::LogisticFunction, X::AbstractVector, t)
    s = 0.0
    @inbounds @simd for x in X
        s += logistic(lfun, x + t)
    end

    s
end

function binsearch_optim_topk(lfun::LogisticFunction, X, k::Float64; tol=1e-1, maxiters=64)
    low, high = -1e6, 1e6 # extrema(X)
    
    iter = 0
    t = 0.0
    
	while low < high
        t = 0.5 * (low + high)
        h = smooth_topk(lfun, X, t)
        # @show k, h, iter, t, low, high
        abs(k - h) <= tol && break
        if k < h
            high = t
        else
            low = t
        end
        
        iter += 1
        iter == maxiters && break
	end
	
    t
end

struct SmoothedTopK <: AbstractSurrogate
    topk::Int
    dim::Int
    kscale::Int
    lfun::LogisticFunction
    
    SmoothedTopK(topk, dim, kscale) = new(topk, dim, kscale, LogisticFunction(8))
end

kscale(T::SmoothedTopK) = T.kscale
topk(T::SmoothedTopK) = T.topk
dim(T::SmoothedTopK) = T.dim

function encode(M::SmoothedTopK, out, X)
    lfun = M.lfun
    t = binsearch_optim_topk(lfun, X, 5.0)
    #@show t smooth_topk(lfun, X, t)
    @inbounds for i in eachindex(X)
        out[i] = logistic(lfun, X[i] + t)
    end
    
    out
end

function encode(M::SmoothedTopK, db_::AbstractDatabase)
    D = Matrix{Float32}(undef, dim(M), length(db_))
    Threads.@threads for i in eachindex(db_)
        tid = Threads.threadid()
        encode(M, view(D, :, i), db_[i])
    end

    MatrixDatabase(D)
end

function encode(M::SmoothedTopK, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = L2Distance()
    db = encode(M, db_)
    queries = encode(M, queries_)
    params["surrogate"] = "SmoothedTopK"
    params["topk"] = topk(M)
    params["kscale"] = kscale(M)
    
    (; db, queries, params, dist)
end