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
    params["surrogate"] = "CSmoothedTopK"
    params["topk"] = topk(M)
    params["kscale"] = kscale(M)
    
    (; db, queries, params, dist)
end