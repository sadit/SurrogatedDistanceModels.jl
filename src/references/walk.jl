export BinWalk


struct BinWalk{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractSurrogate
    dist::DistType
    refs::DbType
    pool::Matrix{Int32}    
end

distance(::BinWalk) = BinaryHammingDistance()

function fit(::Type{BinWalk}, dist::SemiMetric, refs::AbstractDatabase; permsize::Integer=64)
    2 <= permsize <= 64 || throw(ArgumentError("invalid permsize $permsize"))
    n = length(refs)
    @assert n % permsize == 0
    nperms = n ÷ permsize
    pool = Matrix{Int32}(undef, permsize, nperms)
    P = vec(pool)
    for i in 1:n
        P[i] = i
    end

    shuffle!(P)
    BinWalk(dist, refs, pool)
end

@inline permsize(M::BinWalk) = size(M.pool, 1)
@inline nperms(M::BinWalk) = size(M.pool, 2)

function encode_object!(M::BinWalk, vout, v, cache::PermsCacheEncoder)
    for i in 1:nperms(M)
        col = view(M.pool, :, i)
        for (j, k) in enumerate(col)
            cache.vec[j] = evaluate(M.dist, M.refs[k], v)
        end
                
        E = zero(UInt64)
        for j in 1:permsize(M)-1
            s = cache.vec[j] < cache.vec[j+1]
            E |= s << (j-1)
        end

        j = permsize(M)
        s = cache.vec[j] < cache.vec[1]  # circular comparison
        E |= s << (j-1)
        
        vout[i] = E
    end
    
    vout
end

function predict(M::BinWalk, db::AbstractDatabase; minbatch::Int=4)
    D = Matrix{UInt64}(undef, nperms(M), length(db))
    B = [PermsCacheEncoder(M) for _ in 1:Threads.nthreads()]
    
    @batch per=thread minbatch=minbatch for i in eachindex(db)
        encode_object!(M, view(D, :, i), db[i], B[Threads.threadid()])
    end

    MatrixDatabase(D)
end

predict(M::BinWalk, v::AbstractVector) = permscache() do cache
    out = Vector{UInt64}(undef, nperms(M))
    encode_object!(M, out, v, cache)
end

#=
function encode(M::BinWalk, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = encode_database(M, db_)
    queries = encode_database(M, queries_)
    params["surrogate"] = "RBW"
    params["permsize"] = permsize(M)
    params["nperms"] = nperms(M)
    
    (; db, queries, params, dist)
end
=#
