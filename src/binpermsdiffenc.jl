export BinPermsDiffEnc

struct BinPermsDiffEnc{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractSurrogate
    dist::DistType
    refs::DbType
    pool::Matrix{UInt32}    
end

distance(::BinPermsDiffEnc) = BinaryHammingDistance()

function fit(::Type{BinPermsDiffEnc}, dist::SemiMetric, refs::AbstractDatabase, nbits::Int)
    permsize = 64
    nperms = nbits ÷ permsize
    nbits % permsize == 0 || throw(ArgumentError("nbits should be a factor of 64"))
    pool = Matrix{UInt32}(undef, permsize, nperms) 
    P = collect(1:permsize)

    for i in 1:nperms
        shuffle!(P)
        pool[:, i] .= P[1:permsize]
    end

    BinPermsDiffEnc(dist, refs, pool)
end

@inline permsize(M::BinPermsDiffEnc) = size(M.pool, 1)
@inline nperms(M::BinPermsDiffEnc) = size(M.pool, 2)

function encode_object!(M::BinPermsDiffEnc, vout, v, cache::PermsCacheEncoder)
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

function predict(M::BinPermsDiffEnc, db::AbstractDatabase; minbatch::Int=4)
    D = Matrix{UInt64}(undef, nperms(M), length(db))
    B = [PermsCacheEncoder(permsize(M)) for _ in 1:Threads.nthreads()]
    
    @batch per=thread minbatch=minbatch for i in eachindex(db)
        encode_object!(M, view(D, :, i), db[i], B[Threads.threadid()])
    end

    MatrixDatabase(D)
end

predict(M::BinPermsDiffEnc, v::AbstractVector) = permscache(permsize(M)) do cache
    out = Vector{UInt64}(undef, nperms(M))
    encode_object!(M, out, v, cache)
end

#=
function encode(M::BinPermsDiffEnc, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = encode_database(M, db_)
    queries = encode_database(M, queries_)
    params["surrogate"] = "RBW"
    params["permsize"] = permsize(M)
    params["nperms"] = nperms(M)
    
    (; db, queries, params, dist)
end
=#
