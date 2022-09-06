export BinPerms


struct BinPerms{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractSurrogate
    dist::DistType
    refs::DbType
    pool::Matrix{Int32}
    shift::Int
    
    function BinPerms(dist::SemiMetric, refs::AbstractDatabase, nperms::Integer; permsize::Integer=64, shift::Integer=permsize ÷ 3)
        2 <= permsize <= 64 || throw(ArgumentError("invalid permsize $permsize"))
        pool = Matrix{Int32}(undef, permsize, nperms)
        perm = Vector{Int32}(1:length(refs))

        for i in 1:nperms
            shuffle!(perm)
            pool[:, i] .= view(perm, 1:permsize)
        end
        
        new{typeof(dist), typeof(refs)}(dist, refs, pool, shift)
    end
end

@inline permsize(M::BinPerms) = size(M.pool, 1)
@inline nperms(M::BinPerms) = size(M.pool, 2)
@inline shift(M::BinPerms) = M.shift

function encode_object!(M::BinPerms, vout, v, cache::PermsCacheEncoder)
    for i in 1:nperms(M)
        col = view(M.pool, :, i)
        for (j, k) in enumerate(col)
            cache.vec[j] = evaluate(M.dist, M.refs[k], v)
        end
        
        sortperm!(cache.P, cache.vec)
        invperm!(cache.invP, cache.P)
        
        E = zero(UInt64)
        for j in 1:permsize(M)
            s = abs(cache.invP[j] - j) > shift(M)
            E |= s << (j-1)
        end
        
        vout[i] = E
    end
    
    vout
end

function encode_database(M::BinPerms, db::AbstractDatabase)
    D = Matrix{UInt64}(undef, nperms(M), length(db))
    B = [PermsCacheEncoder(M) for _ in 1:Threads.nthreads()]
    
    Threads.@threads for i in eachindex(db)
        encode_object!(M, view(D, :, i), db[i], B[Threads.threadid()])
    end

    MatrixDatabase(D)
end

function encode(M::BinPerms, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = encode_database(M, db_)
    queries = encode_database(M, queries_)
    params["surrogate"] = "RBP"
    params["permsize"] = permsize(M)
    params["nperms"] = nperms(M)
    params["shift"] = shift(M)
    
    (; db, queries, params, dist)
end
