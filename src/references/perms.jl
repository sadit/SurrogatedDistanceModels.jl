export Perms

struct Perms{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractSurrogate
    dist::DistType
    refs::DbType
    pool::Matrix{Int32}

    function Perms(dist::SemiMetric, refs::AbstractDatabase, nperms::Integer; permsize::Integer=64)
        2 <= permsize <= length(refs)|| throw(ArgumentError("invalid permsize $permsize"))
        pool = Matrix{Int32}(undef, permsize, nperms)
        perm = Vector{Int32}(1:dim)

        for i in 1:nperms
            shuffle!(perm)
            pool[:, i] .= view(perm, 1:permsize)
        end
        
        new{typeof(dist), typeof(refs)}(dist, refs, pool)
    end
end

@inline permsize(M::Perms) = size(M.pool, 1)
@inline nperms(M::Perms) = size(M.pool, 2)

function encode_object!(M::Perms, vout, v, cache::PermsCacheEncoder)
    for i in 1:nperms(M)
        col = view(M.pool, :, i)
        for (j, k) in enumerate(col)
            cache.vec[j] = evaluate(M.dist, M.refs[k], v)
        end
        
        sortperm!(cache.P, cache.vec)
        invperm!(cache.invP, cache.P)
        vout[:, i] .= cache.invP
    end
    
    vout
end

function encode_database(M::Perms, db_::AbstractDatabase)
    D = Matrix{Float32}(undef, permsize(M) * nperms(M), length(db_))
    B = [PermsCacheEncoder(M) for _ in 1:Threads.nthreads()]
    
    Threads.@threads for i in eachindex(db_)
        x = reshape(view(D, :, i), permsize(M), nperms(M))
        encode_object!(M, x, db_[i], B[Threads.threadid()])
    end

    MatrixDatabase(D)
end

function encode(M::Perms, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = SqL2Distance()
    db = encode_database(M, db_)
    queries = encode_database(M, queries_)
    params["surrogate"] = "RP"
    params["permsize"] = permsize(M)
    params["nperms"] = nperms(M)
    
    (; db, queries, params, dist)
end
