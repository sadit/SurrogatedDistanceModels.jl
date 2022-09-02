struct RandWalk <: AbstractSurrogate
    kscale::Int
    pool::Matrix{Int32}
    
    function RandWalk(permsize::Integer, nperms::Integer, dim::Integer, kscale::Integer)
        2 <= permsize || throw(ArgumentError("invalid permsize $permsize"))
        @show :RandWalk, permsize, nperms, dim, kscale
        pool = Matrix{Int32}(undef, permsize, nperms)
        perm = Vector{Int32}(1:dim)

        for i in 1:nperms
            shuffle!(perm)
            pool[:, i] .= view(perm, 1:permsize)
        end
        
        new(kscale, pool)
    end
end

@inline permsize(M::RandWalk) = size(M.pool, 1)
@inline nperms(M::RandWalk) = size(M.pool, 2)
@inline kscale(M::RandWalk) = M.kscale

function encode_object(M::RandWalk, vout, v, cache::PermsCacheEncoder)
    for i in 1:nperms(M)
        col = view(M.pool, :, i)
        for (j, k) in enumerate(col)
            cache.vec[j] = v[k]
        end
        
        E = zero(UInt64)
        for j in 1:permsize(M)-1
            if cache.vec[j]
            s = abs(cache.invP[j] - j) > shift(M)
            E |= s << (j-1)
        end
        
        vout[i] = E
    end
    
    vout
end

function encode_database(M::RandWalk, db_::AbstractDatabase)
    D = Matrix{UInt64}(undef, nperms(M), length(db_))
    B = [PermsCacheEncoder(M) for i in 1:Threads.nthreads()]
    
    Threads.@threads for i in eachindex(db_)
        encode_object(M, view(D, :, i), db_[i], B[Threads.threadid()])
    end

    MatrixDatabase(D)
end

function encode(M::RandWalk, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = SqL2Distance()
    db = encode_database(M, db_)
    queries = encode_database(M, queries_)
    params["surrogate"] = "RW"
    params["permsize"] = permsize(M)
    params["nperms"] = nperms(M)
    params["kscale"] = kscale(M)
    
    (; db, queries, params, dist)
end