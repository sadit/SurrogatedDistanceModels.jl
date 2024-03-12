export Perms

struct Perms{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractSurrogate
    dist::DistType
    refs::DbType
    pool::Matrix{Int32}
end

distance(::Perms) = L2Distance()
    
function fit(::Type{Perms}, dist::SemiMetric, refs::AbstractDatabase, nperms::Integer; permsize::Integer=64)
    2 <= permsize <= length(refs)|| throw(ArgumentError("invalid permsize $permsize"))
    pool = Matrix{Int32}(undef, permsize, nperms)
    perm = Vector{Int32}(1:length(refs))

    for i in 1:nperms
        shuffle!(perm)
        pool[:, i] .= view(perm, 1:permsize)
    end
    
    new{typeof(dist), typeof(refs)}(dist, refs, pool)
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

function predict(M::Perms, db_::AbstractDatabase; minbatch=4)
    D = Matrix{Float32}(undef, permsize(M) * nperms(M), length(db_))
    B = [PermsCacheEncoder(M) for _ in 1:Threads.nthreads()]
    
    @batch per=thread minbatch=minbatch for i in eachindex(db_)
        x = reshape(view(D, :, i), permsize(M), nperms(M))
        encode_object!(M, x, db_[i], B[Threads.threadid()])
    end

    MatrixDatabase(D)
end

function predict(M::Perms, obj)
    permscache() do cache
        out = Vector{Float32}(undef, permsize() * nperms(M))
        encode_object!(M, out, obj, cache)
    end
end

#=
function encode(M::Perms, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = SqL2Distance()
    db = predict(M, db_)
    queries = predict(M, queries_)
    params["surrogate"] = "RP"
    params["permsize"] = permsize(M)
    params["nperms"] = nperms(M)
    
    (; db, queries, params, dist)
end
=#
