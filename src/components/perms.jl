export CompPerms

struct CompPerms <: AbstractSurrogate
    pool::Matrix{Int32}
    
end

distance(::CompPerms) = L2Distance()
    
function fit(::Type{CompPerms}, nperms::Integer, dim::Integer; permsize::Integer=64)
    2 <= permsize || throw(ArgumentError("invalid permsize $permsize"))
    @show :CompPerms, permsize, nperms, dim
    pool = Matrix{Int32}(undef, permsize, nperms)
    perm = Vector{Int32}(1:dim)

    for i in 1:nperms
        shuffle!(perm)
        pool[:, i] .= view(perm, 1:permsize)
    end
    
    new(pool)
end

@inline permsize(M::CompPerms) = size(M.pool, 1)
@inline nperms(M::CompPerms) = size(M.pool, 2)

function encode_object!(M::CompPerms, vout, v, cache::PermsCacheEncoder)
    for i in 1:nperms(M)
        col = view(M.pool, :, i)
        for (j, k) in enumerate(col)
            cache.vec[j] = v[k]
        end
        
        sortperm!(cache.P, cache.vec)
        invperm!(cache.invP, cache.P)
        vout[:, i] .= cache.invP
    end
    
    vout
end

function predict(M::CompPerms, db_::AbstractDatabase; minbatch=4)
    D = Matrix{Float32}(undef, permsize(M) * nperms(M), length(db_))
    B = [PermsCacheEncoder(permsize(M)) for _ in 1:Threads.nthreads()]
    
    @batch per=thread minbatch=minbatch for i in eachindex(db_)
        x = reshape(view(D, :, i), permsize(M), nperms(M))
        encode_object!(M, x, db_[i], B[Threads.threadid()])
    end

    MatrixDatabase(D)
end

predict(M::CompPerms, v::AbstractVector) = permscache(permsize(M)) do cache 
    encode_object!(M, Vector{Float32}(undef, permsize(M) * nperms(M)), v, cache)
end

#=
function encode(M::CompPerms, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = SqL2Distance()
    db = predict(M, db_)
    queries = predict(M, queries_)
    params["surrogate"] = "CP"
    params["permsize"] = permsize(M)
    params["nperms"] = nperms(M)
    
    (; db, queries, params, dist)
end
=#
