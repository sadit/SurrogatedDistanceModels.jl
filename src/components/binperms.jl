export CompBinPerms

struct CompBinPerms <: AbstractSurrogate
    shift::Int
    pool::Matrix{Int32}    
end

distance(::CompBinPerms) = BinaryHammingDistance()

function fit(::Type{CompBinPerms}, nperms::Integer, dim::Integer; permsize::Integer=64, shift=dim ÷ 3)
    2 <= permsize <= 64 || throw(ArgumentError("invalid permsize $permsize"))
    pool = Matrix{Int32}(undef, permsize, nperms)
    perm = Vector{Int32}(1:dim)

    for i in 1:nperms
        shuffle!(perm)
        pool[:, i] .= view(perm, 1:permsize)
    end
    
    new(shift, pool)
end

@inline permsize(M::CompBinPerms) = size(M.pool, 1)
@inline nperms(M::CompBinPerms) = size(M.pool, 2)
@inline shift(M::CompBinPerms) = M.shift

function encode_object!(M::CompBinPerms, vout, v, cache::PermsCacheEncoder)
    for i in 1:nperms(M)
        col = view(M.pool, :, i)
        for (j, k) in enumerate(col)
            cache.vec[j] = v[k]
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

function predict(M::CompBinPerms, db::AbstractDatabase; minbatch::Int=4)
    D = Matrix{UInt64}(undef, nperms(M), length(db))
    B = [PermsCacheEncoder(M) for i in 1:Threads.nthreads()]
    
    @batch per=thread minbatch=minbatch for i in eachindex(db)
        encode_object!(M, view(D, :, i), db[i], B[Threads.threadid()])
    end

    MatrixDatabase(D)
end

function predict(M::CompBinPerms, v::AbstractVector)
    permscache() do cache
        encode_object!(M, Vector{UInt64}(undef, nperms(M)), v, cache)
    end
end

#=
function encode(M::CompBinPerms, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = encode_database(M, db_)
    queries = encode_database(M, queries_)
    params["surrogate"] = "CBP"
    params["permsize"] = permsize(M)
    params["nperms"] = nperms(M)
    params["shift"] = shift(M)
    
    (; db, queries, params, dist)
end
=#
