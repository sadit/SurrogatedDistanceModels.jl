export BinPerms

struct BinPerms{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractSurrogate
    dist::DistType
    refs::DbType
    pool::Matrix{Int32}
    shift::Int    
end

distance(::BinPerms) = BinaryHammingDistance()

function fit(::Type{BinPerms}, dist::SemiMetric, refs::AbstractDatabase, nbits::Int; delta_shift::AbstractFloat=0.33333f0)
    permsize = 64
    nbits % permsize  == 0 || throw(ArgumentError("nbits must be a factor of 64"))
    nperms = nbits ÷ permsize
    shift = ceil(Int, permsize * delta_shift)
    length(refs) >= permsize || throw(ArgumentError("the number of references should be higher than 64"))
    pool = Matrix{Int32}(undef, permsize, nperms)
    perm = Vector{Int32}(1:length(refs))

    for i in 1:nperms
        shuffle!(perm)
        pool[:, i] .= view(perm, 1:permsize)
    end
    
    BinPerms(dist, refs, pool, shift)
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

function predict(M::BinPerms, db::AbstractDatabase; minbatch::Int=4)
    D = Matrix{UInt64}(undef, nperms(M), length(db))
    B = [PermsCacheEncoder(permsize(M)) for _ in 1:Threads.nthreads()]
    
    @batch per=thread minbatch=minbatch for i in eachindex(db)
        encode_object!(M, view(D, :, i), db[i], B[Threads.threadid()])
    end

    MatrixDatabase(D)
end

predict(M::BinPerms, v::AbstractVector) = permscache(permsize(M)) do cache
    encode_object!(M, Vector{UInt64}(undef, nperms(M)), v, cache)
end

#=
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
=#
