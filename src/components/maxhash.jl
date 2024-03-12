export MaxHash

struct MaxHash <: AbstractSurrogate
    pool::Matrix{Int32}
end

distance(::MaxHash) = StringHammingDistance()

function fit(::Type{MaxHash}, npools::Integer, dim::Integer; samplesize=8)
    samplesize < 256 || throw(ArgumentError("samplesize < 256: $samplesize"))
    pool = Matrix{Int32}(undef, samplesize, npools)
    perm = Vector{Int32}(1:dim)
  
    for i in 1:npools
        randperm!(perm)
        pool[:, i] .= view(perm, 1:samplesize)
    end
    
    MaxHash(pool)
end

samplesize(M::MaxHash) = size(M.pool, 1)
npools(M::MaxHash) = size(M.pool, 2)

function encode_object!(M::MaxHash, vout, v)
    for i in eachindex(vout)
        vout[i] = findmax(j -> v[j], view(M.pool, :, i)) |> last
    end
    
    vout
end

function predict(M::MaxHash, db_::AbstractDatabase; minbatch::Int=4)
    D = Matrix{UInt8}(undef, npools(M), length(db_))
    
    @batch per=thread minbatch=minbatch for i in eachindex(db_)
        encode_object!(M, view(D, :, i), db_[i])
    end

    MatrixDatabase(D)
end

function predict(M::MaxHash, v::AbstractVector)
    encode_object!(M, Vector{UInt8}(undef, npools(M)), v)
end
#=
function encode(M::MaxHash, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = StringHammingDistance()
    db = encode(M, db_)
    queries = encode(M, queries_)
    params["surrogate"] = "CMH"
    params["samplesize"] = samplesize(M)
    params["npools"] = npools(M)
    
    (; db, queries, params, dist)
end
=#
