export NearestReference

struct NearestReference{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractSurrogate
    dist::DistType
    refs::DbType
    pool::Matrix{Int32}

    function NearestReference(dist::SemiMetric, refs::AbstractDatabase, npools::Integer, dim::Integer; samplesize::Integer=8)
        samplesize < 256 || throw(ArgumentError("samplesize < 256: $samplesize"))
        pool = Matrix{Int32}(undef, samplesize, npools)
        perm = Vector{Int32}(1:dim)
      
        for i in 1:npools
            randperm!(perm)
            pool[:, i] .= view(perm, 1:samplesize)
        end
        
        new{typeof(dist), typeof(refs)}(dist, refs, pool)
    end
end

samplesize(M::NearestReference) = size(M.pool, 1)
npools(M::NearestReference) = size(M.pool, 2)

function encode_object!(M::NearestReference, vout, v)
    for i in eachindex(vout)
        vout[i] = findmax(j -> v[j], view(M.pool, :, i)) |> last
    end
    
    vout
end

function encode_database(M::NearestReference, db::AbstractDatabase)
    D = Matrix{UInt8}(undef, npools(M), length(db))
    
    Threads.@threads for i in eachindex(db)
        encode_object!(M, view(D, :, i), db[i])
    end

    MatrixDatabase(D)
end

function encode(M::NearestReference, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = StringHammingDistance()
    db = encode_database(M, db_)
    queries = encode_database(M, queries_)
    params["surrogate"] = "RN"
    params["samplesize"] = samplesize(M)
    params["npools"] = npools(M)
    
    (; db, queries, params, dist)
end
