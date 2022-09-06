export NearestReference

struct NearestReference{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractSurrogate
    dist::DistType
    refs::DbType
    pool::Matrix{Int32}

    function NearestReference(dist::SemiMetric, refs::AbstractDatabase; permsize::Integer=32)
        n = length(refs)
        @assert n % permsize == 0
        npools = n ÷ permsize
        pool = Matrix{Int32}(undef, permsize, npools)
        P = vec(pool)
        for i in 1:n
            P[i] = i
        end

        shuffle!(P)
        new{typeof(dist), typeof(refs)}(dist, refs, pool)
    end
end

permsize(M::NearestReference) = size(M.pool, 1)
npools(M::NearestReference) = size(M.pool, 2)

function encode_object!(M::NearestReference, vout, v, nn)
    for i in eachindex(vout)
        col = view(M.pool, :, i)
        nn = reuse!(nn, 1)
        for j in col
            push!(nn, j, evaluate(M.dist, v, M.refs[j]))
        end

        vout[i] = argmin(nn)
    end

    vout
end

function encode_database(M::NearestReference, db::AbstractDatabase)
    D = Matrix{UInt32}(undef, npools(M), length(db))
    Threads.@threads for i in eachindex(db)
        encode_object!(M, view(D, :, i), db[i], getknnresult(1))
    end

    MatrixDatabase(D)
end

function encode(M::NearestReference, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = StringHammingDistance()
    db = encode_database(M, db_)
    queries = encode_database(M, queries_)
    params["surrogate"] = "RN"
    params["permsize"] = permsize(M)
    params["npools"] = npools(M)
    
    (; db, queries, params, dist)
end
