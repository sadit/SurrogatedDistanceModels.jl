export NearestReference

struct NearestReference{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractSurrogate
    dist::DistType
    refs::DbType
    pool::Matrix{Int32}
end

distance(::NearestReference) = LevenshteinDistance()

function fit(::Type{NearestReference}, dist::SemiMetric, refs::AbstractDatabase; permsize::Integer=32)
    n = length(refs)
    @assert n % permsize == 0
    npools = n ÷ permsize
    pool = Matrix{Int32}(undef, permsize, npools)
    P = vec(pool)
    for i in 1:n
        P[i] = i
    end

    shuffle!(P)
    NearestReference(dist, refs, pool)
end

permsize(M::NearestReference) = size(M.pool, 1)
npools(M::NearestReference) = size(M.pool, 2)

function encode_object!(M::NearestReference, vout, v, nn)
    for i in eachindex(vout)
        col = view(M.pool, :, i)
        nn = reuse!(nn, 1)
        for j in col
            push_item!(nn, IdWeight(j, evaluate(M.dist, v, M.refs[j])))
        end

        vout[i] = argmin(nn)
    end

    vout
end

function predict(M::NearestReference, db::AbstractDatabase; minbatch::Int=4)
    D = Matrix{UInt32}(undef, npools(M), length(db))
    @batch per=thread minbatch=minbatch for i in eachindex(db)
        encode_object!(M, view(D, :, i), db[i], getknnresult(1))
    end

    MatrixDatabase(D)
end

predict(M::NearestReference, v::AbstractVector) = encode_object!(M, Vector{UInt32}(undef, npools(M)), v, getknnresult(1))

#=
function encode(M::NearestReference, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = StringHammingDistance()
    db = predict(M, db_)
    queries = predict(M, queries_)
    params["surrogate"] = "RN"
    params["permsize"] = permsize(M)
    params["npools"] = npools(M)
    
    (; db, queries, params, dist)
end
=#
