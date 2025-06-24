export NearestReference

struct NearestReference{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractSurrogate
    dist::DistType
    refs::DbType
    pool::Matrix{UInt32}
end

distance(::NearestReference) = StringHammingDistance()

function fit(::Type{NearestReference}, dist::SemiMetric, refs::AbstractDatabase; seqsize::Int, vocsize::Int=32)
    vocsize <= 255 || throw(ArgumentError("vocsize should be smaller than 255 since we use 8bit to represent each ref"))
    n = length(refs)
    vocsize < 0.5 * n || throw(ArgumentError("vocsize is too small"))

    pool = Matrix{UInt32}(undef, vocsize, seqsize)
    P = collect(1:n)
    for c in eachcol(pool)
        shuffle!(P)
        c .= @view P[1:vocsize]
    end

    NearestReference(dist, refs, pool)
end

vocsize(M::NearestReference) = size(M.pool, 1)
seqsize(M::NearestReference) = size(M.pool, 2)

function encode_object!(M::NearestReference, vout, v, nn)
    for i in eachindex(vout)
        col = view(M.pool, :, i)
        nn = reuse!(nn, 1)
        for (j, candID) in enumerate(col)
            push_item!(nn, IdWeight(j, evaluate(M.dist, v, M.refs[candID])))
        end

        vout[i] = argmin(nn)
    end

    vout
end

function predict(M::NearestReference, db::AbstractDatabase; minbatch::Int=4)
    D = Matrix{UInt8}(undef, seqsize(M), length(db))
    ctx = GenericContext()
    @batch per=thread minbatch=minbatch for i in eachindex(db)
        encode_object!(M, view(D, :, i), db[i], getknnresult(1, ctx))
    end

    MatrixDatabase(D)
end

predict(M::NearestReference, v::AbstractVector) = encode_object!(M, Vector{UInt8}(undef, seqsize(M)), v, KnnResult(1))

#=
function encode(M::NearestReference, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = StringHammingDistance()
    db = predict(M, db_)
    queries = predict(M, queries_)
    params["surrogate"] = "RN"
    params["vocsize"] = vocsize(M)
    params["seqsize"] = seqsize(M)
    
    (; db, queries, params, dist)
end
=#
