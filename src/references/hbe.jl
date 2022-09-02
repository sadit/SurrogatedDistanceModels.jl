export HyperplaneBinaryEncoding

struct HyperplaneBinaryEncoding{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractSurrogate
    dist::DistType
    refs::DbType
end

numrefs(B::HyperplaneBinaryEncoding) = length(B.refs) ÷ 2

function encode_pair(B::HyperplaneBinaryEncoding, v, i::Integer)::Bool
    i = 2i
    evaluate(B.dist, B.refs[i-1], v) < evaluate(B.dist, B.refs[i], v)
end

function encode_object!(B::HyperplaneBinaryEncoding, vout, v)
    for i in eachindex(vout)
        vout[i] = encode_pair(B, v, i)
    end
end

function encode_database(B::HyperplaneBinaryEncoding, X::AbstractDatabase)
    D = BitMatrix(numrefs(B), length(X))

    Threads.@threads for i in eachindex(X)
        encode_object!(B, view(D, :, i), X[i])
    end

    MatrixDatabase(D)
end

function encode(B::HyperplaneBinaryEncoding, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = encode_database(B, db_)
	queries = encode_database(B, queries_)
    params["surrogate"] = "RHB"
    params["numrefs"] = numrefs(B)
    (; db, queries, params, dist)
end