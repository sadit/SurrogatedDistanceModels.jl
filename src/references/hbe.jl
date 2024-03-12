export HyperplaneBinaryEncoding, select_random_refs

function select_random_refs(X::Matrix, m::Integer)
    n = size(X, 2)
    @assert 2 <= m <= n
    I = collect(1:n)
    shuffle!(I)
    MatrixDatabase(X[:, I[1:m]])
end

struct HyperplaneBinaryEncoding{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractSurrogate
    dist::DistType
    refs::DbType
end

distance(::HyperplaneBinaryEncoding) = BinaryHammingDistance()

function fit(::Type{HyperplaneBinaryEncoding}, dist::SemiMetric, X::MatrixDatabase, npairs::Integer)
    @assert npairs % 64 == 0
    HyperplaneBinaryEncoding(dist, select_random_refs(X.matrix, 2npairs))
end

numrefs(B::HyperplaneBinaryEncoding) = length(B.refs) ÷ 2
nblocks(B::HyperplaneBinaryEncoding) = length(B.refs) ÷ 128

function encode_pair(B::HyperplaneBinaryEncoding, v, i::Integer)::Bool
    i = 2i
    evaluate(B.dist, B.refs[i-1], v) < evaluate(B.dist, B.refs[i], v)
end

function encode_object!(B::HyperplaneBinaryEncoding, vout, v)
    j = 0

    for i in eachindex(vout)
        j += 1
        E = zero(UInt64)
        for j in 1:64
            s = encode_pair(B, v, j)
            E |= s << (j-1)
        end

        vout[i] = E
    end
end

function predict(B::HyperplaneBinaryEncoding, X::AbstractDatabase)
    D = Matrix{UInt64}(undef, nblocks(B), length(X))

    Threads.@threads for i in eachindex(X)
        encode_object!(B, view(D, :, i), X[i])
    end

    MatrixDatabase(D)
end

predict(B::HyperplaneBinaryEncoding, v::AbstractVector) = encode_object!(B, Vector{UInt64}(undef, nblocks(B)), v)

#=
function encode(B::HyperplaneBinaryEncoding, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = encode_database(B, db_)
	queries = encode_database(B, queries_)
    params["surrogate"] = "RHB"
    params["numrefs"] = numrefs(B)
    (; db, queries, params, dist)
end
=#
