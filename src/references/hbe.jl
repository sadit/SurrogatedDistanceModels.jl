export HyperplaneEncoding, select_random_refs

struct HyperplaneEncoding{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractSurrogate
    dist::DistType
    refs::DbType
end

distance(::HyperplaneEncoding) = BinaryHammingDistance()

function fit(::Type{HyperplaneEncoding}, dist::SemiMetric, refs::MatrixDatabase, npairs::Integer)
    npairs % 64 == 0 || throw(ArgumentError("npairs must be a factor of 64"))
    length(refs) == 2npairs || throw(ArgumentError("refs must cointain 2*npairs elements"))
    HyperplaneEncoding(dist, refs)
end

numrefs(B::HyperplaneEncoding) = length(B.refs) ÷ 2
nblocks(B::HyperplaneEncoding) = length(B.refs) ÷ 128

function encode_pair(B::HyperplaneEncoding, v, i::Integer)::Bool
    i = 2i
    evaluate(B.dist, B.refs[i-1], v) < evaluate(B.dist, B.refs[i], v)
end

function encode_object!(B::HyperplaneEncoding, vout, v)
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

function predict(B::HyperplaneEncoding, X::AbstractDatabase)
    D = Matrix{UInt64}(undef, nblocks(B), length(X))

    Threads.@threads for i in eachindex(X)
        encode_object!(B, view(D, :, i), X[i])
    end

    MatrixDatabase(D)
end

predict(B::HyperplaneEncoding, v::AbstractVector) = encode_object!(B, Vector{UInt64}(undef, nblocks(B)), v)

#=
function encode(B::HyperplaneEncoding, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = encode_database(B, db_)
	queries = encode_database(B, queries_)
    params["surrogate"] = "RHB"
    params["numrefs"] = numrefs(B)
    (; db, queries, params, dist)
end
=#
