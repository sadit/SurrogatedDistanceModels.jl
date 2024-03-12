export BinEncoder

struct BinEncoder <: AbstractSurrogate
    pairs::Array{Int32}

end

distance(::BinEncoder) = BinaryHammingDistance()

function fit(::Type{BinEncoder}, npairs::Integer, dim::Integer)
    nblocks = ceil(Int, npairs / 64)
    P = zeros(Int32, 2, 64, nblocks)
    prob = 1.2 * npairs / ((dim^2 + dim) / 2)

    ii = 1
    for i in 1:dim
        for j in i+1:dim
            if prob < rand() && ii < length(P)
                P[ii] = i
                P[ii+1] = j
                ii += 2
            end
        end
    end

    BinEncoder(P)
end

npairs(m::BinEncoder) = 64 * size(m.pairs, 3)

function encode_object!(B::BinEncoder, vout, v)
    for i in eachindex(vout)
        E = zero(UInt64)
        for j in 1:64
            a, b = B.pairs[:, j, i]
            s = v[a] > v[b]
            E |= s << (j-1)
        end

        vout[i] = E
    end

    vout
end

function predict(B::BinEncoder, obj::AbstractVector)
    predict(B, Vector{UInt64}(undef, size(B.pairs, 3)), obj)
end

function predict(B::BinEncoder, X::AbstractDatabase; minbatch::Int=4)
    D = Matrix{UInt64}(undef, size(B.pairs, 3), length(X))
    @batch per=thread minbatch=minbatch for i in eachindex(X)
        encode_object!(B, view(D, :, i), X[i])
    end

    MatrixDatabase(D)
end

#=
function encode(B::BinEncoder, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = predict(B, db_)
   	queries = predict(B, queries_)
    params["surrogate"] = "CBE"
    params["npairs"] = npairs(B)
    (; db, queries, params, dist)
end
=#
