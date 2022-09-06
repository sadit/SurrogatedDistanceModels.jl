export BinEncoder

struct BinEncoder <: AbstractSurrogate
    pairs::Array{Int32}

    function BinEncoder(npairs::Integer, dim::Integer)
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
    
        new(P)
    end
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
end

function encode_database(B::BinEncoder, X::AbstractDatabase)
    D = Matrix{UInt64}(undef, size(B.pairs, 3), length(X))
    Threads.@threads for i in eachindex(X)
        encode_object!(B, view(D, :, i), X[i])
    end

    MatrixDatabase(D)
end

function encode(B::BinEncoder, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = encode_database(B, db_)
	queries = encode_database(B, queries_)
    params["surrogate"] = "CBE"
    params["npairs"] = npairs(B)
    (; db, queries, params, dist)
end