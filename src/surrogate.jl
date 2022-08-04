abstract type AbstractSurrogate end

struct BinaryHammingFixedPairs <: AbstractSurrogate
    kscale::Int
end

encode(::BinaryHammingFixedPairs, v, i::Integer)::Bool = v[i] < v[end-i+1]
encode(B::BinaryHammingFixedPairs, v) = (encode.((B,), (v,), 1:length(v) ÷ 2)).chunks

function encode(B::BinaryHammingFixedPairs, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = VectorDatabase([encode(B, db_[i]) for i in eachindex(db_)])
	queries = VectorDatabase([encode(B, queries_[i]) for i in eachindex(queries_)])
    params["surrogate"] = "BHFP"
    params["kscale"] = B.kscale
    (; db, queries, params, dist)
end


function random_sorted_pair(dim)
    a = rand(one(Int32):dim)
    b = rand(one(Int32):dim)
    a < b ? (a, b) : (b, a)
end

struct BinaryHammingSurrogate <: AbstractSurrogate
    kscale::Int
    pairs::Vector{Tuple{Int32,Int32}}
    
    function BinaryHammingSurrogate(kscale::Integer, npairs::Integer, dim::Integer)
        dim = convert(Int32, dim)
        P = Set([random_sorted_pair(dim)])
        for i in 2:npairs
            push!(P, random_sorted_pair(dim))
        end
        
        S = collect(P); sort!(S)
        new(kscale, S)
    end
end

encode(B::BinaryHammingSurrogate, v, p::Tuple)::Bool = v[p[1]] < v[p[2]]
encode(B::BinaryHammingSurrogate, v) = (encode.((B,), (v,), B.pairs)).chunks

function encode(B::BinaryHammingSurrogate, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = VectorDatabase([encode(B, db_[i]) for i in eachindex(db_)])
	queries = VectorDatabase([encode(B, queries_[i]) for i in eachindex(queries_)])
    params["surrogate"] = "BHS"
    params["kscale"] = B.kscale
    params["npairs"] = length(B.pairs)
    (; db, queries, params, dist)
end
