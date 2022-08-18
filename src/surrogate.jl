using Random

abstract type AbstractSurrogate end

struct BinaryHammingFixedPairs <: AbstractSurrogate
    kscale::Int
end

kscale(m::BinaryHammingFixedPairs) = m.kscale

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

#############

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

kscale(m::BinaryHammingSurrogate) = m.kscale
encode(B::BinaryHammingSurrogate, v, p::Tuple)::Bool = v[p[1]] < v[p[2]]
encode(B::BinaryHammingSurrogate, v) = (encode.((B,), (v,), B.pairs)).chunks

function encode(B::BinaryHammingSurrogate, X::AbstractDatabase)
    V = Vector{Vector{UInt64}}(undef, length(X))
    Threads.@threads for i in eachindex(X)
        V[i] = encode(B, X[i])
    end

    VectorDatabase(V)
end

function encode(B::BinaryHammingSurrogate, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = encode(B, db_)
	queries = encode(B, queries_)
    params["surrogate"] = "BHS"
    params["kscale"] = B.kscale
    params["npairs"] = length(B.pairs)
    (; db, queries, params, dist)
end

#=

struct MaxPoolSurrogate <: AbstractSurrogate
    kscale::Int
    pool::Matrix{Int32}
    
    function MaxPoolSurrogate(samplesize::Integer, npools::Integer, dim::Integer, kscale::Integer)
        pool = Matrix{Float32}(undef, samplesize, npools)
        perm = Vector{Int32}(1:dim)
      
        for i in 1:npools
            randperm!(perm)
            pool[:, i] .= view(perm, 1:samplesize)
        end
        
        new(kscale, pool)
    end
end

samplesize(M::MaxPoolSurrogate) = size(M.pool, 1)
npools(M::MaxPoolSurrogate) = size(M.pool, 2)
kscale(M::MaxPoolSurrogate) = M.kscale

function encode(M::MaxPoolSurrogate, vout, v)
    for i in eachindex(vout)
        vout[i] = maximum(j -> v[j], view(M.pool, :, i))
    end
    
    vout
end

function encode(M::MaxPoolSurrogate, db_::AbstractDatabase)
    D = Matrix{Float32}(undef, npools(M), length(db_))
    
    for i in eachindex(db_)
        encode(M, view(D, :, i), db_[i])
    end

    MatrixDatabase(D)
end

function encode(M::MaxPoolSurrogate, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = L2Distance()
    db = encode(M, db_)
    queries = encode(M, queries_)
    params["surrogate"] = "maxpool"
    params["kscale"] = M.kscale
    (; db, queries, params, dist)
end

=#

###############

struct MaxHashSurrogate <: AbstractSurrogate
    kscale::Int
    pool::Matrix{Int32}
    
    function MaxHashSurrogate(samplesize::Integer, npools::Integer, dim::Integer, kscale::Integer)
        samplesize < 256 || throw(ArgumentError("samplesize < 256: $samplesize"))
        pool = Matrix{Int32}(undef, samplesize, npools)
        perm = Vector{Int32}(1:dim)
      
        for i in 1:npools
            randperm!(perm)
            pool[:, i] .= view(perm, 1:samplesize)
        end
        
        new(kscale, pool)
    end
end

samplesize(M::MaxHashSurrogate) = size(M.pool, 1)
npools(M::MaxHashSurrogate) = size(M.pool, 2)
kscale(M::MaxHashSurrogate) = M.kscale

function encode(M::MaxHashSurrogate, vout, v)
    for i in eachindex(vout)
        #vout[i] = maximum(j -> v[j], view(M.pool, :, i))
        vout[i] = findmax(j -> v[j], view(M.pool, :, i)) |> last
    end
    
    vout
end

function encode(M::MaxHashSurrogate, db_::AbstractDatabase)
    D = Matrix{UInt8}(undef, npools(M), length(db_))
    
    Threads.@threads for i in eachindex(db_)
        encode(M, view(D, :, i), db_[i])
    end

    MatrixDatabase(D)
end

function encode(M::MaxHashSurrogate, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = StringHammingDistance()
    db = encode(M, db_)
    queries = encode(M, queries_)
    params["surrogate"] = "MaxHash"
    params["samplesize"] = samplesize(M)
    params["npools"] = npools(M)
    params["kscale"] = kscale(M)
    
    (; db, queries, params, dist)
end

