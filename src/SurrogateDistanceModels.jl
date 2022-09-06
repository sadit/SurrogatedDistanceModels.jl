module SurrogateDistanceModels

using SimilaritySearch, Random

export AbstractSurrogate, encode_database

abstract type AbstractSurrogate end

struct PermsCacheEncoder
    P::Vector{Int32}
    invP::Vector{Int32}
    vec::Vector{Float32}
    
    function PermsCacheEncoder(M)
        n = permsize(M)
        new(zeros(Int32, n), zeros(Int32, n), zeros(Float32, n))
    end
end

function invperm!(invP, P)
    for i in 1:length(P)
        invP[P[i]] = i
    end
 
    invP
end

###############

    include("components/binencoder.jl")
    include("components/perms.jl")
    include("components/binperms.jl")
    include("components/maxhash.jl")
    include("components/topk.jl")
    include("components/smoothedtopk.jl")

    include("references/hbe.jl")
    include("references/perms.jl")
    include("references/binperms.jl")
    include("references/nref.jl")
    include("references/walk.jl")
end