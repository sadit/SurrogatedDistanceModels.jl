module SurrogatedDistanceModels

using SimilaritySearch, Random, StatsBase, LinearAlgebra, Polyester, Distributions
using SimilaritySearch: evaluate

abstract type AbstractSurrogate end
import StatsAPI: fit, predict
import SimilaritySearch: distance
export fit, predict, AbstractSurrogate


struct PermsCacheEncoder
    P::Vector{Int32}
    invP::Vector{Int32}
    vec::Vector{Float32}
    
    function PermsCacheEncoder(n::Integer)
        new(zeros(Int32, n), zeros(Int32, n), zeros(Float32, n))
    end
end

const PERMS_CACHES = Channel{PermsCacheEncoder}(Inf)

function Base.empty!(buff::PermsCacheEncoder)
    empty!(buff.P)
    empty!(buff.invP)
    empty!(buff.vec)
end
   
function __init__()
    for _ in 1:2*Threads.nthreads()+4
        put!(PERMS_CACHES, PermsCacheEncoder(32))
    end
end

@inline function permscache(f, n::Integer)
    buff = take!(PERMS_CACHE)
    try
        resize!(buff.P, n)
        resize!(buff.invP, n)
        resize!(buff.vec, n)

        return f(buff) 
    finally
        put!(PERMS_CACHE, buff)
    end
end

function invperm!(invP, P)
    for i in 1:length(P)
        invP[P[i]] = i
    end
 
    invP
end

###############
    include("rp.jl")
    include("pca.jl")
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
    include("references/highentropy.jl")
end
