module SurrogateDistanceModels

using SimilaritySearch, Random, StatsBase, LinearAlgebra, Polyester, Distributions

abstract type AbstractSurrogate end
import StatsAPI: fit, predict
export fit, predict, AbstractSurrogate


struct PermsCacheEncoder
    P::Vector{Int32}
    invP::Vector{Int32}
    vec::Vector{Float32}
    
    function PermsCacheEncoder(M)
        n = permsize(M)
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
        put!(PERMS_CACHES, PermsCacheEncoder())
    end
end

@inline function permscache(f)
    buff = take!(PERMS_CACHE)
    try
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
    include("random-projections.jl")
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
