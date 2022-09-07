using Pkg
Pkg.activate("notebooks")
# Pkg.develop(path=".")
using SurrogateDistanceModels, HypertextLiteral, Random
include("load.jl") # load datasets
include("experiment.jl") # run tests

@show Threads.nthreads()

const DATAPATH = "comp-data/"
mkpath(DATAPATH)


function run_experiment(D, k;
        kscalelist=[1, 8, 16],
        npairslist=[256, 512, 1024, 2048],
        npoolslist=[32, 64, 128, 256],
        ssizelist=[4, 8, 16],
        topklist=[15, 31, 63],
        npermslist=[4, 8, 16],
        permsizelist=[64]
    )
    D.params["k"] = k
    D.params["enctime"] = 0.0
    Gold = test_exhaustive(DATAPATH, nothing, D.db, D.queries, D.dist, copy(D.params), k, [1])
    test_searchgraph(DATAPATH, Gold, D.db, D.queries, D.dist, copy(D.params), k, [1])
    test_searchgraph(DATAPATH, Gold, D.db, D.queries, D.dist, copy(D.params), k, [1], 0.6)

    surrogates = []
    dim = length(D.db[1])
    
    for npairs in npairslist
        push!(surrogates, BinEncoder(npairs, dim))
    end
    
    for nperms in npermslist, permsize in permsizelist
        push!(surrogates, CompPerms(nperms, dim; permsize))
        push!(surrogates, CompBinPerms(nperms, dim; permsize))
    end
    
    for samplesize in ssizelist, npools in npoolslist
        push!(surrogates, MaxHash(npools, dim; samplesize))
    end        
    
    #=for topk in topklist
        #push!(surrogates, TopKSurrogate(topk, dim))
        push!(surrogates, SmoothedTopK(topk, dim))
    end =#

    for E in surrogates
        enctime = @elapsed H = SurrogateDistanceModels.encode(E, D.db, D.queries, copy(D.params))
        H.params["enctime"] = enctime
        test_exhaustive(DATAPATH, Gold, H.db, H.queries, H.dist, copy(H.params), k, kscalelist)
        test_searchgraph(DATAPATH, Gold, H.db, H.queries, H.dist, copy(H.params), k, kscalelist, 0)
    end
end

function main()
    k=32

    let
        D = load_glove_400k()
        @show size(D.db.matrix), D.dist
        run_experiment(D, k)
    end
    
    let
        D = load_wit_300k()
        @show size(D.db.matrix), D.dist
        run_experiment(D, k)
    end

    let
        D = load_glove_1m()
        @show size(D.db.matrix), D.dist
        run_experiment(D, k)
    end

    let
        D = load_bigann_1m()
        @show size(D.db.matrix), D.dist
        run_experiment(D, k)
    end
end