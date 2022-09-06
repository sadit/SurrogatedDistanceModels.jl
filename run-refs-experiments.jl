using Pkg
Pkg.activate("notebooks")
# Pkg.develop(path=".")
using SurrogateDistanceModels, HypertextLiteral, Random
include("load.jl") # load datasets
include("experiment.jl") # run tests

@show Threads.nthreads()

const DATAPATH = "refs-data/"
mkpath(DATAPATH)

function run_experiment(D, k;
        kscalelist=[1, 8, 16],
        npairslist=[64, 128, 256, 512, 1024],
        npoolslist=[32, 64, 128, 256],
        nnpermsizelist=[4, 8, 16, 32],
        topklist=[15, 31, 63],
        npermslist=[4, 8, 16, 32],
        permsizelist=[64]
    )
    D.params["k"] = k
    D.params["enctime"] = 0.0
    Gold = test_exhaustive(DATAPATH, nothing, D.db, D.queries, D.dist, copy(D.params), k, [1])
    test_searchgraph(DATAPATH, Gold, D.db, D.queries, D.dist, copy(D.params), k, [1])
    test_searchgraph(DATAPATH, Gold, D.db, D.queries, D.dist, copy(D.params), k, [1], 0.6)

    surrogates = []

    for npairs in npairslist
        push!(surrogates, HyperplaneBinaryEncoding(D.dist, D.db, npairs))
    end
    
    for nperms in npermslist, permsize in permsizelist
        R = select_random_refs(D.db.matrix, nperms * permsize)
        push!(surrogates, Perms(D.dist, R, nperms; permsize))
        push!(surrogates, BinPerms(D.dist, R, nperms; permsize))
        push!(surrogates, BinWalk(D.dist, R; permsize))
    end

    for permsize in nnpermsizelist, npools in npoolslist
        R = select_random_refs(D.db.matrix, permsize * npools)
        push!(surrogates, NearestReference(D.dist, R; permsize))
    end 

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