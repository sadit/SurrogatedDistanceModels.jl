using Pkg
Pkg.activate("notebooks")
# Pkg.develop(path=".")
using SurrogateDistanceModels, Random, SimilaritySearch, FileIO, HDF5, OrderedCollections
include("load.jl") # load datasets

@show Threads.nthreads()

const DATAPATH = "shared-data/"
mkpath(DATAPATH)

function goldstandard(filename, D, k)
    isfile(filename) && return
    E = ExhaustiveSearch(; dist=D.dist, db=D.db)
    elapsed = @elapsed I, D = searchbatch(E, D.queries, k)
    save(filename, OrderedDict("I" => I, "D" => D, "searchtime" => elapsed))
end

function transform(path, D, npermslist, permsizelist)
    for nperms in npermslist, permsize in permsizelist
        R = select_random_refs(D.db.matrix, nperms * permsize)
        let
            E = BinPerms(D.dist, R, nperms; permsize)
            enctime = @elapsed H = SurrogateDistanceModels.encode(E, D.db, D.queries, copy(D.params))
            save(path * "--surrogate=RBP--nperms=$nperms--permsize=$permsize.h5",
                OrderedDict("train" => H.db.matrix, "test" => H.queries.matrix, "enctime" => enctime))
        end

        let
            E = BinWalk(D.dist, R; permsize)
                enctime = @elapsed H = SurrogateDistanceModels.encode(E, D.db, D.queries, copy(D.params))
                save(path * "--surrogate=RBW--nperms=$nperms--permsize=$permsize.h5",
                    OrderedDict("train" => H.db.matrix, "test" => H.queries.matrix, "enctime" => enctime))
        end

    end
end

function main()
    k = 32
    npermslist = [4, 8, 16, 32]
    permsizelist = [64]

    let
        D = load_glove_400k()
        @show size(D.db.matrix), D.dist
        goldstandard(joinpath(DATAPATH, "gold-standard--glove-400k--k=$k.h5"), D, k)
        transform(joinpath(DATAPATH, "bitmap--glove-400k"), D, npermslist, permsizelist)
    end

    let
        D = load_wit_300k()
        @show size(D.db.matrix), D.dist
        goldstandard(joinpath(DATAPATH, "gold-standard--wit-400k--k=$k.h5"), D, k)
        transform(joinpath(DATAPATH, "bitmap--with-300k"), D, npermslist, permsizelist)
    end

    let
        D = load_glove_1m()
        @show size(D.db.matrix), D.dist
        goldstandard(joinpath(DATAPATH, "gold-standard--glove-1m--k=$k.h5"), D, k)
        transform(joinpath(DATAPATH, "bitmap--glove-1m"), D, npermslist, permsizelist)
    end

    let
        D = load_bigann_1m()
        @show size(D.db.matrix), D.dist
        goldstandard(joinpath(DATAPATH, "gold-standard--bigann-1m--k=$k.h5"), D, k)
        transform(joinpath(DATAPATH, "bitmap--bigann-1m"), D, npermslist, permsizelist)
    end
end