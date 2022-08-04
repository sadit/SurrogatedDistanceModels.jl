using JLD2, LinearAlgebra, SimilaritySearch

const DBPATH = "../metric-datasets/"

function pack_cos_db(train, test, name)
    for c in eachcol(train) normalize!(c) end
    for c in eachcol(test) normalize!(c) end
    db = MatrixDatabase(train)
    queries = MatrixDatabase(test)
    params = Dict("name" => name, "dim" => size(train, 1), "n" => size(train, 2), "m" => size(test, 2))
    (; db, queries, params, dist=NormalizedCosineDistance())
end

function load_glove_400k()
    train, test = load(joinpath(DBPATH, "glove-400k-en-100d-angular.h5"), "train", "test")
    pack_cos_db(train, test, "Glove-400K")
end

function load_glove_1m()
    train, test = load(joinpath(DBPATH, "glove-100-angular.hdf5"), "train", "test")
    pack_cos_db(train, test, "Glove-1M")
end

function load_wit_300k()
    train, test = load(joinpath(DBPATH, "WIT-Clip-angular.h5"), "train", "test")
    pack_cos_db(train, test, "WIT-300K")
end

function load_bigann_1m()
    train, test = load(joinpath(DBPATH, "bigann_1M.h5"), "train", "test")
    db = MatrixDatabase(Matrix{Float32}(train))
    queries = MatrixDatabase(Matrix{Float32}(test))
    params = Dict("name" => "BigANN-1M", "dim" => size(train, 1), "n" => size(train, 2), "m" => size(test, 2))
    (; db, queries, params, dist=SqL2Distance())
end
