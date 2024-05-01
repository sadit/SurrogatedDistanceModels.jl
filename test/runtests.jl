using SimilaritySearch, SurrogatedDistanceModels 
using Test, LinearAlgebra, StatsBase, Random, Downloads, JLD2

url = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-pca32v2-n=300K.h5"
dbfile = basename(url)

isfile(dbfile) || Downloads.download(url, dbfile)
X, Q = jldopen(dbfile) do f
    data = f["pca32"] # 32 * 32 = 1024 bits 
    pos = ceil(Int, size(data, 2) * 0.95)
    MatrixDatabase(data[:, 1:pos]), MatrixDatabase(data[:, pos+1:end])
end

k = 16
O = ExhaustiveSearch(dist=L2Distance(), db=X)
knns, _ = searchbatch(O, Q, k)


@testset "Random projection" begin
    idim = length(X[1])
    odim = 8

    rp = fit(GaussianRandomProjection{Float32}, idim => odim) # 8 * 32 = 2^8 bits
    X̂ = predict(rp, X)
    Q̂ = predict(rp, Q)

    O_ = ExhaustiveSearch(dist=distance(rp), db=X̂)
    knns_, _ = searchbatch(O_, Q̂, k)

    r = macrorecall(knns, knns_)
    @info "recall $(typeof(rp)) $r"
    @test r > 0.05
end

@testset "PCA projection" begin
    idim = length(X[1])
    odim = 8

    p = fit(PCAProjection, X, odim) # 8 * 32  = 2^8 bits 
    X̂ = predict(p, X)
    Q̂ = predict(p, Q)

    O_ = ExhaustiveSearch(dist=distance(p), db=X̂)
    knns_, _ = searchbatch(O_, Q̂, k)

    r = macrorecall(knns, knns_)
    @info "recall $(typeof(p)): $r"
    @test r > 0.10
end

@testset "Perms" begin
    idim = length(X[1])

    p = fit(Perms, SqL2Distance(), rand(X, 128), 2; permsize=8) # 8 * 2 * 32 = 2^9 bits
    X̂ = predict(p, X)
    Q̂ = predict(p, Q)

    O_ = ExhaustiveSearch(dist=distance(p), db=X̂)
    knns_, _ = searchbatch(O_, Q̂, k)

    r = macrorecall(knns, knns_)
    @info "recall $(typeof(p)): $r"
    @test r > 0.04
end

@testset "BinPerms" begin
    idim = length(X[1])

    p = fit(BinPerms, SqL2Distance(), rand(X, 128), 256) # 256 bits
    X̂ = predict(p, X)
    Q̂ = predict(p, Q)

    O_ = ExhaustiveSearch(dist=distance(p), db=X̂)
    knns_, _ = searchbatch(O_, Q̂, k)

    r = macrorecall(knns, knns_)
    @info "recall $(typeof(p)): $r"
    @test r > 0.25
end

@testset "HyperplaneEncoding" begin
    idim = length(X[1])

    p = fit(HyperplaneEncoding, SqL2Distance(), MatrixDatabase(rand(X, 512)), 256) #  256 bits
    X̂ = predict(p, X)
    Q̂ = predict(p, Q)

    O_ = ExhaustiveSearch(dist=distance(p), db=X̂)
    knns_, _ = searchbatch(O_, Q̂, k)

    r = macrorecall(knns, knns_)
    @info "recall $(typeof(p)): $r"
    @test r > 0.10
end

@testset "DistantHyperplanes" begin
    idim = length(X[1])

    p = fit(DistantHyperplanes, SqL2Distance(), X, 256; verbose=false) # 256 bits
    X̂ = predict(p, X)
    Q̂ = predict(p, Q)

    O_ = ExhaustiveSearch(dist=distance(p), db=X̂)
    knns_, _ = searchbatch(O_, Q̂, k)

    r = macrorecall(knns, knns_)
    @info "recall $(typeof(p)): $r"
    @test r > 0.33
end

#=
@testset "Nearest references" begin
    idim = length(X[1])

    p = fit(NearestReference, SqL2Distance(), rand(X, 32), 8; permsize=4) # 256 bits
    X̂ = predict(p, X)
    Q̂ = predict(p, Q)
    display(Q̂)
    O_ = ExhaustiveSearch(dist=distance(p), db=X̂)
    knns_, _ = searchbatch(O_, Q̂, k)

    r = macrorecall(knns, knns_)
    @info "recall $(typeof(p)): $r"
    @test r > 0.33
end
=#
@testset "Binary Permutations - Random walk" begin
    idim = length(X[1])

    p = fit(BinPermsDiffEnc, SqL2Distance(), rand(X, 256), 256) # 256 bits
    X̂ = predict(p, X)
    Q̂ = predict(p, Q)
    O_ = ExhaustiveSearch(dist=distance(p), db=X̂)
    knns_, _ = searchbatch(O_, Q̂, k)

    r = macrorecall(knns, knns_)
    @info "recall $(typeof(p)): $r"
    @test r > 0.23
end
