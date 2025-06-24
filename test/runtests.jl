using SimilaritySearch, SurrogatedDistanceModels 
using Test, LinearAlgebra, StatsBase, Random, Downloads, HDF5

url = "https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-ccnews-fp16.h5?download=true"
dbfile = first(split(basename(url), '?'))

isfile(dbfile) || Downloads.download(url, dbfile)

k = 16
X, Q, knns = h5open(dbfile) do f
    MatrixDatabase(f["train"][]), MatrixDatabase(f["itest/queries"][]), f["itest/knns"][1:k, :]
end
ctx = GenericContext() # shared context 

#=
@testset "Nearest references" begin
    idim = length(X[1])

    p = fit(NearestReference, NormalizedCosine_asf32(), rand(X, 2048); vocsize=16, seqsize=128) # 256 bits
    X̂ = predict(p, X)
    Q̂ = predict(p, Q)
    O_ = ExhaustiveSearch(dist=distance(p), db=X̂)
    knns_, _ = searchbatch(O_, ctx, Q̂, k)

    r = macrorecall(knns, knns_)
    @info "recall $(typeof(p)): $r"
    @test r > 0.33
end
=#


@testset "PQFFT4" begin
    idim = length(X[1])
    p = fit(PQFFT4, SqL2Distance(), SubDatabase(X, 1:2^14); blocksize=2) # 256 bits
    #display(p.refs.db.matrix)
    X̂ = predict(p, X)
    Q̂ = predict(p, Q)
    QQ = predict(p, SubDatabase(Q, [1, 2, 3]))
    #display(knns[:, 1])
    #display(Q̂[1])
    #display(QQ[1])
    #display(evaluate(distance(p), Q̂[1], QQ[1]))
    O_ = ExhaustiveSearch(dist=distance(p), db=X̂)
    knns_, _ = searchbatch(O_, ctx, Q̂, k)
    
    #display(Q̂[1])
    #display(X̂[26711])
    #display(evaluate(O_.dist, Q̂[1], X̂[26711]))
    #display(evaluate(O_.dist, Q̂[1], X̂[26713]))

    r = macrorecall(Set.(eachcol(knns)), Set.(eachcol(knns_)))
    @info "recall $(typeof(p)): $r"
    @test r > 0.33
end


@testset "Random projection" begin
    idim = length(X[1])
    odim = 8

    rp = fit(GaussianRandomProjection{Float32}, idim => odim) # 8 * 32 = 2^8 bits
    X̂ = predict(rp, X)
    Q̂ = predict(rp, Q)

    O_ = ExhaustiveSearch(dist=distance(rp), db=X̂)
    knns_, _ = searchbatch(O_, ctx, Q̂, k)

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
    knns_, _ = searchbatch(O_, ctx, Q̂, k)

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
    knns_, _ = searchbatch(O_, ctx, Q̂, k)

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
    knns_, _ = searchbatch(O_, ctx, Q̂, k)

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
    knns_, _ = searchbatch(O_, ctx, Q̂, k)

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
    knns_, _ = searchbatch(O_, ctx, Q̂, k)

    r = macrorecall(knns, knns_)
    @info "recall $(typeof(p)): $r"
    @test r > 0.33
end



@testset "Binary Permutations - Random walk" begin
    idim = length(X[1])

    p = fit(BinPermsDiffEnc, SqL2Distance(), rand(X, 256), 256) # 256 bits
    X̂ = predict(p, X)
    Q̂ = predict(p, Q)
    O_ = ExhaustiveSearch(dist=distance(p), db=X̂)
    knns_, _ = searchbatch(O_, ctx, Q̂, k)

    r = macrorecall(knns, knns_)
    @info "recall $(typeof(p)): $r"
    @test r > 0.23
end
