using MultivariateStats: PCA
export PCAProjection

struct PCAProjection{PCA_<:PCA} <: AbstractSurrogate
    pca::PCA_
end

fit(::Type{PCAProjection}, train::MatrixDatabase, maxoutdim::Int) = fit(PCAProjection, train.matrix, maxoutdim)
fit(::Type{PCAProjection}, train::SubDatabase, maxoutdim::Int) = fit(PCAProjection, view(train.parent, train.map), maxoutdim)

function fit(::Type{PCAProjection}, train::AbstractMatrix, maxoutdim::Int)
    pca = fit(PCA, train; maxoutdim)
    PCAProjection(pca)
end

distance(::PCAProjection) = L2Distance()

function predict(pca::PCAProjection, X::MatrixDatabase)
    predict(pca.pca, X.matrix) |> MatrixDatabase
end

function predict(pca::PCAProjection, X::SubDatabase)
    predict(pca.pca, view(X.parent, :, X.map)) |> MatrixDatabase
end

function predict(pca::PCAProjection, X::AbstractVector)
    predict(pca.pca, x)
end

