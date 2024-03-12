using MultivariateStats
export PCAProjection

struct PCAProjection{PCA_<:PCA} <: AbstractSurrogate
    pca::PCA_
end


function fit(::Type{PCAProjection{FloatType}}, train::AbstractMatrix, maxoutdim::Int) where {FloatType<:AbstractFloat}
    pca = fit(PCA, train; maxoutdim)
    PCAProjection(pca)
end

distance(::PCAProjection) = L2Distance()

function predict(pca::PCAProjection{F}, X::MatrixDatabase) where {F<:AbstractFloat}
    predict(pca.pca, X.matrix) |> MatrixDatabase
end

function predict(pca::PCAProjection{F}, X::SubDatabase) where {F<:AbstractFloat}
    predict(pca.pca, view(X.parent, :, X.map)) |> MatrixDatabase
end

function predict(pca::PCAProjection{F}, X::AbstractVector) where {F}
    predict(pca.pca, x)
end

