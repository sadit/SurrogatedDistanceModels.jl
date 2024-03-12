

export GaussianRandomProjection, in_dim, out_dim

struct GaussianRandomProjection{FloatType} <: AbstractSurrogate
    map::Matrix{FloatType}
end

function fit(::Type{GaussianRandomProjection{FloatType}}, (in_dim, out_dim)::Pair) where {FloatType<:AbstractFloat}
    #(in_dim, out_dim) = map_dims
    N = Normal(0f0, 1f0 / Float32(in_dim))
    m = rand(N, in_dim, out_dim)
    GaussianRandomProjection(m)
end

distance(::GaussianRandomProjection) = L2Distance()
Base.size(rp::GaussianRandomProjection) = size(rp.map)
in_dim(rp::GaussianRandomProjection) = size(rp.map, 1)
out_dim(rp::GaussianRandomProjection) = size(rp.map, 2)


function predict!(rp::GaussianRandomProjection, out::AbstractVector, v::AbstractVector)
    @inbounds for i in 1:out_dim(rp)
        x = view(rp.map, :, i)
        out[i] = dot(x, v)
    end

    out
end

function predict(rp::GaussianRandomProjection{F}, v::AbstractVector) where {F<:AbstractFloat}
    predict!(rp, Vector{F}(undef, out_dim(rp)), v)
end


function predict(rp::GaussianRandomProjection{F}, X::AbstractMatrix) where {F}
    O = Matrix{F}(undef, out_dim(rp), size(X, 2)) 
    predict!(rp, O, MatrixDatabase(X))
end

function predict(rp::GaussianRandomProjection{F}, X::AbstractDatabase) where {F}
    O = Matrix{F}(undef, out_dim(rp), length(X))
    predict!(rp, O, X)
end

function predict!(rp::GaussianRandomProjection{F}, O::AbstractMatrix, X::AbstractDatabase; minbatch::Int=4) where {F}
    @batch per=thread minbatch=minbatch for i in 1:size(X, 2)
        o = view(O, :, i)
        predict!(rp, o, X[i]) 
    end

    O
end

