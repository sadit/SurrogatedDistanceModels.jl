export PQFFT4

# This file is a part of SimilaritySearch.jl

export StringHamming4, StringHamming2

"""
   StringHamming4()

Computes Hamming distance using 4-bit codes
"""
struct StringHamming4 <: SemiMetric end

"""
    evaluate(::StringHamming4, a, b)

Computes the hamming distance for 4-bit codes
"""
function SimilaritySearch.evaluate(::StringHamming4, A::AbstractVector{UInt64}, B::AbstractVector{UInt64})::Float32
    d = 0
    @inbounds for i in eachindex(A)
        a, b = A[i], B[i]
        for i in 1:15
            d += Bool((a & 0x000000000000000f) !== (b & 0x000000000000000f))
            a >>= 4;
            b >>= 4;
        end

        d += Bool(a !== b)
    end

    Float32(d)
end

"""
   StringHamming2()

Computes Hamming distance using 4-bit codes
"""
struct StringHamming2 <: SemiMetric end

function SimilaritySearch.evaluate(::StringHamming2, A::AbstractVector{UInt64}, B::AbstractVector{UInt64})::Float32
    d = 0
    @inbounds for i in eachindex(A)
        a, b = A[i], B[i]
        for i in 1:32
            d += (a & 0x0000000000000003) !== (b & 0x0000000000000003)
            a >>= 2;
            b >>= 2;
        end
    end

    Float32(d)
end

# these implementations are made to be correct and can be improved for the case; but first we need to know
# the algorithmic and metric parts are correct; later we can improve the low level implementation
struct H2V64{V64}
    vec::V64
end

Base.length(x::H2V64) = length(x.vec) * 32
Base.eachindex(x::H2V64) = 1:length(x)

function Base.setindex!(x::H2V64, v, index::Integer)::UInt8
    i = ((index-1) >> 5) + 1
    j = (index-1) & 0x000000000000001f
    v &= 0x0000000000000003  # cleaning extra bits; UInt64
    j <<= 1
    u = x.vec[i]
    m = 0x0000000000000003 << j
    #@show i, j, m, u, v
    u &= ~m # cleaning previous data
    #@show "PREV" string(m, base=16, pad=16) string(u, base=16, pad=16) string(v, base=16, pad=16)
    u |= (v << j) & m  # setting and cleaning
    #@show "POST" string(m, base=16, pad=16) string(u, base=16, pad=16) string(v, base=16, pad=16)
    x.vec[i] = u
    v

end

function Base.getindex(x::H2V64, index::Integer)::UInt8
    i = ((index-1) >> 5) + 1
    j = (index-1) & 0x000000000000001f # 31
    (x.vec[i] >> 2j) & 0x3
end

struct H4V64{V64}
    vec::V64
end

Base.length(x::H4V64) = length(x.vec) * 16
Base.eachindex(x::H4V64) = 1:length(x)

function Base.setindex!(x::H4V64, v, index::Integer)::UInt8
    i = ((index-1) >> 4) + 1
    j = (index-1) & 0x000000000000000f
    v &= 0x000000000000000f  # cleaning extra bits; UInt64
    j <<= 2
    u = x.vec[i]
    m = 0x000000000000000f << j
    u &= ~m # cleaning previous data
    u |= (v << j) & m  # setting and cleaning
    x.vec[i] = u
    v
end

function Base.getindex(x::H4V64, index::Integer)::UInt8
    i = ((index-1) >> 4) + 1
    j = (index-1) & 0x000000000000000f
    (x.vec[i] >> 4j) & 0xf
end

struct PQFFT4{IndexType,CTX} <: AbstractSurrogate
    refs::IndexType
    ctx::CTX
    dim::Int
    step::Int
    blocksize::Int
    outdim64::Int
end

distance(::PQFFT4) = StringHamming4()
steps(dim, step, blocksize) = 1:step:dim - blocksize - 1

function fit(::Type{PQFFT4}, dist::SemiMetric, X::AbstractDatabase; blocksize::Int=8, step=blocksize)
    @assert 0 < step <= blocksize "1 <= step <= blocksize must be followed"
    vocsize::Int = 16  # 4 bits
    m = length(X[1])
    outdim = floor(Int, (m - blocksize) / step) + 1 # fractional blocks are ignored
    outdim64 = ceil(Int, outdim / 16) # padding to u64

    refs = let v = X[1], R = [view(v, i:i+step-1) for i in steps(m, step, blocksize)]
        for j in 2:length(X)
            v = X[j]
            for i in steps(m, step, blocksize)
                push!(R, view(v, i:i+step-1))
            end
        end

        C = fft(dist, VectorDatabase(R), vocsize)
        db = MatrixDatabase(hcat(R[C.centers]...))
        @info C.centers
        ExhaustiveSearch(; db, dist)
    end

    PQFFT4(refs, GenericContext(), m, step, blocksize, outdim64)
end

vocsize(M::PQFFT4) = length(M.refs)
steps(M::PQFFT4) = steps(M.dim, M.step, M.blocksize)

function encode_object!(M::PQFFT4, out, v)
    #@info "========", steps(M), length(steps(M))
    knnres = getknnresult(1, M.ctx)
    for (i, sp) in enumerate(steps(M))
        obj = view(v, sp:sp+M.step-1)
        knnres = reuse!(knnres, 1)
        search(M.refs, M.ctx, obj, knnres)
        nn = argmin(knnres) - 1
        #@assert 0 <= nn <= 15
        out[i] = nn
        #i == 1 && display((sp, M.step), i, o, out[i])
    end
end

function predict(M::PQFFT4, db::AbstractDatabase; minbatch::Int=4)
    D = zeros(UInt64, M.outdim64, length(db))
    @batch per=thread minbatch=minbatch for i in eachindex(db)
        out = H4V64(view(D, :, i))
        encode_object!(M, out, db[i])
    end

    MatrixDatabase(D)
end

predict(M::PQFFT4, v::AbstractVector) = encode_object!(M, H4V64(zeros(UInt64, M.outdim64)), v)

#=
function encode(M::PQFFT4, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = StringHammingDistance()
    db = predict(M, db_)
    queries = predict(M, queries_)
    params["surrogate"] = "RN"
    params["vocsize"] = vocsize(M)
    params["seqsize"] = seqsize(M)
    
    (; db, queries, params, dist)
end
=#
