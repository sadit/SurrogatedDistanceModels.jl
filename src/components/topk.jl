export TopK

struct TopK <: AbstractSurrogate
    topk::Int
    dim::Int
   
end

distance(::TopK) = BinaryHammingDistance()
fit(::Type{TopK}, topk, dim) = TopK(topk, ceil(Int, dim / 64) * 64)

topk(T::TopK) = T.topk
dim(T::TopK) = T.dim

function encode_object!(M::TopK, out, v, res::KnnResult, tmp::BitArray)
    reuse!(res, topk(M))
    fill!(tmp, 0)
    
    for i in eachindex(v)
        push_item!(res, IdWeight(i, -v[i]))
    end
    
    for i in idview(res)
        tmp[i] = 1
    end
    
    copy!(out, tmp.chunks)
end

function predict(M::TopK, db_::AbstractDatabase; minbatch::Int=4)
    D = Matrix{UInt64}(undef, dim(M) ÷ 64, length(db_))
    R = [KnnResult(M.topk) for _ in 1:Threads.nthreads()]
    B = [BitArray(undef, dim(M)) for _ in 1:Threads.nthreads()]

    @batch per=thread minbatch=minbatch for i in eachindex(db_)
        tid = Threads.threadid()
        encode_object!(M, view(D, :, i), db_[i], R[tid], B[tid])
    end

    MatrixDatabase(D)
end

predict(M::TopK, v::AbstractVector) = encode_object!(M, Vector{UInt64}(undef, dim(M) ÷ 64, v, KnnResult(M.topk), BitArray(undef, dim(M)))

#=
function encode(M::TopK, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = encode(M, db_)
    queries = encode(M, queries_)
    params["surrogate"] = "CTopK"
    params["topk"] = topk(M)
    
    (; db, queries, params, dist)
end
=#
